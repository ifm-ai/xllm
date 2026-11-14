"""Check the unfused MoE under FSDP2 when a rank's experts receive no rows.

Runs NormalizedMoE with TP 2 and FSDP2 over DP 2 (4 processes). On data-parallel rank 1 the router bias sends
every token to TP rank 0's experts, so TP rank 1's experts get no rows there while they do on data-parallel
rank 0. FSDP2 reduce-scatters only the parameters that have a gradient, so every rank must give its experts a
gradient (zero if no rows), or the two data-parallel ranks reduce different parameter sets. Without FSDP, the
experts without rows must get gradients that are exactly zero; with FSDP, the gradients are compared with the same
layer run without FSDP and averaged over data-parallel ranks.

    torchrun --nproc_per_node 4 tests/moe/test_moe_fsdp_empty_rows.py
"""
import copy
import os
from datetime import timedelta

import fire
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from xllm.distributed import (
    initialize_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.modules.moe.moe import NormalizedMoE


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12)).item()


def main(
    backend: str = 'triton',
    bsz: int = 2,
    seq_len: int = 64,
    model_dim: int = 256,
    inter_dim: int = 64,
    n_experts: int = 16,
    topk: int = 4,
    seed: int = 0,
    timeout: int = 120,
    tol: float = 2e-2,
):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', timeout=timedelta(seconds=timeout))
    initialize_model_parallel(2, 1, timeout=timeout)
    assert get_data_parallel_world_size() == 2, "run with 4 processes: TP 2 x DP 2"
    rank, dp_rank = get_model_parallel_rank(), get_data_parallel_rank()
    world_size = get_model_parallel_world_size()
    n_local = n_experts // world_size
    assert topk <= n_local

    # same weights on both data-parallel ranks
    torch.manual_seed(seed)
    moe = NormalizedMoE(
        0, model_dim, inter_dim, n_experts, topk, 1, expert_backend='sequential', permutation_backend=backend,
        router_score_func='sigmoid', router_bias=True, router_bias_update_rate=1e-3, router_scaling_factor=2.5,
        apply_rmsnorm=True,
    ).cuda().bfloat16()
    bias = torch.zeros(n_experts, device='cuda')
    if dp_rank == 1:
        bias[:n_local] = 10.0  # every token goes to TP rank 0's experts
    ref = copy.deepcopy(moe)

    # different tokens on each data-parallel rank; each TP rank holds a slice of the model dimension
    torch.manual_seed(seed + 1 + dp_rank)
    x = torch.randn(bsz, seq_len, model_dim, device='cuda').bfloat16()
    out_grad = torch.randn(bsz, seq_len, model_dim, device='cuda').bfloat16()
    dim = model_dim // world_size
    x, out_grad = (t[..., rank * dim:(rank + 1) * dim].contiguous() for t in (x, out_grad))

    def run(module):
        rows = {}
        module.router.bias.copy_(bias)
        module.router.register_forward_hook(
            lambda _, __, out: rows.update(n=int(out[2][rank * n_local:(rank + 1) * n_local].sum()))
        )
        out, _ = module(x)
        out.backward(out_grad)
        torch.cuda.synchronize()
        return rows['n']

    # reference: no FSDP, gradients averaged over the data-parallel ranks
    n_rows = run(ref)
    expected, missing_ref = {}, []
    for name, p in ref.named_parameters():
        if p.grad is None:
            missing_ref.append(name)
        g = p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
        torch.distributed.all_reduce(g, group=get_data_parallel_group())
        expected[name] = g / get_data_parallel_world_size()
    print(f"dp={dp_rank} tp={rank}: {n_rows} local rows; params without grad (no FSDP): {missing_ref or 'none'}",
          flush=True)
    # the reference runs the same layer, so check on its own that experts without rows get zero gradients
    failed = False
    if dp_rank == 1 and rank == 1 and n_rows != 0:
        failed = True
        print(f"dp={dp_rank} tp={rank}: expected 0 local rows (the router bias no longer empties this rank) FAIL",
              flush=True)
    if n_rows == 0:
        zero = {name: p.grad is not None and not p.grad.any() for name, p in ref.experts.named_parameters()}
        failed = not all(zero.values())
        print(f"dp={dp_rank} tp={rank} expert grads with no rows are zero (no FSDP): {zero} "
              f"{'FAIL' if failed else 'ok'}", flush=True)

    mesh = DeviceMesh.from_group(get_data_parallel_group(), 'cuda', mesh_dim_names=('dp',))
    fully_shard(moe, mesh=mesh)
    run(moe)

    params = list(moe.named_parameters())
    # gathering a gradient is a collective: only gather those that every rank has
    has_grad = torch.tensor([p.grad is not None for _, p in params], device='cuda', dtype=torch.int32)
    everywhere = has_grad.clone()
    torch.distributed.all_reduce(everywhere, op=torch.distributed.ReduceOp.MIN)
    for (name, p), here, all_ranks in zip(params, has_grad.tolist(), everywhere.tolist()):
        if not all_ranks:
            ok = False
            msg = 'grad=None' if not here else 'grad=None on another rank'
        else:
            got = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            err = rel_err(got, expected[name])
            ok = err < tol
            msg = f'rel_err={err:.2e}'
        failed |= not ok
        print(f"dp={dp_rank} tp={rank} {name:18s} {msg} {'ok' if ok else 'FAIL'}", flush=True)

    torch.distributed.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
