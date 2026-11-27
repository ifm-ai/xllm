"""Time the fused MoE forward + backward and count the rows each rank sends to the others.

Launch one process per TP rank, e.g. a 36B-like layer at TP 2:

    torchrun --nproc_per_node 2 tests/moe/test_dedup_dispatch_speed.py \
        --model_dim 2560 --n_experts 100 --inter_dim 768 --topk 8 --tokens 65536

It also runs on main, for a baseline.
"""
import os

import fire
import torch

from xllm.distributed import (
    initialize_model_parallel,
    get_model_parallel_group,
    get_model_parallel_rank,
)
from xllm.models.fused_blocks.moe import moe_fwd, moe_bwd


def main(
    model_dim: int = 2560,
    n_experts: int = 100,
    inter_dim: int = 768,
    topk: int = 8,
    tokens: int = 65536,
    expert_backend: str = 'torch',
    permutation_backend: str = 'triton',
    warmup: int = 3,
    iters: int = 10,
):
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    initialize_model_parallel(world_size, 1, timeout=600)
    rank = get_model_parallel_rank()
    dim = model_dim // world_size
    n_local = n_experts // world_size

    torch.manual_seed(rank)
    xf = torch.randn(1, tokens, dim, device='cuda', dtype=torch.bfloat16)
    out_grad = torch.randn_like(xf)
    router_w = torch.randn(n_experts, dim, device='cuda', dtype=torch.bfloat16) / model_dim ** 0.5
    ew1 = torch.randn(n_local * inter_dim, model_dim, device='cuda', dtype=torch.bfloat16) / model_dim ** 0.5
    ew2 = torch.randn(n_local * model_dim, inter_dim, device='cuda', dtype=torch.bfloat16) / inter_dim ** 0.5
    ew3 = torch.randn(n_local * inter_dim, model_dim, device='cuda', dtype=torch.bfloat16) / model_dim ** 0.5

    def step():
        _, _, routing_out, _, routing_state, rng_states = moe_fwd(
            xf, router_w, None, 'sigmoid', 2.5, None, None, None, None,
            n_experts, n_local, rank * n_local, (rank + 1) * n_local, topk,
            ew1, ew2, ew3, expert_backend, permutation_backend, 0.0, False, False,
        )
        moe_bwd(
            out_grad, None, xf, xf.view(-1, dim), *routing_out,
            router_w, None, None, 'sigmoid', 2.5, None, None, None, None, None, None,
            n_experts, n_local, topk, ew1, ew2, ew3, expert_backend, permutation_backend, 0.0,
            *routing_state, *rng_states, False,
        )
        return routing_state

    for _ in range(warmup):
        step()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.distributed.barrier(group=get_model_parallel_group())
        start.record()
        routing_state = step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # rows sent to other ranks per dispatch, one per (token, rank); main has no per-rank counts, so use its
    # input splits there, one row per (token, expert)
    sent = routing_state[4] if len(routing_state) > 4 else routing_state[1]
    stats = torch.tensor(
        [sorted(times)[len(times) // 2], float(sum(sent) - sent[rank])], device='cuda', dtype=torch.float64
    )
    torch.distributed.all_reduce(stats, group=get_model_parallel_group())
    stats /= world_size
    if torch.distributed.get_rank() == 0:
        mb = stats[1].item() * dim * 2 / 2 ** 20
        print(f"tp={world_size} dim={model_dim} experts={n_experts} topk={topk} tokens={tokens}: "
              f"fwd+bwd {stats[0].item():.1f} ms, {stats[1].item():.0f} rows ({mb:.0f} MiB) "
              f"sent to other ranks per dispatch")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
