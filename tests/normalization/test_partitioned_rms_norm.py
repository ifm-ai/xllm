"""Check GroupRMSNorm with partitioned_output against an fp32 full-width reference and the gathered norm.

Each rank normalises its D/TP slice with one per-token all-reduce; the slices of the output, the input gradient and
the weight gradient must match the full-width norm. It also runs the gathered GroupRMSNorm the way the MoE layer used
it and compares the two (--compare_gathered, on by default). It runs once per input RMS in --x_scales: the default
small one (1e-3) makes eps matter. Needs the built xllm_extension (rms_norm.py imports it). One TP group spans all
processes. Launch one process per rank:

    torchrun --nproc_per_node 8 tests/normalization/test_partitioned_rms_norm.py
    torchrun --nproc_per_node 2 tests/normalization/test_partitioned_rms_norm.py --affine False --dtype float32

Exits with status 1 if an error is not finite or not below its tolerance: against the fp32 reference 1e-5 (fp32) or
1e-2 (bf16), against the gathered norm 1e-6 or 1e-3. Every error printed is the worst over ranks.
"""
import math
import os
import sys

import fire
import torch

from xllm.distributed import (
    initialize_model_parallel,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.modules.rms_norm import GroupRMSNorm

TOLERANCE = {'float32': 1e-5, 'bfloat16': 1e-2}  # against the fp32 reference
TOLERANCE_GATHERED = {'float32': 1e-6, 'bfloat16': 1e-3}  # against the gathered norm: rounding-level differences


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def run_norm(norm: GroupRMSNorm, x: torch.Tensor, y_grad: torch.Tensor, part: slice):
    """Forward and backward on this rank's slice of x; returns this rank's slices of y, x grad and weight grad."""
    xs = x[..., part].contiguous().requires_grad_()
    y = norm(xs)
    if y.size(-1) != xs.size(-1):  # the gathered norm returns all features; the MoE layer kept its own slice
        y.backward(y_grad)
        y = y[..., part]
    else:
        y.backward(y_grad[..., part].contiguous())
    w_grad = norm.weight.grad if norm.weight is not None else None
    return y.detach(), xs.grad, w_grad


def check(bsz: int, seq_len: int, model_dim: int, eps: float, affine: bool, dtype: str, compare_gathered: bool,
          x_scale: float) -> bool:
    """One comparison; returns whether it failed on any rank."""
    rank, tp = get_model_parallel_rank(), get_model_parallel_world_size()
    dt = getattr(torch, dtype)

    # the same full tensors on every rank; each rank takes its slice of the features
    gen = torch.Generator(device='cuda').manual_seed(0)
    x = (torch.randn(bsz, seq_len, model_dim, device='cuda', generator=gen) * x_scale).to(dt)
    w = torch.randn(model_dim, device='cuda', generator=gen) * 0.1
    y_grad = torch.randn(bsz, seq_len, model_dim, device='cuda', generator=gen).to(dt)
    part = slice(rank * model_dim // tp, (rank + 1) * model_dim // tp)

    # fp32 reference on the full features (GroupRMSNorm applies weight + 1)
    xr = x.float().clone().requires_grad_()
    wr = w.clone().requires_grad_()
    yr = xr * torch.rsqrt(xr.square().mean(dim=-1, keepdim=True) + eps)
    if affine:
        yr = yr * (wr + 1.0)
    yr.backward(y_grad.float())

    def build(partitioned: bool) -> GroupRMSNorm:
        norm = GroupRMSNorm(model_dim, num_groups=1, eps=eps, elementwise_affine=affine,
                            disable_input_reduce=True, partitioned_output=partitioned).cuda()
        if affine:
            norm.weight.data = w[part].to(dt).contiguous()
        return norm

    if tp == 1:
        if rank == 0:
            print("tp=1: partitioned_output is off, the norm keeps its unpartitioned path", flush=True)
        return False
    norm = build(True)
    assert norm.partitioned_output and not norm.gather_input
    y, xg, wg = run_norm(norm, x, y_grad, part)
    errs = {'y': rel_err(y, yr[..., part]), 'x_grad': rel_err(xg, xr.grad[..., part])}
    if affine:
        errs['w_grad'] = rel_err(wg, wr.grad[part])
    if compare_gathered:
        gathered = build(False)
        assert gathered.gather_input
        yg, xgg, wgg = run_norm(gathered, x, y_grad, part)
        errs.update({'y_vs_gathered': rel_err(y, yg), 'x_grad_vs_gathered': rel_err(xg, xgg)})
        errs['gathered_y'] = rel_err(yg, yr[..., part])
        if affine:
            errs['w_grad_vs_gathered'] = rel_err(wg, wgg)
    # a NaN may not survive a MAX all-reduce, so each rank also reports whether its errors are finite
    nonfinite = torch.tensor([float(not all(math.isfinite(v) for v in errs.values()))], device='cuda')
    worst = torch.tensor(list(errs.values()), device='cuda')
    torch.distributed.all_reduce(worst, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(nonfinite, op=torch.distributed.ReduceOp.MAX)
    errs = dict(zip(errs, worst.tolist()))  # each error is now the worst over ranks
    bad = [k for k, v in errs.items() if not v < (TOLERANCE_GATHERED if k.endswith('vs_gathered') else TOLERANCE)[dtype]]
    if nonfinite.item():
        bad.append('non-finite error on some rank')
    failed = bool(bad)
    if rank == 0:
        print(f"tp={tp} {dtype} affine={affine} tokens={bsz * seq_len} dim={model_dim} x_scale={x_scale}, worst over ranks: "
              + ' '.join(f'{k}={v:.2e}' for k, v in errs.items())
              + (f" FAIL {bad}" if failed else " ok"), flush=True)
    return failed


def main(
    bsz: int = 2,
    seq_len: int = 1024,
    model_dim: int = 2048,
    eps: float = 1e-6,
    affine: bool = True,
    dtype: str = 'bfloat16',
    compare_gathered: bool = True,
    x_scales: tuple = (3.0, 1e-3),
):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    initialize_model_parallel(torch.distributed.get_world_size(), 1, timeout=600)
    failed = False
    for x_scale in (x_scales if isinstance(x_scales, (tuple, list)) else (x_scales,)):
        failed |= check(bsz, seq_len, model_dim, eps, affine, dtype, compare_gathered, float(x_scale))
    torch.distributed.destroy_process_group()
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    fire.Fire(main)
