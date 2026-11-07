"""Check the CuTe DSL timestep decay norm forward against an independent FP64 PyTorch reference.

Run from the repository root:
    python -m pytest tests/normalization/test_cute_timestep_decay_norm.py

The kernel module is loaded from its file so that the test does not need the compiled
xllm_extension that the xllm.modules package imports.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[2] / "xllm/modules/fused_ops/cute/timestep_decay_norm.py"
_spec = importlib.util.spec_from_file_location("cute_timestep_decay_norm", _PATH)
tsdn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsdn)

BETA1, BETA2, EPS = 0.999, 0.9999, 1e-5


def reference_fwd(x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta, num_groups, padding_mask):
    """FP64 time loop: returns y, count, mean, var, cummean [B, G, L], cumrstd [B, G, L], step count [B, L]."""
    B, L, H = x.shape
    G = num_groups
    xd = x.double().view(B, L, G, H // G)
    mt = xd.mean(-1)
    vt = xd.var(-1, correction=0)
    bos = bos_mask if bos_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=x.device)
    pad = padding_mask if padding_mask is not None else torch.zeros(B, L, dtype=torch.bool, device=x.device)
    count = prev_count.clone()
    M = prev_mean.double()
    V = prev_var.double()
    cms, crs, cnts = [], [], []
    for t in range(L):
        live = (~pad[:, t]).view(B, 1)
        reset = (bos[:, t] & ~pad[:, t]).view(B, 1)
        M = torch.where(live, torch.where(reset, 0.0, BETA1) * M + (1 - BETA1) * mt[:, t], M)
        V = torch.where(live, torch.where(reset, 0.0, BETA2) * V + (1 - BETA2) * vt[:, t], V)
        count = torch.where(live.view(B), torch.where(reset.view(B), 1, count + 1), count)
        c = count.double().view(B, 1)
        ok = c > 0
        cms.append(torch.where(ok, M / (1 - BETA1 ** c).clamp_min(1e-300), 0.0))
        crs.append(torch.where(ok, torch.rsqrt(V / (1 - BETA2 ** c).clamp_min(1e-300) + EPS), EPS ** -0.5))
        cnts.append(count.clone())
    cummean = torch.stack(cms, dim=2)
    cumrstd = torch.stack(crs, dim=2)
    xn = (xd - cummean.transpose(1, 2).unsqueeze(-1)) * cumrstd.transpose(1, 2).unsqueeze(-1)
    y = (xn.reshape(B, L, H) * gamma.double() + beta.double()).masked_fill(pad.unsqueeze(-1), 0.0)
    return y, count, M, V, cummean, cumrstd, torch.stack(cnts, dim=1)


def make_inputs(B, L, H, G, dtype, bos_ratio, pad_ratio, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(B, L, H, device="cuda", generator=g) * 2 + 0.5).to(dtype)
    bos = torch.rand(B, L, device="cuda", generator=g) < bos_ratio if bos_ratio > 0 else None
    pad = torch.rand(B, L, device="cuda", generator=g) < pad_ratio if pad_ratio > 0 else None
    prev_count = torch.randint(0, 50, (B,), device="cuda", generator=g)
    prev_mean = (torch.randn(B, G, device="cuda", generator=g) * 0.1).to(dtype)
    prev_var = (torch.rand(B, G, device="cuda", generator=g) * 0.1).to(dtype)
    gamma = torch.randn(H, device="cuda", generator=g).to(dtype)
    beta = torch.randn(H, device="cuda", generator=g).to(dtype)
    return x, bos, prev_count, prev_mean, prev_var, gamma, beta, pad


def rel_err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


CASES = [  # B, L, H, G, dtype, bos_ratio, pad_ratio, config
    (2, 128, 256, 4, torch.float32, 0.0, 0.0, None),
    (3, 300, 256, 4, torch.float32, 0.02, 0.0, None),  # L not a multiple of the tile length
    (2, 517, 512, 8, torch.float32, 0.02, 0.1, None),
    (2, 517, 512, 16, torch.float32, 0.01, 0.1, None),
    (2, 1000, 256, 32, torch.bfloat16, 0.01, 0.05, None),  # several groups per tile
    (1, 9000, 1024, 16, torch.bfloat16, 0.0005, 0.0, None),  # > 32 segments: multi-window look-back
    (2, 4100, 2048, 32, torch.bfloat16, 0.001, 0.01, None),
    (2, 700, 512, 8, torch.bfloat16, 0.01, 0.0, (2, 64, 128, 4, 2, 1)),
    (2, 700, 512, 8, torch.bfloat16, 0.01, 0.0, (1, 256, 256, 2, 2, 1)),
    (3, 2000, 512, 8, torch.bfloat16, 0.01, 0.05, (1, 64, 64, 4, 1, 1)),
    (2, 3000, 512, 8, torch.bfloat16, 0.01, 0.05, (1, 128, 128, 1, 2, 2)),  # lag 2
    (2, 3000, 512, 8, torch.bfloat16, 0.01, 0.05, (1, 64, 64, 2, 1, 3)),  # lag 3
    (2, 333, 512, 8, torch.float16, 0.01, 0.05, None),
]


@pytest.mark.parametrize("B,L,H,G,dtype,bos_ratio,pad_ratio,config", CASES)
def test_forward_matches_reference(B, L, H, G, dtype, bos_ratio, pad_ratio, config):
    x, bos, pc, pm, pv, w, b, pad = make_inputs(B, L, H, G, dtype, bos_ratio, pad_ratio)
    y, count, mean, var, cummean, cumrstd, step_count = tsdn.timestep_decay_norm_fwd(
        x, bos, pc, pm, pv, w, b, pad, G, BETA1, BETA2, EPS, config)
    ry, rcount, rmean, rvar, rcm, rcr, rsc = reference_fwd(x, bos, pc, pm, pv, w, b, G, pad)
    # fp32 matches the reference to fp32 accumulation error; 16-bit outputs add their own rounding
    tol = 3e-4 if dtype == torch.float32 else 1.5e-2
    assert rel_err(y, ry) < tol
    assert rel_err(mean, rmean) < tol
    assert rel_err(var, rvar) < tol
    assert rel_err(cummean, rcm) < 3e-4
    assert rel_err(cumrstd, rcr) < 3e-4
    assert torch.equal(count, rcount)
    assert torch.equal(step_count.long(), rsc)


def test_repeated_interleaved_calls():
    """Many back-to-back calls of different shapes: catches races in the pipeline and the look-back."""
    cases = []
    for B, L, H, G, pad_ratio in [(4, 4100, 2048, 32, 0.02), (2, 9000, 1024, 16, 0.0), (1, 300, 512, 8, 0.05)]:
        args = make_inputs(B, L, H, G, torch.bfloat16, 0.002, pad_ratio, seed=B * L)
        x, bos, pc, pm, pv, w, b, pad = args
        call = (x, bos, pc, pm, pv, w, b, pad, G, BETA1, BETA2, EPS)
        cases.append((call, [t.clone() for t in tsdn.timestep_decay_norm_fwd(*call)]))
    # The look-back composes whichever predecessor states are published first, so results are not
    # bitwise reproducible (fp32 rounding can flip a 16-bit output by one ulp); races would be far larger.
    for _ in range(20):
        for call, first in cases:
            for out, ref in zip(tsdn.timestep_decay_norm_fwd(*call), first):
                scale = max(1.0, ref.double().abs().max().item())
                assert (out.double() - ref.double()).abs().max().item() <= 1e-2 * scale


BWD_CASES = [  # B, L, H, G, dtype, bos_ratio, pad_ratio
    (2, 128, 256, 4, torch.float32, 0.0, 0.0),
    (3, 300, 256, 4, torch.float32, 0.02, 0.0),
    (2, 517, 512, 8, torch.float32, 0.02, 0.1),
    (2, 1000, 256, 32, torch.bfloat16, 0.01, 0.05),
    (1, 9000, 1024, 16, torch.bfloat16, 0.0005, 0.0),
    (2, 4100, 2048, 32, torch.bfloat16, 0.001, 0.01),
]


@pytest.mark.parametrize("recompute_counts", [False, True])
@pytest.mark.parametrize("B,L,H,G,dtype,bos_ratio,pad_ratio", BWD_CASES)
def test_backward_matches_autograd(B, L, H, G, dtype, bos_ratio, pad_ratio, recompute_counts):
    x, bos, pc, pm, pv, w, b, pad = make_inputs(B, L, H, G, dtype, bos_ratio, pad_ratio)
    _, _, _, _, cummean, cumrstd, step_count = tsdn.timestep_decay_norm_fwd(
        x, bos, pc, pm, pv, w, b, pad, G, BETA1, BETA2, EPS)
    g = torch.Generator(device="cuda").manual_seed(1)
    dy = torch.randn(B, L, H, device="cuda", generator=g).to(dtype)
    mean_grad = torch.randn(B, G, device="cuda", generator=g).to(dtype)
    var_grad = torch.randn(B, G, device="cuda", generator=g).to(dtype)
    grads = tsdn.timestep_decay_norm_bwd(
        dy, mean_grad, var_grad, x, bos, cummean, cumrstd, None if recompute_counts else step_count, w, pad,
        G, BETA1, BETA2, prev_count=pc)
    leaves = [t.double().requires_grad_() for t in (x, pm, pv, w, b)]
    ry, _, rmean, rvar, _, _, _ = reference_fwd(leaves[0], bos, pc, leaves[1], leaves[2], leaves[3], leaves[4], G, pad)
    ((ry * dy.double()).sum() + (rmean * mean_grad.double()).sum() + (rvar * var_grad.double()).sum()).backward()
    tol = 3e-4 if dtype == torch.float32 else 1e-2
    for out, leaf in zip(grads, leaves):
        assert rel_err(out, leaf.grad) < tol


@pytest.mark.parametrize("B,L,H,G,dtype,bos_ratio,pad_ratio", [BWD_CASES[0], BWD_CASES[2], BWD_CASES[4]])
def test_backward_residual_accumulate(B, L, H, G, dtype, bos_ratio, pad_ratio):
    """residual=: dx reduce-added into the residual gradient by the TMA stores (x_grad is the residual tensor)."""
    x, bos, pc, pm, pv, w, b, pad = make_inputs(B, L, H, G, dtype, bos_ratio, pad_ratio)
    _, _, _, _, cummean, cumrstd, step_count = tsdn.timestep_decay_norm_fwd(
        x, bos, pc, pm, pv, w, b, pad, G, BETA1, BETA2, EPS)
    g = torch.Generator(device="cuda").manual_seed(1)
    dy = torch.randn(B, L, H, device="cuda", generator=g).to(dtype)
    mean_grad = torch.randn(B, G, device="cuda", generator=g).to(dtype)
    var_grad = torch.randn(B, G, device="cuda", generator=g).to(dtype)
    res = torch.randn(B, L, H, device="cuda", generator=g).to(dtype)
    args = (dy, mean_grad, var_grad, x, bos, cummean, cumrstd, step_count, w, pad, G, BETA1, BETA2)
    ref = tsdn.timestep_decay_norm_bwd(*args, prev_count=pc)
    acc = res.clone()
    got = tsdn.timestep_decay_norm_bwd(*args, prev_count=pc, residual=acc)
    assert got[0] is acc
    torch.testing.assert_close(got[0], (ref[0] + res).to(dtype), rtol=0, atol=0)  # same bf16 / fp32 add as torch
    for a, r in zip(got[1:], ref[1:]):
        # gamma / beta gradients come from fp32 atomics (not bitwise reproducible) rounded to dtype: two summation
        # orders can land on neighbouring bf16 values, so allow one ulp of the output dtype
        torch.testing.assert_close(a, r, rtol=1e-3 if a.dtype == torch.float32 else 2 ** -7, atol=1e-3)


@pytest.mark.parametrize("dtype,H,G", [(torch.bfloat16, 512, 4), (torch.float32, 512, 8)])
def test_backward_addend(dtype, H, G):
    """timestep_decay_norm_bwd(addend=r) == timestep_decay_norm_bwd() + r (fused into the dx epilogue)."""
    B, L = 2, 3000
    x, bos, pc, pm, pv, w, b, pad = make_inputs(B, L, H, G, dtype, 0.01, 0.05)
    _, _, _, _, cummean, cumrstd, step_count = tsdn.timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, pad, G, BETA1, BETA2, EPS)
    g = torch.Generator(device="cuda").manual_seed(2)
    dy = torch.randn(B, L, H, device="cuda", generator=g).to(dtype)
    r = torch.randn(B, L, H, device="cuda", generator=g).to(dtype)
    mg = torch.randn(B, G, device="cuda", generator=g).to(dtype)
    vg = torch.randn(B, G, device="cuda", generator=g).to(dtype)
    g0 = tsdn.timestep_decay_norm_bwd(dy, mg, vg, x, bos, cummean, cumrstd, step_count, w, pad, G, BETA1, BETA2, prev_count=pc)
    g1 = tsdn.timestep_decay_norm_bwd(dy, mg, vg, x, bos, cummean, cumrstd, step_count, w, pad, G, BETA1, BETA2, prev_count=pc, addend=r)
    for a, c in zip(g0[1:], g1[1:]):  # the dgamma / dbeta atomics are not bitwise deterministic
        assert rel_err(c, a.double()) < 1e-4
    ref = (g0[0].float() + r.float()).to(dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert rel_err(g1[0], ref.double()) < tol


def test_step_counts_match_forward():
    x, bos, pc, pm, pv, w, b, pad = make_inputs(3, 2000, 256, 4, torch.float32, 0.01, 0.05)
    step_count = tsdn.timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, pad, 4, BETA1, BETA2, EPS)[6]
    assert torch.equal(step_count, tsdn.step_counts(bos, pad, pc, 2000))
    assert torch.equal(tsdn.timestep_decay_norm_fwd(x, None, pc, pm, pv, w, b, None, 4, BETA1, BETA2, EPS)[6],
                       tsdn.step_counts(None, None, pc, 2000))
