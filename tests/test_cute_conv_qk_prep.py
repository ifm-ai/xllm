"""Fused causal conv + q / k preparation (cute/conv_qk_prep.py) against the unfused Triton conv + cute qk_prep path:
outputs, input gradient, conv weight gradient (with and without the tap softmax) and q norm weight gradient."""
import sys
import types

import pytest
import torch

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
utils = pytest.importorskip("xllm.models.fused_blocks.utils")
qk = pytest.importorskip("xllm.modules.fused_ops.cute.qk_prep")
cq = pytest.importorskip("xllm.modules.fused_ops.cute.conv_qk_prep")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs Hopper or newer")


def _freqs(L, R):
    t = torch.arange(L, device="cuda", dtype=torch.float32)[:, None] * (500000 ** (-torch.arange(0, R, 2, device="cuda") / R))[None]
    return torch.polar(torch.ones_like(t), t)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("B,L,G,S,R,affine,extra,p_bos,normalize", [
    (1, 256, 2, 256, 128, True, False, 0.0, True),
    (2, 1000, 4, 256, 128, False, True, 0.02, True),
    (1, 515, 8, 128, 128, True, False, 0.05, False),
    (2, 4096, 8, 256, 128, True, False, 1 / 300, True),
    (1, 777, 4, 256, 0, False, True, 0.01, True),
    (1, 2048, 32, 128, 128, True, False, 0.01, True),
    (1, 64, 2, 64, 32, True, True, 0.2, True),
    (2, 300, 4, 256, 128, False, True, None, True),
])
def test_conv_qk_prep(B, L, G, S, R, affine, extra, p_bos, normalize):
    g = torch.Generator(device="cuda").manual_seed(0)
    D = G * S
    fc = _freqs(L, R) if R else None
    bos = (torch.rand(B, L, device="cuda", generator=g) < p_bos) if p_bos is not None else None
    x = torch.randn(B, L, D, device="cuda", generator=g).bfloat16()
    cw = torch.randn(D, 4, device="cuda", generator=g) * (1.0 if normalize else 0.5)
    nw = torch.randn(D, device="cuda", generator=g) * 0.1 if affine else None
    xc, _, _ = utils.causal_conv1d_fwd(x, cw, None, None, bos, False, None, normalize, "triton", False)
    y0, _, p0, _ = qk.qk_prep_fwd(xc, G, nw, fc, R, 1e-6)
    c1, y1, p1 = cq.conv_qk_prep_fwd(x, cw, normalize, bos, G, nw, fc, R, 1e-6, want_c=True)
    dy = torch.randn(B, L, G, S, device="cuda", generator=g).bfloat16()
    dp = (torch.randn(B, L, G, S, device="cuda", generator=g) * 0.1).bfloat16()
    ex = torch.randn(B, L, G, S, device="cuda", generator=g).bfloat16() if extra else None
    dxc, dn0 = qk.qk_prep_bwd(dy, dp, ex, xc, G, nw, fc, R, 1e-6)
    dx0, _, dcw0, _ = utils.causal_conv1d_bwd(dxc, None, x, cw, None, None, bos, None, normalize, "triton", False)
    dx1, dcw1, dn1 = cq.conv_qk_prep_bwd(dy, dp, ex, x, cw, normalize, bos, G, nw, fc, R, 1e-6)
    # c may differ from the Triton conv by an fp32 summation-order ulp before the bf16 rounding
    assert _rel(c1, xc) < 1e-4 and _rel(y1, y0) < 1e-4 and _rel(p1, p0) < 1e-4
    assert _rel(dx1, dx0) < 1e-3 and _rel(dcw1, dcw0) < 1e-3
    if affine:
        assert _rel(dn1, dn0) < 1e-3


def test_conv_qk_prep_bwd_rows_past_end_stale_smem():
    """The look-ahead rows past the sequence end take zero gradients; their rope-table ring slot must be zeroed too,
    or stale shared memory (NaN left by an earlier kernel) times the zero dy poisons dx rows L - 3 .. L - 1."""
    B, L, G, S, R = 2, 2048, 8, 256, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    D = G * S
    fc = _freqs(L, R)
    bos = torch.rand(B, L, device="cuda", generator=g) < 0.01
    x = torch.randn(B, L, D, device="cuda", generator=g).bfloat16()
    cw = torch.randn(D, 4, device="cuda", generator=g)
    nw = torch.randn(D, device="cuda", generator=g) * 0.1
    dy = torch.randn(B, L, G, S, device="cuda", generator=g).bfloat16()
    dp = (torch.randn(B, L, G, S, device="cuda", generator=g) * 0.1).bfloat16()
    xc, _, _ = utils.causal_conv1d_fwd(x, cw, None, None, bos, False, None, True, "triton", False)
    dxc, _ = qk.qk_prep_bwd(dy, dp, None, xc, G, nw, fc, R, 1e-6)
    dx0, _, _, _ = utils.causal_conv1d_bwd(dxc, None, x, cw, None, None, bos, None, True, "triton", False)
    nan = torch.full((4096, 4096), float("nan"), device="cuda", dtype=torch.bfloat16)
    nan @ nan  # cuBLAS stages NaN tiles through shared memory on every SM
    dx1, _, _ = cq.conv_qk_prep_bwd(dy, dp, None, x, cw, True, bos, G, nw, fc, R, 1e-6)
    assert torch.isfinite(dx1).all()
    assert _rel(dx1[:, -3:], dx0[:, -3:]) < 1e-2
