"""CuTe DSL kernels of the H100 port (any arch) against their torch / xllm_extension references: softmax backward with
accumulation, group RMS / layer norm forward and backward (incl. fused rotary embedding). The rope / SwiGLU / gate /
cross-entropy kernels have their own tests (tests/test_cute_{rope,swiglu,attn_gate,cross_entropy}.py)."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _ulp_close(a, b, ulps=1):
    """bf16 results within `ulps` units in the last place of the larger magnitude."""
    tol = ulps * torch.finfo(torch.bfloat16).eps * torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-3)
    assert ((a.float() - b.float()).abs() <= tol + 1e-6).all(), (a.float() - b.float()).abs().max()


@pytest.mark.parametrize("shape", [(2, 512, 8, 256), (1, 256, 32, 128), (3, 128, 2, 64)])
@pytest.mark.parametrize("with_add", [False, True])
def test_softmax_bwd_acc(shape, with_add):
    from xllm.modules.fused_ops.cute.softmax_bwd import softmax_bwd_acc_
    torch.manual_seed(0)
    x = (torch.randn(*shape, device="cuda") * 3).bfloat16()
    g, acc = torch.randn_like(x), torch.randn_like(x)
    add = torch.randn_like(x) if with_add else None
    y = F.softmax(x.double(), -1)
    dx = y * (g.double() - (g.double() * y).sum(-1, keepdim=True))
    ref = acc.double() + dx + (add.double() if with_add else 0)
    out = softmax_bwd_acc_(acc.clone(), x, g, add)
    torch.testing.assert_close(out.double(), ref, atol=3e-2, rtol=1e-2)


@pytest.mark.parametrize("D,G,affine", [(1024, 4, False), (2048, 8, True), (4096, 32, True), (1024, 8, False)])
def test_group_rms_norm_fwd(D, G, affine):
    import xllm_extension.ops as ops
    from xllm.modules.fused_ops.cute.group_norm import group_norm_fwd
    torch.manual_seed(0)
    x = torch.randn(2, 512, D, device="cuda").bfloat16()
    w = torch.randn(D, device="cuda").bfloat16()
    ry, rr = ops.group_rms_norm_fwd_affine(x, D, G, w, 1e-6) if affine else ops.group_rms_norm_fwd(x, D, G, 1e-6)
    y, r = group_norm_fwd(x, G, 1e-6, w if affine else None)
    assert r.shape == rr.shape
    torch.testing.assert_close(r, rr, atol=0, rtol=1e-6)
    _ulp_close(y, ry)


def test_group_layer_norm_fwd():
    import xllm_extension.ops as ops
    from xllm.modules.fused_ops.cute.group_norm import group_norm_fwd
    torch.manual_seed(0)
    D, G = 1536, 2
    x = torch.randn(2, 512, D, device="cuda").bfloat16()
    w, b = torch.randn(D, device="cuda").bfloat16(), torch.randn(D, device="cuda").bfloat16()
    ry, rm, rr = ops.group_layer_norm_fwd_affine(x, D, G, w, b, 1e-6)
    y, m, r = group_norm_fwd(x, G, 1e-6, w, b, layer=True)
    assert m.shape == rm.shape and r.shape == rr.shape
    torch.testing.assert_close(m, rm, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(r, rr, atol=0, rtol=1e-6)
    _ulp_close(y, ry)


@pytest.mark.parametrize("kind,D,G", [("rms", 1024, 4), ("rms_aff", 2048, 8), ("rms_mem", 2048, 8), ("rms_aff", 4096, 32),
                                      ("layer", 1536, 2), ("layer", 2560, 2)])
def test_group_norm_bwd(kind, D, G):
    import xllm_extension.ops as ops
    from xllm.modules.fused_ops.cute.group_norm import group_norm_bwd
    torch.manual_seed(0)
    x = torch.randn(2, 1024, D, device="cuda").bfloat16()
    dy = torch.randn_like(x)
    w = (torch.randn(D, device="cuda") * 0.5 + 1).bfloat16()
    b = torch.randn(D, device="cuda").bfloat16()
    if kind == "layer":
        _, m, r = ops.group_layer_norm_fwd_affine(x, D, G, w, b, 1e-5)
        ref = ops.group_layer_norm_bwd_affine(dy, x, D, G, m, r, w, b, False)
        out = group_norm_bwd(dy, x, G, r, m, w, layer=True)
    elif kind == "rms_aff":
        _, r = ops.group_rms_norm_fwd_affine(x, D, G, w, 1e-6)
        ref = ops.group_rms_norm_bwd_affine(dy, x, D, G, r, w, False)
        out = group_norm_bwd(dy, x, G, r, None, w)
    elif kind == "rms_mem":
        y, r = ops.group_rms_norm_fwd(x, D, G, 1e-6)
        ref = (ops.group_rms_norm_bwd(dy, y, D, G, r, True),)
        out = group_norm_bwd(dy, y, G, r, mem_eff=True)
    else:
        _, r = ops.group_rms_norm_fwd(x, D, G, 1e-6)
        ref = (ops.group_rms_norm_bwd(dy, x, D, G, r, False),)
        out = group_norm_bwd(dy, x, G, r)
    assert len(out) == len(ref)
    for a, c in zip(out, ref):
        torch.testing.assert_close(a.float(), c.float(), atol=4e-2, rtol=1e-2)


@pytest.mark.parametrize("H,S,R,affine", [(8, 256, 128, True), (4, 256, 128, False), (32, 128, 128, True), (4, 256, 64, False)])
def test_group_norm_rope_fused(H, S, R, affine):
    """rms norm + rotary embedding in one kernel (forward and backward) vs norm then apply_rope."""
    from xllm.models.fused_blocks import utils
    from xllm.modules.fused_ops.cute.group_norm import group_norm_bwd, group_norm_fwd
    torch.manual_seed(0)
    B, L, D = 2, 256, H * S
    x = torch.randn(B, L, D, device="cuda").bfloat16()
    w = (torch.randn(D, device="cuda") * 0.5).bfloat16() if affine else None
    phase = torch.randn(L, R // 2, device="cuda")
    f = torch.polar(torch.ones_like(phase), phase)
    y_ref, r_ref = utils.rmsnorm_fwd(x, w, H, 1e-6)
    y_ref = utils.apply_rope(y_ref.view(B, L, H, S), f, S, R, False).reshape(B, L, D)
    y, r = group_norm_fwd(x, H, 1e-6, w + 1.0 if affine else None, freqs_cis=f, rope_head_dim=R)
    # absolute tolerance: the unfused reference rounds the normalized values to bf16 before the rotation mixes them
    torch.testing.assert_close(y.float(), y_ref.float(), atol=4e-2, rtol=1e-2)
    torch.testing.assert_close(r, r_ref, atol=0, rtol=1e-6)
    dy = torch.randn_like(x)
    dyr = utils.apply_rope(dy.view(B, L, H, S), f, S, R, True).reshape(B, L, D)
    ref = utils.rmsnorm_bwd(dyr, x, r_ref, w, H)
    out = group_norm_bwd(dy, x, H, r, None, w + 1.0 if affine else None, freqs_cis=f, rope_head_dim=R)
    torch.testing.assert_close(out[0].float(), ref[0].float(), atol=7e-2, rtol=2e-2)
    if affine:
        torch.testing.assert_close(out[1].float(), ref[1].float(), atol=0.5, rtol=1e-2)
