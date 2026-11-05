"""CuTe DSL causal depthwise conv1d (forward and backward, with the tap softmax fused) against the repo's reference."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "cute_causal_conv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "causal_conv.py"))
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)
_rspec = importlib.util.spec_from_file_location(
    "causal_conv1d_reference", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "conv", "triton",
                                            "causal_conv1d_reference.py"))
_ref = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(_ref)
causal_conv1d_reference_fwd, causal_conv1d_reference_bwd = _ref.causal_conv1d_reference_fwd, _ref.causal_conv1d_reference_bwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                                reason="needs a Blackwell GPU")


@pytest.mark.parametrize("B,L,D,act,bosp", [(2, 512, 2048, None, 0.01), (2, 300, 1024, "silu", 0.02), (1, 128, 8192, "silu", 0.0),
                                            (3, 100, 64, "silu", 0.1), (1, 35, 8, None, 0.0), (2, 70, 16, None, 0.3)])
def test_causal_conv(B, L, D, act, bosp):
    torch.manual_seed(0)
    x = torch.randn(B, L, D, device="cuda").to(torch.bfloat16)
    w = torch.softmax(torch.randn(D, 4, device="cuda"), -1)
    bos = None
    if bosp > 0:
        bos = torch.rand(B, L, device="cuda") < bosp
        bos[:, 0] = True
    g = torch.randn_like(x)
    out_ref, _ = causal_conv1d_reference_fwd(x, w, None, None, bos, act)
    out = cv.conv_fwd(x, w, bos, act)
    assert torch.allclose(out.float(), out_ref.float(), atol=2e-2, rtol=1e-2)
    dx_ref, _, dw_ref, _ = causal_conv1d_reference_bwd(g, None, x, w, None, None, bos, act)
    dx, dw = cv.conv_bwd(g, x, w, bos, act)
    assert torch.allclose(dx.float(), dx_ref.float(), atol=2e-2, rtol=1e-2)
    assert torch.allclose(dw, dw_ref, atol=1e-3 * dw_ref.abs().max().item(), rtol=1e-3)


@pytest.mark.parametrize("B,L,D,act,bosp,wdt", [(2, 512, 2048, None, 0.01, torch.bfloat16), (1, 300, 1024, "silu", 0.02, torch.float32),
                                                (1, 128, 8192, "silu", 0.0, torch.bfloat16)])
def test_causal_conv_fused_softmax(B, L, D, act, bosp, wdt):
    """conv_fwd(softmax=True) normalizes the raw taps inside the kernel == conv of the softmaxed weight."""
    torch.manual_seed(1)
    x = torch.randn(B, L, D, device="cuda").to(torch.bfloat16)
    raw = (torch.randn(D, 4, device="cuda") * 2).to(wdt)
    w = torch.softmax(raw, -1, dtype=torch.float32)
    bos = None
    if bosp > 0:
        bos = torch.rand(B, L, device="cuda") < bosp
        bos[:, 0] = True
    out_ref, _ = causal_conv1d_reference_fwd(x, w, None, None, bos, act)
    out = cv.conv_fwd(x, raw, bos, act, softmax=True)
    assert torch.allclose(out.float(), out_ref.float(), atol=2e-2, rtol=1e-2)
    out2 = cv.conv_fwd(x, w, bos, act)
    assert torch.allclose(out.float(), out2.float(), atol=1e-2, rtol=1e-2)
    # backward w.r.t. the raw weight: softmax backward of the reference tap gradient
    g = torch.randn_like(x)
    dx_ref, _, dw_ref, _ = causal_conv1d_reference_bwd(g, None, x, w, None, None, bos, act)
    dw_raw_ref = w * (dw_ref.float() - (w * dw_ref.float()).sum(-1, keepdim=True))
    dx, dw_raw = cv.conv_bwd(g, x, raw, bos, act, softmax=True)
    assert dw_raw.dtype == wdt and dw_raw.shape == raw.shape
    assert torch.allclose(dx.float(), dx_ref.float(), atol=2e-2, rtol=1e-2)
    tol = (1e-2 if wdt == torch.bfloat16 else 1e-3) * dw_raw_ref.abs().max().item()
    assert torch.allclose(dw_raw.float(), dw_raw_ref, atol=tol, rtol=1e-2)


def test_causal_conv_supported():
    x = torch.zeros(1, 8, 64, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(64, 4, device="cuda")
    assert cv.conv_supported(x, w, None, None, False)
    assert not cv.conv_supported(x, w, torch.zeros(64, device="cuda"), None, False)
    assert not cv.conv_supported(x, w, None, None, True)
    assert not cv.conv_supported(x, torch.zeros(64, 3, device="cuda"), None, None, False)
    assert cv.conv_bwd_preferred(x, w, None, None)
    assert not cv.conv_bwd_preferred(torch.zeros(2, 8192, 2048, dtype=torch.bfloat16, device="cuda"), torch.zeros(2048, 4, device="cuda"), None, None)
