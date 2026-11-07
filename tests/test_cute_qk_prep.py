"""CuTe DSL fused q / k preparation (rmsnorm + rope + softmax, and its backward) against fp32 autograd."""
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "cute_qk_prep", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "qk_prep.py"))
qk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qk)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs a Blackwell GPU")
EPS = 1e-6


def _freqs(L, R, base=500000.0):
    f = 1.0 / (base ** (torch.arange(0, R, 2, device="cuda").float() / R))
    return torch.polar(torch.ones(L, R // 2, device="cuda"), torch.outer(torch.arange(L, device="cuda").float(), f))


def _reference(x, w, fc, G, S, R):
    """fp32: y = rope(rmsnorm(x) (w + 1)), p = softmax(x)."""
    B, L, _ = x.shape
    xv = x.view(B, L, G, S)
    y = xv * torch.rsqrt((xv * xv).mean(-1, keepdim=True) + EPS)
    if w is not None:
        y = y * (w.view(G, S) + 1.0)
    if R > 0:
        yr = torch.view_as_real(torch.view_as_complex(y[..., :R].reshape(B, L, G, R // 2, 2)) * fc.unsqueeze(1)).flatten(3)
        y = torch.cat((yr, y[..., R:]), -1)
    return y, F.softmax(xv, -1)


@pytest.mark.parametrize("B,L,G,S,R,affine,extra", [
    (2, 512, 8, 256, 128, True, False), (2, 256, 4, 256, 128, False, True), (1, 512, 32, 128, 128, True, False),
    (1, 256, 4, 256, 256, True, True), (1, 128, 2, 64, 32, False, False), (1, 64, 3, 64, 0, True, True)])
def test_qk_prep(B, L, G, S, R, affine, extra):
    torch.manual_seed(0)
    x = torch.randn(B, L, G * S, device="cuda").to(torch.bfloat16)
    w = (torch.randn(G * S, device="cuda") * 0.1).to(torch.bfloat16) if affine else None
    fc = _freqs(L, R) if R > 0 else None
    y, pre, p, rstd = qk.qk_prep_fwd(x, G, w, fc, R, EPS, want_pre=True, want_sm=True)
    xf = x.float().requires_grad_(True)
    wf = w.float().requires_grad_(True) if affine else None
    yr, pr = _reference(xf, wf, fc, G, S, R)
    assert torch.allclose(y.float(), yr, atol=3e-2, rtol=2e-2)
    assert torch.allclose(p.float(), pr, atol=2e-3, rtol=2e-2)
    pre_ref, _ = _reference(xf, wf, None, G, S, 0)
    assert torch.allclose(pre.float(), pre_ref, atol=3e-2, rtol=2e-2)
    assert torch.allclose(rstd.view(-1), torch.rsqrt((xf.view(-1, S) ** 2).mean(-1) + EPS).detach(), rtol=1e-5)
    dy = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16)
    dp = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16)
    ex = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16) if extra else None
    (yr * dy.float()).sum().backward()
    (pr * dp.float()).sum().backward()
    dx_ref = xf.grad.view(B, L, G * S) + (ex.float().view(B, L, G * S) if extra else 0)
    dx, dw = qk.qk_prep_bwd(dy, dp, ex, x, G, w, fc, R, EPS)
    assert dx.shape == (B, L, G * S) and dx.dtype == torch.bfloat16
    assert torch.allclose(dx.float(), dx_ref, atol=6e-2, rtol=2e-2)
    if affine:
        assert torch.allclose(dw.float(), wf.grad, atol=0.02 * wf.grad.abs().max(), rtol=1e-2)
    else:
        assert dw is None


def test_qk_prep_no_softmax():
    torch.manual_seed(0)
    B, L, G, S, R = 1, 128, 2, 128, 64
    x = torch.randn(B, L, G * S, device="cuda").to(torch.bfloat16)
    fc = _freqs(L, R)
    y, pre, p, _ = qk.qk_prep_fwd(x, G, None, fc, R, EPS, want_pre=False, want_sm=False)
    assert pre is None and p is None
    yr, _ = _reference(x.float(), None, fc, G, S, R)
    assert torch.allclose(y.float(), yr, atol=3e-2, rtol=2e-2)
    dy = torch.randn(B, L, G, S, device="cuda").to(torch.bfloat16)
    xf = x.float().requires_grad_(True)
    yr, _ = _reference(xf, None, fc, G, S, R)
    (yr * dy.float()).sum().backward()
    dx, _ = qk.qk_prep_bwd(dy, None, None, x, G, None, fc, R, EPS)
    assert torch.allclose(dx.float(), xf.grad.view(B, L, G * S), atol=6e-2, rtol=2e-2)
