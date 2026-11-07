"""CuTe DSL group layer norm (forward / backward) against fp32 autograd."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "cute_ln", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "layer_norm.py"))
ln = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ln)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs a Blackwell GPU")
EPS = 1e-5


@pytest.mark.parametrize("B,L,D,G", [(2, 512, 1536, 2), (1, 256, 2560, 2), (1, 128, 4096, 4), (1, 100, 512, 2), (1, 64, 4096, 1), (3, 33, 256, 1)])
def test_layer_norm(B, L, D, G):
    torch.manual_seed(0)
    x = (torch.randn(B, L, D, device="cuda") * 2 + 1).to(torch.bfloat16)
    w = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    b = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    dy = torch.randn_like(x)
    y, mean, rstd = ln.ln_fwd(x, w, b, G, EPS)
    dx, dw, db = ln.ln_bwd(dy, x, mean, rstd, w, G)
    xf, wf, bf = [t.float().requires_grad_(True) for t in (x, w, b)]
    xv = xf.view(B, L, G, D // G)
    mu = xv.mean(-1, keepdim=True)
    var = ((xv - mu) ** 2).mean(-1, keepdim=True)
    yr = ((xv - mu) * torch.rsqrt(var + EPS)).view(B, L, D) * (wf + 1.0) + bf
    (yr * dy.float()).sum().backward()
    assert y.shape == x.shape and mean.shape == (B * L, G) and rstd.shape == (B * L, G)
    assert torch.allclose(y.float(), yr, atol=4e-2, rtol=2e-2)
    assert torch.allclose(mean.view(-1), mu.reshape(-1).detach(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(rstd.view(-1), torch.rsqrt(var + EPS).reshape(-1).detach(), rtol=1e-4)
    assert torch.allclose(dx.float(), xf.grad, atol=3e-2, rtol=2e-2)
    assert torch.allclose(dw.float(), wf.grad, atol=1e-2 * wf.grad.abs().max().item(), rtol=1e-2)
    assert torch.allclose(db.float(), bf.grad, atol=1e-2 * bf.grad.abs().max().item(), rtol=1e-2)
    res = torch.randn_like(x)
    dx2, dw2, db2 = ln.ln_bwd(dy, x, mean, rstd, w, G, addend=res)
    assert torch.allclose(dx2.float(), xf.grad + res.float(), atol=3e-2, rtol=2e-2)
    assert torch.equal(dw2, dw) and torch.equal(db2, db)


@pytest.mark.parametrize("B,L,D,G", [(2, 512, 1536, 2), (1, 100, 512, 2)])
def test_layer_norm_bwd_addend(B, L, D, G):
    """ln_bwd(addend=r) == ln_bwd() + r (the residual-branch gradient fused into the dx store)."""
    torch.manual_seed(1)
    x = (torch.randn(B, L, D, device="cuda") * 2 + 1).to(torch.bfloat16)
    w = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    b = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    dy = torch.randn_like(x)
    r = torch.randn_like(x)
    _, mean, rstd = ln.ln_fwd(x, w, b, G, EPS)
    dx, dw, db = ln.ln_bwd(dy, x, mean, rstd, w, G)
    dxa, dwa, dba = ln.ln_bwd(dy, x, mean, rstd, w, G, addend=r)
    assert torch.equal(dw, dwa) and torch.equal(db, dba)
    assert torch.allclose(dxa.float(), dx.float() + r.float(), atol=2e-2, rtol=1e-2)


def test_layer_norm_supported():
    x = torch.zeros(1, 8, 1536, dtype=torch.bfloat16, device="cuda")
    assert ln.ln_supported(x, 2)
    assert not ln.ln_supported(x, 5)
    assert not ln.ln_supported(x.float(), 2)
    assert not ln.ln_supported(torch.zeros(1, 8, 96, dtype=torch.bfloat16, device="cuda"), 1)
