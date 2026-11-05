"""CuTe DSL fused attention output gate (rmsnorm(sca + awk) * gate(rmx) and its backward) against fp32 autograd."""
import importlib.util
import math
import os

import pytest
import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "cute_attn_gate", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "attn_gate.py"))
ag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ag)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs a Blackwell GPU")
EPS = 1e-6


def _gate(t, gate):
    return F.silu(t) if gate == "silu" else F.softplus(t, beta=math.log(2))


@pytest.mark.parametrize("B,L,H,V,gate", [(2, 256, 8, 256, "softplus"), (1, 512, 32, 128, "softplus"), (1, 128, 4, 2048, "softplus"),
                                          (1, 64, 4, 512, "silu"), (1, 100, 3, 256, "softplus"), (1, 50, 2, 1024, "silu")])
def test_attn_gate(B, L, H, V, gate):
    torch.manual_seed(0)
    sca = torch.randn(B, L, H * V, device="cuda").to(torch.bfloat16)
    awk = torch.randn_like(sca)
    rmx = (torch.randn_like(sca) * 8)  # exercises the softplus threshold
    g = torch.randn_like(sca)
    out = ag.attn_gate_fwd(sca, awk, rmx, H, gate, EPS)
    ds, dr, attn, delta = ag.attn_gate_bwd(g, sca, awk, rmx, H, gate, EPS, want_attn=True, want_delta=True)
    assert torch.equal(attn, out)
    delta_ref = (ds.float().view(B, L, H, V) * sca.float().view(B, L, H, V)).sum(-1).permute(0, 2, 1)
    assert torch.allclose(delta, delta_ref, atol=1e-2 * delta_ref.abs().max().item(), rtol=1e-3)
    sf, af, rf = [t.float().requires_grad_(True) for t in (sca, awk, rmx)]
    s = (sf + af).view(B, L, H, V)
    n = s * torch.rsqrt((s * s).mean(-1, keepdim=True) + EPS)
    o = n.view(B, L, H * V) * _gate(rf, gate)
    (o * g.float()).sum().backward()
    scale = o.abs().max().item()
    assert torch.allclose(out.float(), o, atol=1.5e-2 * scale, rtol=2e-2)
    assert torch.allclose(ds.float(), sf.grad, atol=1.5e-2 * sf.grad.abs().max().item(), rtol=2e-2)
    assert torch.allclose(ds.float(), af.grad, atol=1.5e-2 * sf.grad.abs().max().item(), rtol=2e-2)
    assert torch.allclose(dr.float(), rf.grad, atol=1.5e-2 * rf.grad.abs().max().item(), rtol=2e-2)


def test_attn_gate_supported():
    x = torch.zeros(1, 8, 1024, dtype=torch.bfloat16, device="cuda")
    assert ag.attn_gate_supported(x, 4, "softplus", 0.0)
    assert not ag.attn_gate_supported(x, 4, "softplus", 0.1)
    assert not ag.attn_gate_supported(x, 4, "gelu", 0.0)
    assert not ag.attn_gate_supported(x.float(), 4, "softplus", 0.0)
    assert not ag.attn_gate_supported(torch.zeros(1, 8, 96, dtype=torch.bfloat16, device="cuda"), 1, "softplus", 0.0)
