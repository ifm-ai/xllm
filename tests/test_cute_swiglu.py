"""CuTe DSL fused SwiGLU elementwise kernels against the torch ops they replace."""
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "cute_swiglu", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "swiglu.py"))
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs a Blackwell GPU")


@pytest.mark.parametrize("R,E", [(256, 4096), (100, 64), (3, 8), (1024, 9216)])
def test_swiglu(R, E):
    torch.manual_seed(0)
    h1 = (torch.randn(R, E, device="cuda") * 2).to(torch.bfloat16)
    h3 = torch.randn_like(h1)
    g = torch.randn_like(h1)
    hidden = F.silu(h1) * h3
    assert torch.equal(sg.swiglu_fwd(h1, h3), hidden)
    h, d1, d3 = sg.swiglu_bwd(h1, h3, g)
    assert torch.equal(h, hidden)
    assert torch.equal(d3, g * F.silu(h1))
    d1_ref = torch.ops.aten.silu_backward(g * h3, h1)
    assert torch.allclose(d1.float(), d1_ref.float(), atol=1e-2, rtol=1e-2)
    # fp32 autograd
    h1f, h3f = h1.float().requires_grad_(True), h3.float().requires_grad_(True)
    (F.silu(h1f) * h3f * g.float()).sum().backward()
    assert torch.allclose(d1.float(), h1f.grad, atol=5e-2, rtol=2e-2)
    assert torch.allclose(d3.float(), h3f.grad, atol=5e-2, rtol=2e-2)


def test_swiglu_supported():
    h1 = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
    assert sg.swiglu_supported(h1, 0.0)
    assert not sg.swiglu_supported(h1, 0.1)
    assert not sg.swiglu_supported(h1.float(), 0.0)
    assert not sg.swiglu_supported(torch.zeros(3, 3, dtype=torch.bfloat16, device="cuda"), 0.0)
