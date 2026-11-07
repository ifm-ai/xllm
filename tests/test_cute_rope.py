"""CuTe DSL rotary embedding (Blackwell) against the torch complex-multiply version."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "cute_rope", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "rope.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs a Blackwell GPU")


def _freqs(L, R, base=500000.0):
    f = 1.0 / (base ** (torch.arange(0, R, 2, device="cuda").float() / R))
    return torch.polar(torch.ones(L, R // 2, device="cuda"), torch.outer(torch.arange(L, device="cuda").float(), f))


def _reference(x, freqs_cis, R, backward):
    if backward:
        freqs_cis = freqs_cis.conj()
    xr = torch.view_as_complex(x[..., :R].float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xr * freqs_cis.unsqueeze(1)).flatten(3).to(x.dtype)
    return torch.cat((out, x[..., R:]), dim=-1)


@pytest.mark.parametrize("B,L,N,D,R", [(2, 1024, 8, 256, 128), (1, 512, 32, 128, 128), (1, 256, 4, 256, 256), (1, 64, 2, 64, 16), (3, 100, 3, 24, 8)])
@pytest.mark.parametrize("backward", [False, True])
def test_rope(B, L, N, D, R, backward):
    torch.manual_seed(0)
    x = torch.randn(B, L, N, D, device="cuda").to(torch.bfloat16)
    fc = _freqs(L, R)
    out = rp.rope(x, fc, R, backward)
    ref = _reference(x, fc, R, backward)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=1e-2)
    assert torch.equal(out[..., R:], x[..., R:])


def test_rope_rejects_bad_table():
    x = torch.randn(1, 8, 2, 16, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError):
        rp.rope(x, _freqs(8, 8), 16)
