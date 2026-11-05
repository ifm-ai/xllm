"""Muon's Newton-Schulz (xllm/optim/muon.py) on momenta of any size: the result is finite and within 1% of the polar
factor U V^T of the exact SVD, whatever the norm, and scaling the input changes only rounding. The scale
s = ||(X^T X)^2||_F^(1/4) squares X's entries eight times over and underflowed fp32 for ||X||_F below ~1e-5; s = eps
then left X / s far above 1 and the steps overflowed to NaN.

    python -m pytest -q tests/test_muon_newton_schulz.py
"""
import pytest
import torch

from xllm.optim.muon import newton_schulz

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# tall and wide, a hyper-connection controller's [4 D, 8], odd sizes
SHAPES = [(1024, 8), (8, 1024), (256, 768), (64, 256), (100, 37)]
NORMS = [1e2, 1.0, 1e-3, 1e-5, 2e-6, 1e-7, 1e-12]
EPS = 1e-8


def momentum(shape, norm, device):
    torch.manual_seed(0)
    X = torch.randn(1, *shape, device=device)
    return (X / X.norm() * norm).bfloat16()


def polar(X):
    U, _, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return U @ Vh


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("norm", NORMS)
def test_any_norm(device, shape, norm):
    X = momentum(shape, norm, device)
    out = newton_schulz(X, EPS)
    assert torch.isfinite(out).all(), f"||X|| = {norm:g}: non-finite"
    exact = polar(X)
    err = ((out.double() - exact).norm() / exact.norm()).item()
    assert err < 1e-2, f"||X|| = {norm:g}: {err:.3g} from U V^T"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
def test_scale_free(device, shape):
    X = momentum(shape, 1.0, device)
    ref = newton_schulz(X, EPS).double()
    for scale in (1e-9, 1e-4, 1e3):
        out = newton_schulz((X.float() * scale).bfloat16(), EPS).double()
        err = ((out - ref).norm() / ref.norm()).item()
        assert err < 1e-2, f"x {scale:g}: {err:.3g} from the unscaled result"


@pytest.mark.parametrize("device", DEVICES)
def test_zero(device):
    out = newton_schulz(torch.zeros(2, 64, 32, dtype=torch.bfloat16, device=device), EPS)
    assert (out == 0).all()
