"""Fused-block SwiGLU FFN through quack's gated GEMMs (swiglu_forward / swiglu_backward packed path) against the
unfused path, including the recompute forms of h1 / h3."""
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
utils = pytest.importorskip("xllm.models.fused_blocks.utils")
try:
    import quack.gemm_interface  # noqa: F401
except Exception as e:  # not installed, or its custom ops do not load on this torch
    pytest.skip(f"quack.gemm_interface unavailable: {e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs Hopper or newer")


def _run(monkeypatch, quack: bool, recompute: str):
    monkeypatch.setenv("XLLM_QUACK_FFN", "1" if quack else "0")
    B, L, D, E = 2, 512, 512, 1024
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, L, D, device="cuda", generator=g).bfloat16()
    xf = torch.randn(B * L, D, device="cuda", generator=g).bfloat16()
    w1, w3 = ((torch.randn(E, D, device="cuda", generator=g) * 0.05).bfloat16() for _ in range(2))
    w2 = (torch.randn(D, E, device="cuda", generator=g) * 0.05).bfloat16()
    dy = torch.randn(B * L, D, device="cuda", generator=g).bfloat16()
    nw = (torch.randn(D, device="cuda", generator=g) * 0.1).bfloat16()
    nb = (torch.randn(D, device="cuda", generator=g) * 0.1).bfloat16()
    xm = x.float().view(B * L, 1, D)
    mean = xm.mean(-1)
    rstd = torch.rsqrt(xm.var(-1, unbiased=False) + 1e-5)
    h1, h3, out, rng = utils.swiglu_forward(xf, w1, w2, w3, 0.0, True)
    if recompute in ("fc1", "both"):
        h1 = F.linear(xf, w1)
    if recompute in ("fc3", "both"):
        h3 = F.linear(xf, w3)
    if recompute == "helper":  # the gekko backward's recompute of both (packed on the quack path)
        h1, h3 = utils.swiglu_recompute(xf, w1, w3, None, None, 0.0, True)
    grads = utils.swiglu_backward(dy, xf, w1, w2, w3, h1, h3, 0.0, rng, x, mean, rstd, nw, nb, 1, False, False)
    return (out,) + tuple(grads[:4])


@pytest.mark.parametrize("recompute", ["none", "fc1", "fc3", "both", "helper"])
def test_quack_swiglu_ffn(monkeypatch, recompute):
    ref = _run(monkeypatch, False, recompute)
    got = _run(monkeypatch, True, recompute)
    for name, a, b in zip(("out", "x_grad", "w1_grad", "w2_grad", "w3_grad"), got, ref):
        rel = ((a.float() - b.float()).norm() / b.float().norm()).item()
        assert rel < 1e-2, (name, rel)
