"""Fused linear cross entropy through quack's fused path (Hopper) against an fp32 reference and the previous path:
per-token loss (masked / ignored tokens 0), grad_input and grad_weight of sum(loss)."""
import sys
import types

import pytest
import torch

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
ce = pytest.importorskip("xllm.models.fused_blocks.cross_entropy")
try:
    import quack.linear_cross_entropy  # noqa: F401
except Exception as e:  # not installed, or its custom ops do not load on this torch
    pytest.skip(f"quack.linear_cross_entropy unavailable: {e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                                reason="quack's fused linear cross entropy is sm_90 only")


def _reference(x, w, tgt, keep):
    xf, wf = (t.float().requires_grad_(True) for t in (x, w))
    logits = xf.view(-1, xf.shape[-1]) @ wf.t()
    loss = torch.nn.functional.cross_entropy(logits, tgt.view(-1).clamp(min=0), reduction="none") * keep.view(-1)
    loss.sum().backward()
    return loss.detach().view(tgt.shape), xf.grad, wf.grad


@pytest.mark.parametrize("B,L,D,V", [(2, 2048, 512, 32000), (1, 4096, 768, 128256), (1, 3000, 512, 32000)])
def test_quack_linear_ce(B, L, D, V, monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, L, D, device="cuda", generator=g).bfloat16()
    w = (torch.randn(V, D, device="cuda", generator=g) * 0.05).bfloat16()
    tgt = torch.randint(0, V, (B, L), device="cuda", generator=g)
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.1
    tgt[torch.rand(B, L, device="cuda", generator=g) < 0.05] = -100
    keep = (mask & (tgt != -100)).float()
    ref = _reference(x, w, tgt, keep)
    out = {}
    for quack in ("1", "0"):
        monkeypatch.setenv("XLLM_QUACK_CE", quack)
        out[quack] = ce._fused_linear_cross_entropy(x, w, tgt, mask, None, -100, 0.0, True, True, False)
    for (name, r), q, o in zip((("loss", ref[0]), ("dx", ref[1]), ("dw", ref[2])), out["1"], out["0"]):
        rq = ((q.float() - r).norm() / r.norm()).item()
        ro = ((o.float() - r).norm() / r.norm()).item()
        assert rq < 6e-3, (name, rq, ro)  # both paths at bf16 level (quack: dw in fp32, dx from the recomputed softmax)
    assert (out["1"][0][keep == 0] == 0).all()
    # gradient only (the backward's recompute path)
    monkeypatch.setenv("XLLM_QUACK_CE", "1")
    _, dx2, dw2 = ce._fused_linear_cross_entropy(x, w, tgt, mask, None, -100, 0.0, False, True, False)
    assert torch.equal(dx2, out["1"][1]) and torch.equal(dw2, out["1"][2])
