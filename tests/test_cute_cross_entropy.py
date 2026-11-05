"""CuTe DSL fused loss / softmax-scores kernel of the chunked fused-linear cross entropy against torch."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "cute_ce", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute", "cross_entropy.py"))
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs a Blackwell GPU")


def _reference(x, tgt, ign):
    C, V = x.shape
    pl = x.float()
    m = pl.max(-1)[0]
    pl = pl - m[:, None]
    ar = torch.arange(C, device=x.device)
    pred = pl[ar, tgt].clone()
    pred[ign] = 0.0
    e = torch.exp(pl)
    s = e.sum(-1)
    loss = torch.log(s) - pred
    loss[ign] = 0.0
    sc = e / s[:, None]
    sc[ar, tgt] -= (1.0 - ign.float())
    sc.masked_fill_(ign[:, None], 0.0)
    return loss, sc


@pytest.mark.parametrize("C,V", [(256, 32000), (77, 1000), (5, 8), (33, 128256)])
def test_ce_logits(C, V):
    torch.manual_seed(0)
    x = (torch.randn(C, V, device="cuda") * 4).to(torch.bfloat16)
    tgt = torch.randint(0, V, (C,), device="cuda")
    ign = torch.rand(C, device="cuda") < 0.2
    tgt[ign] = 0
    loss, sc = _reference(x, tgt, ign)
    l2, s32, s16 = ce.ce_logits(x, tgt, ign, True, True)
    assert torch.allclose(l2, loss, atol=1e-4, rtol=1e-5)
    assert torch.allclose(s32, sc, atol=1e-5, rtol=1e-4)
    assert torch.allclose(s16.float(), sc.to(torch.bfloat16).float(), atol=1e-3, rtol=1e-2)
    assert (s32[ign] == 0).all() and (l2[ign] == 0).all()
    l3, s, t = ce.ce_logits(x, tgt, ign, True, False)
    assert s is None and t is None and torch.equal(l3, l2)
    l4, s, t = ce.ce_logits(x, tgt, ign, False, True)
    assert l4 is None and torch.equal(s, s32)
    l5, s, t = ce.ce_logits(x, tgt, ign, True, True, want_fp32=False)
    assert s is None and torch.equal(l5, l2) and torch.equal(t, s16)


def test_ce_logits_supported():
    assert ce.ce_logits_supported(torch.zeros(4, 32, dtype=torch.bfloat16, device="cuda"))
    assert not ce.ce_logits_supported(torch.zeros(4, 30, dtype=torch.bfloat16, device="cuda"))
    assert not ce.ce_logits_supported(torch.zeros(4, 32, dtype=torch.float32, device="cuda"))
