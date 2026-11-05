"""swift sliding chunk attention call layouts (one call per chunk, query sub-chunks, segment-aware key starts) against
a dense fp32 reference."""
import sys
import types

import pytest
import torch

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
sca = pytest.importorskip("xllm.modules.fused_ops.attention.sca")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(q, k, v, dy, seg, C, scale):
    L = q.shape[1]
    i = torch.arange(L, device=q.device)
    mask = (i[None, :] <= i[:, None]) & (i[None, :] >= ((i[:, None] // C - 1).clamp(min=0) * C))
    mask = mask[None] & (seg[:, :, None] == seg[:, None, :])
    qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
    s = torch.einsum("blhd,bmhd->bhlm", qf, kf) * scale
    p = s.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
    y = torch.einsum("bhlm,bmhd->blhd", p, vf)
    y.backward(dy.float())
    return y.detach(), qf.grad, kf.grad, vf.grad


@pytest.mark.parametrize("rate", [0.0, 1 / 300, 1 / 50])
@pytest.mark.parametrize("splits,seg_skip", [((1,), False), ((2,), False), ((2, 4), True), ((4,), True)])
def test_swift_blocks(rate, splits, seg_skip, monkeypatch):
    B, L, H, S, V, C = 2, 2048, 2, 64, 128, 512
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k = (torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16() for _ in range(2))
    v = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
    dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
    seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < rate).long(), 1)
    monkeypatch.setattr(sca, "_SWIFT_QSPLIT", splits[0])
    monkeypatch.setattr(sca, "_SWIFT_QSPLITS", splits)
    monkeypatch.setattr(sca, "_SWIFT_SEG_SKIP", seg_skip)
    sca._swift_seg_cache.clear()
    scale = S ** -0.5
    y, y_save, w = sca.sliding_chunk_attention_fwd(q, k, v, C, scale, None, None, None, seg, 0.0, 0, "swift", True)
    grads = sca.sliding_chunk_attention_bwd(dy, q, k, v, y_save, w, C, scale, None, None, None, seg, 0, False, "swift")
    ref = _reference(q, k, v, dy, seg, C, scale)
    for name, a, b in zip(("y", "dq", "dk", "dv"), (y,) + tuple(grads[:3]), ref):
        rel = ((a.float() - b).norm() / b.norm()).item()
        assert rel < 1e-2, (name, rel)


@pytest.mark.parametrize("rate", [0.0, 1 / 300])
def test_swift_fused_ds(rate):
    """The backward with delta = rowsum(dy * y) takes dS = P (dP - delta) from one quack GEMM (attention_bwd_ds_out)."""
    if not sca._swift_ds_ops():
        pytest.skip("quack dS GEMM unavailable")
    B, L, H, S, V, C = 1, 2048, 2, 64, 128, 512
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k = (torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16() for _ in range(2))
    v = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
    dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
    seg = torch.cumsum((torch.rand(B, L, device="cuda", generator=g) < rate).long(), 1)
    sca._swift_seg_cache.clear()
    scale = S ** -0.5
    y, y_save, w = sca.sliding_chunk_attention_fwd(q, k, v, C, scale, None, None, None, seg, 0.0, 0, "swift", True)
    delta = (dy.float() * y.float()).sum(-1).transpose(1, 2).contiguous()
    grads = sca.sliding_chunk_attention_bwd(dy, q, k, v, y_save, w, C, scale, None, None, None, seg, 0, False, "swift",
                                            delta=delta)
    ref = _reference(q, k, v, dy, seg, C, scale)
    for name, a, b in zip(("dq", "dk", "dv"), grads[:3], ref[1:]):
        rel = ((a.float() - b).norm() / b.norm()).item()
        assert rel < 1e-2, (name, rel)
