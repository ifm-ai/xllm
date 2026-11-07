import pytest
import torch

from xllm.distributed.fsdp_patches import _MAX_GRADS, fast_reduce_scatter_copy_in

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("n_grads,world", [(5, 8), (_MAX_GRADS, 4), (_MAX_GRADS + 1, 8), (68, 8), (3 * _MAX_GRADS + 7, 2)])
def test_copy_in_matches_chunk_cat(n_grads, world):
    """The Triton copy-in (any number of gradients: several launches of _MAX_GRADS) equals torch._chunk_cat."""
    g = torch.Generator(device="cuda").manual_seed(n_grads)
    shapes = [(1 + (i * 7) % 13, 1 + (i * 5) % 37) if i % 4 else (1 + (i * 3) % 11,) for i in range(n_grads)]
    grads = [torch.randn(s, device="cuda", generator=g).to(torch.bfloat16) for s in shapes]
    cols = 0
    for t in grads:
        rows = t.shape[0]
        cols += -(-rows // world) * (t.numel() // rows)
    ref = torch.empty(world * cols, dtype=torch.float32, device="cuda")
    torch._chunk_cat(grads, 0, world, out=ref.view(world, -1))
    out = torch.full((world * cols,), -7.0, dtype=torch.float32, device="cuda")
    fast_reduce_scatter_copy_in(grads, out, world)
    assert torch.equal(out, ref)


def test_divide_factor_folded_into_copy_in(monkeypatch):
    """XLLM_FSDP_FOLD_DIV: a power-of-2 post-reduction divide factor becomes an exact scale in the copy-in."""
    import torch.distributed as dist
    import xllm.distributed.fsdp_patches as fp
    monkeypatch.setattr(fp, "_orig_divide_factors", lambda *a, **k: (None, 8.0, dist.ReduceOp.SUM, dist.ReduceOp.SUM))
    assert fp.folded_gradient_divide_factors() == (None, None, dist.ReduceOp.SUM, dist.ReduceOp.SUM)
    assert fp._PENDING_SCALE[0] == 0.125
    g = torch.Generator(device="cuda").manual_seed(0)
    grads = [torch.randn(s, device="cuda", generator=g).to(torch.bfloat16) for s in [(16, 24), (9,), (40, 8)]]
    cols = sum(-(-t.shape[0] // 8) * (t.numel() // t.shape[0]) for t in grads)
    ref = torch.empty(8 * cols, dtype=torch.float32, device="cuda")
    torch._chunk_cat(grads, 0, 8, out=ref.view(8, -1))
    out = torch.empty_like(ref)
    fp.fast_reduce_scatter_copy_in(grads, out, 8)
    assert torch.equal(out, ref / 8) and fp._PENDING_SCALE[0] is None
    # not a power of 2 (or a pre-division): left to FSDP
    monkeypatch.setattr(fp, "_orig_divide_factors", lambda *a, **k: (None, 6.0, dist.ReduceOp.SUM, dist.ReduceOp.SUM))
    assert fp.folded_gradient_divide_factors()[1] == 6.0 and fp._PENDING_SCALE[0] is None
