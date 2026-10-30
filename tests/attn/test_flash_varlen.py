import pytest
import torch
import torch.nn.functional as F

# Packed segments (multi_segments=True) take the flash-attn 2 varlen path. Its binding gained an argument
# (num_splits) in Dao-AILab/flash-attention#2448, so a call written for one build fails with a TypeError on
# the other. Runs the varlen forward and backward against causal SDPA per segment.

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_flash_varlen_matches_sdpa_per_segment():
    from xllm.modules.fused_ops import flash_attention_bwd, flash_attention_fwd

    torch.manual_seed(0)
    B, L, H, D = 2, 256, 4, 64
    q, k, v = (torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    bounds = [0, 100, 256, 300, 512]  # segment starts over the flattened B * L tokens
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    max_seqlen = max(b - a for a, b in zip(bounds, bounds[1:]))
    scale = D ** -0.5

    y, lse, rng_state = flash_attention_fwd(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, scale, 0.0, True)
    dy = torch.randn_like(y)
    dq, dk, dv = flash_attention_bwd(
        dy, y, q, k, v, lse, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, scale, 0.0, rng_state, True
    )

    fq, fk, fv, fdy = (t.flatten(0, 1) for t in (q, k, v, dy))
    ref = []
    for a, b in zip(bounds, bounds[1:]):
        seg = [t[a:b].transpose(0, 1).unsqueeze(0) for t in (fq, fk, fv)]
        ref.append(F.scaled_dot_product_attention(*seg, is_causal=True, scale=scale)[0].transpose(0, 1))
    ref = torch.cat(ref)
    ref.backward(fdy)

    tol = dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(y.flatten(0, 1), ref, **tol)
    torch.testing.assert_close(dq.flatten(0, 1), q.grad.flatten(0, 1), **tol)
    torch.testing.assert_close(dk.flatten(0, 1), k.grad.flatten(0, 1), **tol)
    torch.testing.assert_close(dv.flatten(0, 1), v.grad.flatten(0, 1), **tol)
