import math
import numpy as np
import torch
import torch.nn.functional as F
import fire
from einops import rearrange, repeat

from xllm.modules.fused_ops import (
    xattn_softdelta_attn_fwd,
    xattn_softdelta_attn_bwd
)


def manual_attention(
    query, qkey, kkey, value, g, q_segment_idx, k_segment_idx, scale
):
    bsz, qlen, _, _ = query.shape
    kvlen = kkey.shape[1]

    # B x H x L x D
    xq = query.transpose(1, 2)
    kq = qkey.transpose(1, 2)
    xk = kkey.transpose(1, 2)
    xv = value.transpose(1, 2)
    # B x H x L x D -> B x H x D x L
    xk = xk.transpose(2, 3)

    # B x L1 x L2
    full_inf = torch.full((bsz, qlen, kvlen), float("-inf"), device=xq.device)
    attn_mask1 = torch.triu(full_inf, diagonal=(kvlen - qlen + 1)).type_as(xq)
    attn_mask2 = torch.triu(full_inf, diagonal=(kvlen - qlen)).type_as(xq)
    if q_segment_idx is not None:
        seg_mask = torch.ne(q_segment_idx.unsqueeze(2), k_segment_idx.unsqueeze(1))
        attn_mask1 = attn_mask1.masked_fill(seg_mask, value=float("-inf"))
        attn_mask2 = attn_mask2.masked_fill(seg_mask, value=float("-inf"))
    # B x L1 x 1
    all_inf = torch.all(attn_mask2.isinf(), dim=2, keepdim=True)
    attn_mask2 = attn_mask2.masked_fill(all_inf, value=0.0)
    # B x H x L1 x L2
    scores1 = torch.matmul(xq, xk) * scale
    scores2 = torch.matmul(kq, xk) * scale
    scores1 = scores1 + attn_mask1.unsqueeze(1)
    scores2 = scores2 + attn_mask2.unsqueeze(1)
    scores1 = F.softmax(scores1, dim=-1, dtype=torch.float32).to(xq)
    scores2 = F.softmax(scores2, dim=-1, dtype=torch.float32).to(kq)
    scores2 = scores2 * torch.logical_not(all_inf).to(kq).unsqueeze(1)
    # B x H x L x D -> B x L x H x D
    out1 = rearrange(torch.matmul(scores1, xv), 'b h l d -> b l h d')
    out2 = rearrange(torch.matmul(scores2, xv), 'b h l d -> b l h d')
    out2 = out2 * rearrange(g, 'b l h -> b l h 1')
    output = out1 - out2
    return output


def test(B: int, L1: int, L2: int, H: int, D: int, multi_segment: bool, dtype: str):
    assert L1 <= L2
    if multi_segment:
        assert L1 == L2
    scale = math.sqrt(1.0 / D)
    bos_ratio = 0.01
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

    with torch.no_grad():
        query = torch.randn(B, L1, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        qkey = torch.randn(B, L1, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        kkey = torch.randn(B, L2, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, L2, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        g = torch.randn(B, L1, H, requires_grad=False, dtype=pt_dtype, device="cuda")

        if multi_segment:
            bos_mask = torch.rand(B, L2, requires_grad=False, device='cuda') < bos_ratio
            k_segment_idx = torch.cumsum(bos_mask, dim=-1)
            bos_idx_k = torch.nonzero(F.pad(bos_mask[:, 1:], (1, 0), value=1))
            cu_seqlens_k = bos_idx_k[:, 0] * L2 + bos_idx_k[:, 1]
            cu_seqlens_k = F.pad(cu_seqlens_k, (0, 1), value=B * L2)
            max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
            cu_seqlens_k = cu_seqlens_k.int()
        else:
            bos_mask = None
            k_segment_idx = None
            cu_seqlens_k = None
            max_seqlen_k = None

    q = query.clone().detach().requires_grad_(True)
    kq = qkey.clone().detach().requires_grad_(True)
    xk = kkey.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)
    g = g.clone().detach().requires_grad_(True)

    y_mannual = manual_attention(
        q.float(), kq.float(), xk.float(), v.float(), g.float(), k_segment_idx, k_segment_idx, scale
    )

    y_mannual_flat = y_mannual.flatten()
    num_elem = y_mannual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.float32, device="cuda") / math.sqrt(L1)
    # weight = torch.ones(num_elem, requires_grad=False, dtype=torch.double, device="cuda")
    loss = y_mannual_flat @ weight
    y_mannual.retain_grad()
    loss.backward()
    y_grad = y_mannual.grad.to(pt_dtype)
    q_grad = q.grad
    qk_grad = kq.grad
    kk_grad = xk.grad
    v_grad = v.grad
    g_grad = g.grad

    with torch.no_grad():
        # xattn
        for level in range(2):
            atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
            rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            y_xattn, readers, lse = xattn_softdelta_attn_fwd(
                q, kq, xk, v, g, scale, bos_mask, k_segment_idx, level, True
            )
            torch.testing.assert_close(y_xattn, y_mannual.to(pt_dtype), rtol=rtol, atol=atol)
            print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, xattn-{level}: pass fwd test")

            atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
            rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            q_grad_xattn, qk_grad_xattn, kk_grad_xattn, v_grad_xattn, g_grad_xattn = xattn_softdelta_attn_bwd(
                y_grad, q, kq, xk, v, g, y_xattn, readers, lse, scale, bos_mask, k_segment_idx, level, False
            )
            torch.testing.assert_close(q_grad_xattn, q_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(qk_grad_xattn, qk_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(kk_grad_xattn, kk_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(v_grad_xattn, v_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(g_grad_xattn, g_grad, rtol=rtol, atol=atol)
            print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, xattn-{level}: pass bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L1, L2, H, D in [
        [1, 1, 1, 2, 8],
        [3, 1, 9, 3, 32],
        [1, 2048, 2048, 14, 128],
        [2, 2048, 4096, 32, 256],
        [1, 8192, 8192, 32, 192],
    ]:
        test(B, L1, L2, H, D, False, dtype)
        if L1 == L2:
            test(B, L1, L2, H, D, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
