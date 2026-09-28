import math
import numpy as np
import torch
import torch.nn.functional as F
import fire

from xllm.modules.fused_ops import (
    flash_attention_fwd,
    flash_attention_bwd,
    xattn_causal_flash_attn_fwd,
    xattn_causal_flash_attn_bwd
)


def manual_attention(
    query, key, value, q_segment_idx, k_segment_idx, scale
):
    bsz, qlen, _, _ = query.shape
    kvlen = key.shape[1]

    # B x L1 x L2
    attn_mask = torch.full((bsz, qlen, kvlen), float("-inf"), device=query.device)
    attn_mask = torch.triu(attn_mask, diagonal=(kvlen - qlen + 1)).type_as(query)

    if q_segment_idx is not None:
        seg_mask = torch.ne(q_segment_idx.unsqueeze(2), k_segment_idx.unsqueeze(1))
        attn_mask = attn_mask.masked_fill(seg_mask, value=float("-inf"))

    # B x H x L x D
    xq = query.transpose(1, 2)
    xk = key.transpose(1, 2)
    xv = value.transpose(1, 2)
    # B x H x L x L
    scores = torch.matmul(xq, xk.transpose(2, 3)) * scale
    scores = scores + attn_mask.unsqueeze(1)
    scores = F.softmax(scores, dim=-1, dtype=xq.dtype)
    # B x H x L x S -> B x L x H x S
    output = torch.matmul(scores, xv)
    output = output.transpose(1, 2)
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
        key = torch.randn(B, L2, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, L2, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")

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
    k = key.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)

    y_mannual = manual_attention(q.float(), k.float(), v.float(), k_segment_idx, k_segment_idx, scale)

    y_mannual_flat = y_mannual.flatten()
    num_elem = y_mannual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.float32, device="cuda") / math.sqrt(L1)
    # weight = torch.ones(num_elem, requires_grad=False, dtype=torch.double, device="cuda")
    loss = y_mannual_flat @ weight
    y_mannual.retain_grad()
    loss.backward()
    y_grad = y_mannual.grad.to(pt_dtype)
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad

    with torch.no_grad():
        # flash-attn
        atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]

        y_flash, lse, rng_state = flash_attention_fwd(
            q, k, v, cu_seqlens_k, cu_seqlens_k, max_seqlen_k, max_seqlen_k, scale, 0.0, True
        )
        torch.testing.assert_close(y_flash, y_mannual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, flash: pass fwd test")

        atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        q_grad_flash, k_grad_flash, v_grad_flash = flash_attention_bwd(
            y_grad, y_flash, q, k, v, lse, cu_seqlens_k, cu_seqlens_k, max_seqlen_k, max_seqlen_k,
            scale, 0.0, rng_state, True, False
        )
        torch.testing.assert_close(q_grad_flash, q_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_grad_flash, k_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(v_grad_flash, v_grad, rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, flash: pass bwd test")

        # xattn
        atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        y_xattn, y_bwd, lse = xattn_causal_flash_attn_fwd(
            q, k, v, scale, bos_mask, k_segment_idx, False, True
        )
        torch.testing.assert_close(y_xattn, y_mannual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, xattn: pass fwd test")

        atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        q_grad_xattn, k_grad_xattn, v_grad_xattn = xattn_causal_flash_attn_bwd(
            y_grad, q, k, v, y_bwd, lse, scale, bos_mask, k_segment_idx, False
        )
        torch.testing.assert_close(q_grad_xattn, q_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_grad_xattn, k_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(v_grad_xattn, v_grad, rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, xattn: pass bwd test")

        # xattn-fp32
        atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        y_xattn, y_bwd, lse = xattn_causal_flash_attn_fwd(
            q, k, v, scale, bos_mask, k_segment_idx, True, True
        )
        torch.testing.assert_close(y_xattn, y_mannual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, xattn-fp32: pass fwd test")

        atol = {"fp32": 1e-6, "bf16": 4e-3, "fp16": 2e-4}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        q_grad_xattn, k_grad_xattn, v_grad_xattn = xattn_causal_flash_attn_bwd(
            y_grad, q, k, v, y_bwd, lse, scale, bos_mask, k_segment_idx, False
        )
        torch.testing.assert_close(q_grad_xattn, q_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_grad_xattn, k_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(v_grad_xattn, v_grad, rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, seg={multi_segment}, dtype={dtype}, xattn-fp32: pass bwd test")


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
