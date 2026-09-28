import math
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import fire
from einops import repeat

from xllm.modules.fused_ops import (
    adaptive_working_memory_fwd,
    adaptive_working_memory_accum_fwd,
    adaptive_working_memory_bwd,
)


def manual_rejection(x, h):
    dim = h.shape[-1]
    inv_scale = 1.0 / float(dim)
    # [B, *, S] x [B, *,  S] -> [B, *, 1]
    alpha = torch.sum(x * h, dim=-1, keepdim=True) * inv_scale
    y = torch.addcmul(x, alpha, h, value=-1.0)
    return y


def manual_awk(
    xq: Tensor,  # B x H x L x S
    xqk: Tensor,  # B x K x L x S
    xkk: Tensor,  # B x K x L x S
    xv: Tensor,  # B x K x L x V
    chunk_size: int,
    bos_mask: Tensor,  # B x L
    segment_idx: Tensor,  # B x L
    ortho: bool
):
    bsz, n_heads, length, qk_head_dim = xq.shape
    _, n_kv_heads, _, v_head_dim = xv.shape
    n_rep = n_heads // n_kv_heads

    assert length % chunk_size == 0
    eps = 1e-6

    y = []
    memory = torch.zeros(bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=xq.dtype, device=xq.device)
    next_memory = torch.zeros(bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=xq.dtype, device=xq.device)
    update_memory = torch.zeros(bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=xq.dtype, device=xq.device)
    norm_term = torch.full((bsz, n_kv_heads, qk_head_dim), 0.0, device=xq.device)
    prev_segment_count = torch.full((bsz,), -1, dtype=torch.int64, device=xq.device) if segment_idx is not None else None
    nc = length // chunk_size
    for c in range(nc):
        start = c * chunk_size
        end = (c + 1) * chunk_size
        # B x H x c x S/H
        q = xq[:, :, start:end]
        qk = xqk[:, :, start:end]
        kk = xkk[:, :, start:end]
        v = xv[:, :, start:end]
        # B x C
        boss = bos_mask[:, start:end] if bos_mask is not None else None
        sidx = segment_idx[:, start:end] if segment_idx is not None else None

        curr_norm_term = torch.full((bsz, n_kv_heads, qk_head_dim), 0.0, device=xq.device)
        for t in range(chunk_size):
            # B
            if boss is not None:
                bos = boss[:, t]
                curr_norm_term = curr_norm_term.masked_fill(bos.view(bsz, 1, 1), 0)
                # mask out stale norm terms
                norm_term = norm_term.masked_fill(bos.view(bsz, 1, 1), 0)

            curr_norm_term = curr_norm_term + torch.exp(kk[:, :, t])

        prev_memory = next_memory
        for t in range(chunk_size):
            # B x H x 1 x S/H
            qt = q[:, :, t:(t + 1)]
            qkt = qk[:, :, t:(t + 1)]
            kt = kk[:, :, t:(t + 1)]
            vt = v[:, :, t:(t + 1)]
            # B
            bos = boss[:, t] if boss is not None else None
            sid = sidx[:, t] if sidx is not None else None

            # (B x H x 1 x S/H) x (B x H x S/H, V/H) -> B x H x 1 x V/H
            yt = torch.matmul(qt, repeat(memory, 'b k s v -> b (k n) s v', n=n_rep))
            if sid is not None:
                y_mask = torch.eq(sid, prev_segment_count).view(bsz, 1, 1, 1)
                yt = yt * y_mask.to(qt)
            y.append(yt.squeeze(2))

            if bos is not None:
                prev_memory = prev_memory.masked_fill(bos.view(bsz, 1, 1, 1), 0.0)
                update_memory = update_memory.masked_fill(bos.view(bsz, 1, 1, 1), 0.0)

            # B x K x 1 x V/H
            if ortho:
                rv = torch.matmul(qkt, prev_memory)
                # B x K x 1 x 1
                norm = torch.mean(torch.square(rv), dim=-1, keepdim=True)
                norm = torch.sqrt(norm + np.float64(eps))
                rv = rv / norm
                uv = manual_rejection(vt, rv)
            else:
                uv = vt - torch.matmul(qkt, prev_memory)
            # B x K x 1 x S/H
            uk = torch.exp(kt) / curr_norm_term.unsqueeze(2)
            # B x K x S/H x V/H
            kvb = torch.matmul(uk.transpose(2, 3), uv)
            update_memory = update_memory + kvb

        norm_term = norm_term + curr_norm_term
        # B x K x S/H x 1
        ratio = (curr_norm_term / norm_term).unsqueeze(3)
        memory = next_memory
        next_memory = (1.0 - ratio) * prev_memory + ratio * update_memory
        update_memory = torch.zeros(bsz, n_kv_heads, qk_head_dim, v_head_dim, dtype=xq.dtype, device=xq.device)
        if start > 0 and segment_idx is not None:
            prev_segment_count = segment_idx[:, start - 1]

    # B x H x L x V/H
    y = torch.stack(y, dim=2)
    return y, memory


def awk_fwd(
    xq: Tensor,  # B x H x L x S
    xqk: Tensor,  # B x K x L x S
    xkk: Tensor,  # B x K x L x S
    xv: Tensor,  # B x K x L x V
    memory,
    log_norm_term,
    prev_qk,
    prev_kk,
    prev_v,
    chunk_size,
    segment_idx,
    prev_segment_count,
    ortho,
    eps,
):
    bsz, n_heads, length, qk_head_dim = xq.shape
    _, n_kv_heads, _, v_head_dim = xv.shape

    out, _, memory, log_norm_term = adaptive_working_memory_fwd(
        xq, xqk, xkk, xv, chunk_size, memory, log_norm_term, prev_qk, prev_kk, prev_v,
        segment_idx, prev_segment_count, ortho, eps
    )

    prev_qk = xqk[:, :, length - chunk_size:]
    prev_kk = xkk[:, :, length - chunk_size:]
    prev_v = xv[:, :, length - chunk_size:]
    if segment_idx is not None:
        prev_segment_idx = segment_idx[:, length - chunk_size:]
        prev_segment_count = segment_idx[:, max(length - chunk_size - 1, 0)]
    else:
        prev_segment_idx = None
        prev_segment_count = None
    return out, memory, log_norm_term, prev_qk, prev_kk, prev_v, prev_segment_idx, prev_segment_count


def awk_bwd(
    y_grad, # B x H x L x V
    mem_grad, # B x K x S x V
    lnt_grad, # B x K x S
    xq: Tensor,  # B x H x L x S
    xqk: Tensor,  # B x K x L x S
    xkk: Tensor,  # B x K x L x S
    xv: Tensor,  # B x K x L x V
    memory,
    log_norm_term,
    prev_qk,
    prev_kk,
    prev_v,
    chunk_size,
    segment_idx,
    prev_segment_count,
    ortho,
    eps,
):
    memory_outs, lnt_outs, kv_outs, kk_outs, prev_outs = adaptive_working_memory_accum_fwd(
        xq, xqk, xkk, xv, chunk_size, memory, log_norm_term, prev_qk, prev_kk, prev_v,
        segment_idx, prev_segment_count, ortho, eps, False
    )
    accum_memory, memory_residual, memory_mask = memory_outs
    log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio = lnt_outs
    awk_kkey, awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue, awk_out_rec, awk_out_mask = kv_outs
    ak_fp32, akk_fp32, awk_key_mask = kk_outs
    prev_kkey, prev_k_fp32, prev_kk_fp32, prev_k_mask = prev_outs
    q_grad, qk_grad, kk_grad, v_grad, mem_grad, lnt_grad, prev_qk_grad, prev_kk_grad, prev_v_grad = adaptive_working_memory_bwd(
        y_grad, mem_grad, lnt_grad,
        xq, xqk, awk_kkey, xv, chunk_size,
        awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue,
        ak_fp32, akk_fp32, awk_key_mask,
        accum_memory, memory_residual, memory_mask,
        log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio, awk_out_mask,
        prev_qk, prev_kkey, prev_v, prev_k_fp32, prev_kk_fp32, prev_k_mask, ortho
    )
    return q_grad, qk_grad, kk_grad, v_grad, mem_grad, lnt_grad, prev_qk_grad, prev_kk_grad, prev_v_grad


def test(B: int, L: int, H: int, HKV: int, D: int, chunk_size: int, ortho: bool, multi_segment: bool, dtype: str):
    bos_ratio = 0.2
    eps = 1e-6
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        query = torch.randn(B, H, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        key = torch.randn(B, HKV, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, HKV, L, D * 4, requires_grad=False, dtype=pt_dtype, device="cuda")

        query = F.softmax(query, dim=-1, dtype=torch.float32).to(query)
        qkey = F.softmax(key, dim=-1, dtype=torch.float32).to(key)

        bos_mask = torch.rand(B, L, requires_grad=False, device='cuda') < bos_ratio if multi_segment else None
        segment_idx = torch.cumsum(bos_mask, dim=-1) if multi_segment else None

    q = query.clone().detach().requires_grad_(True)
    qk = qkey.clone().detach().requires_grad_(True)
    kk = key.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)

    y_manual, memory_manual = manual_awk(
        q.double(), qk.double(), kk.double(), v.double(), chunk_size, bos_mask, segment_idx, ortho
    )
    y_manual_flat = y_manual.flatten()
    num_elem_y = y_manual_flat.shape[0]
    weight_y = torch.randn(num_elem_y, 1, requires_grad=False, dtype=torch.double, device="cuda") / max(8.0, math.sqrt(L))
    loss = y_manual_flat @ weight_y
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(pt_dtype)
    q_grad = q.grad
    qk_grad = qk.grad
    kk_grad = kk.grad
    v_grad = v.grad

    with torch.no_grad():
        atol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]

        q1, q2 = q.chunk(2, dim=2)
        qk1, qk2 = qk.chunk(2, dim=2)
        kk1, kk2 = kk.chunk(2, dim=2)
        v1, v2 = v.chunk(2, dim=2)
        y1_grad, y2_grad = y_grad.chunk(2, dim=2)
        if segment_idx is None:
            seg1, seg2 = None, None
        else:
            seg1, seg2 = segment_idx.chunk(2, dim=1)

        y1, memory1, log_norm_term1, prev_qk1, prev_kk1, prev_v1, prev_seg1, prev_count1 = awk_fwd(
            q1, qk1, kk1, v1, None, None, None, None, None, chunk_size, seg1, None, ortho, eps,
        )
        if segment_idx is not None:
            seg2 = torch.cat([prev_seg1, seg2], dim=1)
        y2, memory2, log_norm_term2, prev_qk2, prev_kk2, prev_v2, prev_seg2, prev_count2 = awk_fwd(
            q2, qk2, kk2, v2, memory1, log_norm_term1, prev_qk1, prev_kk1, prev_v1, chunk_size, seg2, prev_count1, ortho, eps
        )

        torch.testing.assert_close(torch.cat([y1, y2], dim=2), y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        torch.testing.assert_close(memory2, memory_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={B}, L={L}, H={H} ({HKV}), D={D}, chunk={chunk_size}, ortho={ortho}, mseg={multi_segment}, dtype={dtype}: pass fwd test")

        q2_grad, qk2_grad, kk2_grad, v2_grad, mem_grad, lnt_grad, prev_qk1_grad, prev_kk1_grad, prev_v1_grad = awk_bwd(
            y2_grad, None, None, q2, qk2, kk2, v2, memory1, log_norm_term1,
            prev_qk1, prev_kk1, prev_v1, chunk_size, seg2, prev_count1, ortho, eps
        )
        q1_grad, qk1_grad, kk1_grad, v1_grad, mem_grad, lnt_grad, prev_qk_grad, prev_kk_grad, prev_v_grad = awk_bwd(
            y1_grad, mem_grad, lnt_grad, q1, qk1, kk1, v1, None, None, None, None, None, chunk_size, seg1, None, ortho, eps
        )
        qk1_grad[:, :, -chunk_size:] += prev_qk1_grad
        kk1_grad[:, :, -chunk_size:] += prev_kk1_grad
        v1_grad[:, :, -chunk_size:] += prev_v1_grad

        atol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        torch.testing.assert_close(torch.cat([q1_grad, q2_grad], dim=2), q_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(torch.cat([qk1_grad, qk2_grad], dim=2), qk_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(torch.cat([kk1_grad, kk2_grad], dim=2), kk_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(torch.cat([v1_grad, v2_grad], dim=2), v_grad, rtol=rtol, atol=atol)
        assert mem_grad is None
        assert lnt_grad is None
        assert prev_qk_grad is None
        assert prev_kk_grad is None
        assert prev_v_grad is None
        print(f"B={B}, L={L}, H={H} ({HKV}), D={D}, chunk={chunk_size}, ortho={ortho}, mseg={multi_segment}, dtype={dtype}: pass bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L, H, D, chunk in [
        [1, 8, 1, 128, 2],
        [2, 8, 2, 256, 2],
        [1, 128, 8, 128, 16],
        [2, 1024, 4, 256, 32],
        [2, 8192, 8, 128, 2048],
        [2, 16384, 4, 139, 2048],
        [1, 32768, 5, 128, 4096],
    ]:
        for ortho in [False, True]:
            test(B, L, H, H, D, chunk, ortho, False, dtype)
            test(B, L, H, H, D, chunk, ortho, True, dtype)
            if H >= 4:
                test(B, L, H, H // 4, D, chunk, ortho, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
