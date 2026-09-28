from timeit import default_timer as timer
import math
import numpy as np
import torch
import torch.nn.functional as F
import fire

import flash_attn_3._C  # Registers operators with PyTorch
flash_attn_gpu = torch.ops.flash_attn_3

from xllm.modules.fused_ops import (
    sliding_chunk_attention_fwd,
    sliding_chunk_attention_bwd,
    swift_efficient_attention_fwd,
    swift_efficient_attention_bwd,
)
from xllm.modules.causal_attention import repeat_kv


def test_chunk_wise_attention_speed(
    query, key, value, out_grad, chunk_size, epochs
):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            bsz, seq_len, heads, _ = query.size()
            kv_heads = key.shape[2]
            nc = seq_len // chunk_size
            n_rep = heads // kv_heads
            q = query.view(bsz * nc, chunk_size, heads, -1)
            k = key.view(bsz * nc, chunk_size, kv_heads, -1)
            v = value.view(bsz * nc, chunk_size, kv_heads, -1)
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)
            y, w = swift_efficient_attention_fwd(
                q, k, v, None, None, use_causal_mask=True
            )

            y_grad = out_grad.view(bsz * nc, chunk_size, heads, -1)
            swift_efficient_attention_bwd(y_grad, q, k, v, w, use_causal_mask=True)
        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            bsz, seq_len, heads, _ = query.size()
            kv_heads = key.shape[2]
            nc = seq_len // chunk_size
            n_rep = heads // kv_heads
            q = query.view(bsz * nc, chunk_size, heads, -1)
            k = key.view(bsz * nc, chunk_size, kv_heads, -1)
            v = value.view(bsz * nc, chunk_size, kv_heads, -1)
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

            y, w = swift_efficient_attention_fwd(
                q, k, v, None, None, use_causal_mask=True
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'chunk-wise attention fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            bsz, seq_len, heads, _ = query.size()
            nc = seq_len // chunk_size
            y_grad = out_grad.view(bsz * nc, chunk_size, heads, -1)
            swift_efficient_attention_bwd(y_grad, q, k, v, w, use_causal_mask=True)
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'chunk-wise attention bwd: {delta:.2f}s')


def test_sliding_chunk_attention_speed(
    query, key, value, out_grad, chunk_size, bos_mask, segment_idx, fp32_output, backend, epochs
):
    with torch.no_grad():
        # warm up
        for i in range(100):
            deterministic = i % 2 == 0
            _, y_bwd, aux = sliding_chunk_attention_fwd(
                query, key, value, chunk_size, 1.0, None, None,
                bos_mask, segment_idx, 0.0, fp32_output, backend, True
            )
            sliding_chunk_attention_bwd(
                out_grad, query, key, value, y_bwd, aux, chunk_size, 1.0, None, None,
                bos_mask, segment_idx, deterministic, backend
            )
        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            _, y_bwd, aux = sliding_chunk_attention_fwd(
                query, key, value, chunk_size, 1.0, None, None,
                bos_mask, segment_idx, 0.0, fp32_output, backend, True
            )
        torch.cuda.synchronize()

        suffix = "-fp32" if fp32_output else ""
        delta = timer() - start
        if backend == 'swift':
            print(f'SCA-{backend} fwd: {delta:.2f}s')
        else:
            api = 'bos' if bos_mask is not None else 'segidx'
            print(f'SCA-{backend}-{api}{suffix} fwd: {delta:.2f}s')

        deterministic = True
        start = timer()
        for _ in range(epochs):
            sliding_chunk_attention_bwd(
                out_grad, query, key, value, y_bwd, aux, chunk_size, 1.0, None, None,
                bos_mask, segment_idx, deterministic, backend
            )
        torch.cuda.synchronize()

        delta1 = timer() - start
        if backend == 'swift':
            print(f'SCA-{backend} bwd: {delta1:.2f}s')
        else:
            deterministic = False
            start = timer()
            for _ in range(epochs):
                sliding_chunk_attention_bwd(
                    out_grad, query, key, value, y_bwd, aux, chunk_size, 1.0, None, None,
                    bos_mask, segment_idx, deterministic, backend
                )
            torch.cuda.synchronize()

            delta = timer() - start
            api = 'bos' if bos_mask is not None else 'segidx'
            print(f'SCA-{backend}-{api}{suffix} bwd: {delta:.2f}s ({delta1:.2f}s)')


def test_flash_sliding_window_attention_speed(
    query, key, value, out_grad, chunk_size, scale, epochs
):
    with torch.no_grad():
        window_size = (chunk_size, 0)
        q_grad, k_grad, v_grad = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
        # warm up
        for _ in range(100):
            out, softmax_lse, *rest = flash_attn_gpu.fwd(
                query, key, value,  None, None, None, None, None, None, None,
                None, None, None, None, None, None, None,
                None, None, None, None, None, None,
                scale, True, window_size[0], window_size[1], 0, 0.0,
                True, None, 1, None, 0,
            )
            flash_attn_gpu.bwd(
                out_grad, query, key, value, out, softmax_lse, q_grad, k_grad, v_grad,
                None, None, None, None, None, None,  # cu_seqlens_q/k, seqused_q/k, max_seqlen_q/k
                scale, True, window_size[0], window_size[1], 0.0, False, 0,
            )
        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            out, softmax_lse, *rest = flash_attn_gpu.fwd(
                query, key, value, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None,
                None, None, None, None, None, None,
                scale, True, window_size[0], window_size[1], 0, 0.0,
                True, None, 1, None, 0,
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'flash3 SWA fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            flash_attn_gpu.bwd(
                out_grad, query, key, value, out, softmax_lse, q_grad, k_grad, v_grad,
                None, None, None, None, None, None,  # cu_seqlens_q/k, seqused_q/k, max_seqlen_q/k
                scale, True, window_size[0], window_size[1], 0.0, True, 0,
            )
        torch.cuda.synchronize()

        delta1 = timer() - start

        start = timer()
        for _ in range(epochs):
            flash_attn_gpu.bwd(
                out_grad, query, key, value, out, softmax_lse, q_grad, k_grad, v_grad,
                None, None, None, None, None, None,  # cu_seqlens_q/k, seqused_q/k, max_seqlen_q/k
                scale, True, window_size[0], window_size[1], 0.0, False, 0,
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'flash3 SWA bwd: {delta:.2f}s ({delta1:.2f}s)')


def test(B: int, L: int, H: int, HKV: int, D: int, V: int, chunk_size: int, avg_len: int, dtype: str):
    assert L % chunk_size == 0
    if avg_len == 0:
        avg_len = L
    bos_ratio = float(L - avg_len) / (L * avg_len)
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        query = torch.randn(B, L, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        key = torch.randn(B, L, HKV, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, L, HKV, V, requires_grad=False, dtype=pt_dtype, device="cuda")

        bos_mask = torch.rand(B, L, requires_grad=False, device='cuda') < bos_ratio
        segment_idx = torch.cumsum(bos_mask, dim=-1)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        value = F.silu(value) + 0.1

    query = query.detach()
    key = key.detach()
    value = value.detach()
    segment_idx = segment_idx.detach()

    out_grad = torch.randn(B, L, H, V, requires_grad=False, dtype=pt_dtype, device="cuda")

    epochs = 1000
    print(f"B={B}, L={L}, H={H} ({HKV}), D={D}, V={V}, chunk={chunk_size}, AvgL={avg_len}, dtype={dtype}:")

    # test_chunk_wise_attention_speed(query, key, value, out_grad, chunk_size * 2, epochs)
    # if D == V and dtype in ['bf16', 'fp16']:
    #     test_flash_sliding_window_attention_speed(query, key, value, out_grad, chunk_size * 2, 1.0 / math.sqrt(D), epochs)
    test_sliding_chunk_attention_speed(query, key, value, out_grad, chunk_size, bos_mask, segment_idx, False, 'swift', epochs)
    test_sliding_chunk_attention_speed(query, key, value, out_grad, chunk_size, bos_mask, segment_idx, False, 'xattn', epochs)
    test_sliding_chunk_attention_speed(query, key, value, out_grad, chunk_size, bos_mask, segment_idx, True, 'xattn', epochs)
    test_sliding_chunk_attention_speed(query, key, value, out_grad, chunk_size, None, segment_idx, False, 'xattn', epochs)
    test_sliding_chunk_attention_speed(query, key, value, out_grad, chunk_size, None, segment_idx, True, 'xattn', epochs)
    print("*" * 70)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, H, D, chunk in [
        # [1, 8192, 4, 256, 2048],
        # [1, 16384, 4, 256, 2048],
        # [2, 16384, 4, 256, 2048],
        # [1, 16384, 4, 256, 4096],
        # [1, 16384, 8, 256, 4096],
        # [1, 16384, 16, 128, 4096],
        # [1, 32768, 4, 256, 4096],
        # [1, 32768, 8, 256, 4096],
        # [1, 32768, 16, 128, 4096],
        # [1, 65536, 8, 256, 4096],
        [4, 65536, 8, 128, 4096],
        [4, 65536, 8, 256, 4096],
        # [1, 65536, 16, 128, 4096],
    ]:
        for avg_len in [10, 100, 1000, 2000, 4000, 8000, 16000, 32000, 0]:
            if D == 128:
                test(B, L, H, H, D, D, chunk, avg_len, dtype)
            else:
                test(B, L, H, H // 2, D, D, chunk, avg_len, dtype)
            if D * 2 <= 256:
                test(B, L, H, H, D, D * 2, chunk, avg_len, dtype)
                # test(B, L, H, H // 4, D, D * 2, chunk, dtype)


if __name__ == "__main__":
    fire.Fire(main)
