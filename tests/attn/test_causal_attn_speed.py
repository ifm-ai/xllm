from timeit import default_timer as timer
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


def test_flash_attention_speed(
    query, key, value, out_grad, cu_seqlens, max_seqlen, scale, epochs
):
    with torch.no_grad():
        # warm up
        for i in range(100):
            deterministic = i % 2 == 0
            out, softmax_lse, rng_state = flash_attention_fwd(
                query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, scale, 0.0, True
            )
            flash_attention_bwd(
                out_grad, out, query, key, value, softmax_lse,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                scale, 0.0, rng_state, True, deterministic
            )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            out, softmax_lse, rng_state = flash_attention_fwd(
                query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, scale, 0.0, True
            )

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'flash attn fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            flash_attention_bwd(
                out_grad, out, query, key, value, softmax_lse,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                scale, 0.0, rng_state, True, True
            )
        torch.cuda.synchronize()

        delta1 = timer() - start

        start = timer()
        for _ in range(epochs):
            flash_attention_bwd(
                out_grad, out, query, key, value, softmax_lse,
                cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                scale, 0.0, rng_state, True, True
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'flash attn bwd: {delta:.2f}s ({delta1:.2f}s)')


def test_xattn_attention_speed(
    query, key, value, out_grad, bos_mask, segment_idx, high_prevision_level, scale, epochs
):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            y, y_bwd, lse = xattn_causal_flash_attn_fwd(
                query, key, value, scale, bos_mask, segment_idx, high_prevision_level, requires_grad=True
            )
            xattn_causal_flash_attn_bwd(
                out_grad, query, key, value, y_bwd, lse,
                scale, bos_mask, segment_idx, high_prevision_level, deterministic=False
            )

        torch.cuda.synchronize()

        api = 'bos' if bos_mask is not None else 'segidx'
        suffix = f"-{high_prevision_level}"

        start = timer()
        for _ in range(epochs):
            y, y_bwd, lse = xattn_causal_flash_attn_fwd(
                query, key, value, scale, bos_mask, segment_idx, high_prevision_level, requires_grad=True
            )

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'xattn-{api}{suffix} fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            xattn_causal_flash_attn_bwd(
                out_grad, query, key, value, y_bwd, lse,
                scale, bos_mask, segment_idx, high_prevision_level, deterministic=True
            )
        torch.cuda.synchronize()

        delta1 = timer() - start

        start = timer()
        for _ in range(epochs):
            xattn_causal_flash_attn_bwd(
                out_grad, query, key, value, y_bwd, lse,
                scale, bos_mask, segment_idx, high_prevision_level, deterministic=False
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'xattn-{api}{suffix} bwd: {delta:.2f}s ({delta1:.2f}s)')


def test(B: int, L: int, H: int, G: int, D: int, avg_len: int, dtype: str):
    scale = math.sqrt(1.0 / D)
    if avg_len == 0:
        avg_len = L
    bos_ratio = float(L - avg_len) / (L * avg_len)
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        query = torch.randn(B, L, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        key = torch.randn(B, L, G, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, L, G, D, requires_grad=False, dtype=pt_dtype, device="cuda")

        bos_mask = torch.rand(B, L, requires_grad=False, device='cuda') < bos_ratio
        segment_idx = torch.cumsum(bos_mask, dim=-1)
        bos_idx_k = torch.nonzero(F.pad(bos_mask[:, 1:], (1, 0), value=1))
        cu_seqlens_k = bos_idx_k[:, 0] * L + bos_idx_k[:, 1]
        cu_seqlens_k = F.pad(cu_seqlens_k, (0, 1), value=B * L)
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
        cu_seqlens_k = cu_seqlens_k.int()

        out_grad = torch.randn(B, L, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")

    query = query.detach()
    key = key.detach()
    value = value.detach()
    out_grad = out_grad.detach()
    bos_mask = bos_mask.detach()
    segment_idx = segment_idx.detach()

    print(f"B={B}, L={L}, H={H} ({G}), D={D}, AvgL={avg_len}, dtype={dtype}:")
    epochs = 5000
    test_flash_attention_speed(query, key, value, out_grad, cu_seqlens_k, max_seqlen_k, scale, epochs)
    for level in range(5):
        test_xattn_attention_speed(query, key, value, out_grad, bos_mask, segment_idx, level, scale, epochs)
    for level in range(5):
        test_xattn_attention_speed(query, key, value, out_grad, None, segment_idx, level, scale, epochs)

    print("*" * 70)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L, H, D in [
        [1, 2011, 16, 128],
        [2, 4096, 32, 128],
        [3, 8192, 32, 128],
        [4, 8192, 64, 128],
        [4, 8192, 32, 256],
        [2, 16384, 32, 128],
        [1, 32768, 32, 128],
    ]:
        for avg_len in [10, 100, 1000, 2000, 4000, 8000, 16000, 32000, 0]:
            if avg_len > L:
                continue

            test(B, L, H, H, D, avg_len, dtype)
            if H % 8 == 0:
                test(B, L, H, 8, D, avg_len, dtype)


if __name__ == "__main__":
    fire.Fire(main)
