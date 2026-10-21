import math
import numpy as np
import torch
import torch.nn.functional as F
import fire

from xllm.modules.fused_ops import (
    sliding_chunk_attention_fwd,
    sliding_chunk_attention_bwd,
)
from xllm.modules.causal_attention import repeat_kv


def mannual_sliding_window_attention(
    query, key, value, prev_key_chunk, prev_value_chunk, chunk_size, segment_idx, prev_segment_idx,
):
    bsz, seq_len, n_heads, _ = query.shape
    n_kv_heads = key.shape[2]
    assert seq_len == key.shape[1] and seq_len == value.shape[1]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads
    outputs = []
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        q_idx = segment_idx[:, qbos:eos]
        k_idx = segment_idx[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)
            k_idx = torch.cat([prev_segment_idx, k_idx], dim=1)
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        ctx_len = k.shape[1]
        attn_mask = torch.full((bsz, chunk_size, ctx_len), float("-inf"), device="cuda")
        attn_mask = torch.triu(attn_mask, diagonal=(ctx_len - chunk_size + 1)).type_as(q)
        seg_mask = torch.ne(q_idx.unsqueeze(2), k_idx.unsqueeze(1))
        attn_mask = attn_mask.masked_fill(seg_mask, value=float("-inf"))

        xq = q.transpose(1, 2)
        xk = k.transpose(1, 2)
        xv = v.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3))
        scores = scores + attn_mask.unsqueeze(1)
        scores = F.softmax(scores, dim=-1, dtype=xq.dtype)
        y = torch.matmul(scores, xv).transpose(1, 2)
        outputs.append(y)

    y = torch.cat(outputs, dim=1)
    return y


def test(B: int, L: int, H: int, HKV: int, D: int, V: int, chunk_size: int, prev_chunk: bool, dtype: str):
    bos_ratio = 0.001
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        query = torch.randn(B, L, H, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        key = torch.randn(B, L, HKV, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, L, HKV, V, requires_grad=False, dtype=pt_dtype, device="cuda")

        bos_mask = torch.rand(B, L, requires_grad=False, device='cuda') < bos_ratio

        if prev_chunk:
            prev_key = torch.randn(B, chunk_size, HKV, D, requires_grad=False, dtype=pt_dtype, device="cuda")
            prev_value = torch.randn(B, chunk_size, HKV, V, requires_grad=False, dtype=pt_dtype, device="cuda")
            prev_key = F.normalize(prev_key, dim=-1)
            prev_value = F.silu(prev_value) + 0.1
            prev_bos_mask = torch.rand(B, chunk_size, requires_grad=False, device='cuda') < bos_ratio
            bos_mask = torch.cat([prev_bos_mask, bos_mask], dim=1)
            segment_idx = torch.cumsum(bos_mask, dim=-1)[:, chunk_size:]
            prev_segment_idx = torch.cumsum(prev_bos_mask, dim=-1)
        else:
            prev_key = None
            prev_value = None
            segment_idx = torch.cumsum(bos_mask, dim=-1)
            prev_segment_idx = None

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        value = F.silu(value) + 0.1

    q = query.clone().detach().requires_grad_(True)
    k = key.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)
    if prev_chunk:
        prev_k = prev_key.clone().detach().requires_grad_(True)
        prev_v = prev_value.clone().detach().requires_grad_(True)
        y_mannual = mannual_sliding_window_attention(
            q.double(), k.double(), v.double(), prev_k.double(), prev_v.double(), chunk_size, segment_idx, prev_segment_idx,
        )
    else:
        prev_k = None
        prev_v = None
        y_mannual = mannual_sliding_window_attention(
            q.double(), k.double(), v.double(), prev_k, prev_v, chunk_size, segment_idx, prev_segment_idx,
        )

    y_mannual_flat = y_mannual.flatten()
    num_elem = y_mannual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(L)
    loss = y_mannual_flat @ weight
    y_mannual.retain_grad()
    loss.backward()
    y_grad = y_mannual.grad.to(pt_dtype)
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad
    if prev_chunk:
        prev_k_grad = prev_k.grad
        prev_v_grad = prev_v.grad
    else:
        prev_k_grad = None
        prev_v_grad = None

    if prev_segment_idx is not None:
        segment_idx = torch.cat([prev_segment_idx, segment_idx], dim=1)

    with torch.no_grad():
        for backend, mode, level in [
            ['swift', 'segidx', 0],
            ['xattn', 'bos', 0],
            ['xattn', 'segidx', 0],
            ['xattn', 'bos', 1],
            ['xattn', 'segidx', 1],
            ['xattn', 'bos', 2],
            ['xattn', 'segidx', 2],
            ['xattn', 'bos', 3],
            ['xattn', 'segidx', 3],
            ['xattn', 'bos', 4],
            ['xattn', 'segidx', 4],
        ]:
            if mode == 'bos':
                bmask = bos_mask
                segidx = None
            elif mode == 'segidx':
                bmask = None
                segidx = segment_idx
            else:
                raise ValueError(f"unknown mode: {mode}")

            suffix = f"{level}"

            atol = {"fp32": 1e-6, "bf16": 1e-3, "fp16": 2e-4}[dtype]
            rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]

            y_sca, y_bwd, aux = sliding_chunk_attention_fwd(
                q, k, v, chunk_size, 1.0, prev_k, prev_v, bmask, segidx,
                0.0, level, backend, requires_grad=True
            )
            torch.testing.assert_close(y_sca, y_mannual.to(pt_dtype), rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, H={H} ({HKV}), D={D}, V={V} chunk={chunk_size}, prev_chunk={prev_chunk}, "
                  f"backend={backend}-{mode}-{suffix}, dtype={dtype}: pass fwd test")

            atol = {"fp32": 1e-6, "bf16": 2e-3, "fp16": 2e-4}[dtype]
            rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]

            # non-deterministic
            q_grad_sca, k_grad_sca, v_grad_sca, prev_k_grad_sca, prev_v_grad_sca = sliding_chunk_attention_bwd(
                y_grad, q, k, v, y_bwd, aux, chunk_size, 1.0,
                prev_k, prev_v, bmask, segidx, level, False, backend
            )
            torch.testing.assert_close(q_grad_sca, q_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(k_grad_sca, k_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(v_grad_sca, v_grad, rtol=rtol, atol=atol)
            if prev_chunk:
                torch.testing.assert_close(prev_k_grad_sca, prev_k_grad, rtol=rtol, atol=atol)
                torch.testing.assert_close(prev_v_grad_sca, prev_v_grad, rtol=rtol, atol=atol)
            else:
                assert prev_k_grad_sca is None and prev_v_grad_sca is None
            print(f"B={B}, L={L}, H={H} ({HKV}), D={D}, V={V}, chunk={chunk_size}, prev_chunk={prev_chunk}, "
                  f"backend={backend}-{mode}-{suffix}-0, dtype={dtype}: pass bwd test")

            # deterministic
            q_grad_sca, k_grad_sca, v_grad_sca, prev_k_grad_sca, prev_v_grad_sca = sliding_chunk_attention_bwd(
                y_grad, q, k, v, y_bwd, aux, chunk_size, 1.0,
                prev_k, prev_v, bmask, segidx, level, True, backend
            )
            torch.testing.assert_close(q_grad_sca, q_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(k_grad_sca, k_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(v_grad_sca, v_grad, rtol=rtol, atol=atol)
            if prev_chunk:
                torch.testing.assert_close(prev_k_grad_sca, prev_k_grad, rtol=rtol, atol=atol)
                torch.testing.assert_close(prev_v_grad_sca, prev_v_grad, rtol=rtol, atol=atol)
            else:
                assert prev_k_grad_sca is None and prev_v_grad_sca is None
            print(f"B={B}, L={L}, H={H} ({HKV}), D={D}, V={V}, chunk={chunk_size}, prev_chunk={prev_chunk}, "
                  f"backend={backend}-{mode}-{suffix}-1, dtype={dtype}: pass bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L, chunk in [
        [1, 2048, 512],
        [2, 3072, 1024],
        [3, 16384, 2048],
        [1, 32768, 4096],
        [1, 32768, 2048]
    ]:
        for H in [1, 2, 3, 4, 6, 8, 16, 32, 64]:
            for G in range(1, H + 1):
                if H % G == 0:
                    for D in [32, 64, 96, 128, 192, 256]:
                        if H * D <= 8192:
                            test(B, L, H, G, D, D, chunk, False, dtype)
                            test(B, L, H, G, D, D, chunk, True, dtype)
                            if 2 * D <= 256:
                                test(B, L, H, G, D, 2 * D, chunk, False, dtype)
                                test(B, L, H, G, D, 2 * D, chunk, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
