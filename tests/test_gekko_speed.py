from timeit import default_timer as timer
import math
import numpy as np
import torch
import torch.nn.functional as F
import fire

from xllm.modules.fused_ops import (
    sliding_chunk_attention_fwd,
    sliding_chunk_attention_bwd,
    adaptive_working_memory_fwd,
    adaptive_working_memory_accum_fwd,
    adaptive_working_memory_bwd,
    causal_conv1d_fwd,
    causal_conv1d_bwd,
)
import xllm_extension.ops as xllm_ops


def awk_fwd(
    xq,
    xk,
    xv,
    memory,
    log_norm_term,
    prev_k,
    prev_v,
    chunk_size,
    segment_idx,
    prev_segment_count,
    ortho,
):
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    return adaptive_working_memory_fwd(
        xq, xk, xk, xv, chunk_size, memory, log_norm_term, prev_k, prev_k, prev_v,
        segment_idx, prev_segment_count, ortho, 1e-6
    )


def awk_bwd(
    y_grad,
    mem_grad,
    lnt_grad,
    xq,
    xk,
    xv,
    memory,
    log_norm_term,
    prev_k,
    prev_v,
    chunk_size,
    segment_idx,
    prev_segment_count,
    ortho,
):
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    y_grad = y_grad.transpose(1, 2)
    memory_outs, lnt_outs, kv_outs, kk_outs, prev_outs = adaptive_working_memory_accum_fwd(
        xq, xk, xk, xv, chunk_size, memory, log_norm_term, prev_k, prev_k, prev_v,
        segment_idx, prev_segment_count, ortho, 1e-6, False
    )
    accum_memory, memory_residual, memory_mask = memory_outs
    log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio = lnt_outs
    awk_kkey, awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue, awk_out_rec, awk_out_mask = kv_outs
    ak_fp32, akk_fp32, awk_key_mask = kk_outs
    prev_kkey, prev_k_fp32, prev_kk_fp32, prev_k_mask = prev_outs
    return adaptive_working_memory_bwd(
        y_grad, mem_grad, lnt_grad,
        xq, xk, awk_kkey, xv, chunk_size,
        awk_rvalue, awk_rvalue_rstd, awk_vvalue, awk_avalue,
        ak_fp32, akk_fp32, awk_key_mask,
        accum_memory, memory_residual, memory_mask,
        log_norm_term, curr_log_norm_term, accum_log_norm_term, ratio, awk_out_mask,
        prev_k, prev_kkey, prev_v, prev_k_fp32, prev_kk_fp32, prev_k_mask, ortho
    )


def test_decay_timestep_norm(
    x, bos_mask, prev_count, prev_mean, prev_var, weight, bias,
    y_grad, mean_grad, var_grad, num_groups, beta1, beta2, eps, epochs, backend,
):
    tsdn_fwd = xllm_ops.group_timestep_decay_norm_fwd if backend == 'chunkwise' else xllm_ops.group_timestep_decay_norm_cub_fwd
    with torch.no_grad():
        # warm up
        for _ in range(100):
            y, count, mean, var, cummean, cumrstd = tsdn_fwd(
                x, bos_mask, prev_count, prev_mean, prev_var, weight, bias,
                None, num_groups, beta1, beta2, eps,
            )
            if backend == 'chunkwise':
                xllm_ops.group_timestep_decay_norm_bwd(
                    y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd,
                    weight, bias, None, num_groups, beta1, beta2, eps, False
                )
            else:
                xllm_ops.group_timestep_decay_norm_cub_bwd(
                    y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd,
                    weight, None, num_groups, beta1, beta2
                )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            y, count, mean, var, cummean, cumrstd = tsdn_fwd(
                x, bos_mask, prev_count, prev_mean, prev_var, weight, bias,
                None, num_groups, beta1, beta2, eps,
            )

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'{backend} timestep decay norm fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            if backend == 'chunkwise':
                xllm_ops.group_timestep_decay_norm_bwd(
                    y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd,
                    weight, bias, None, num_groups, beta1, beta2, eps, False
                )
            else:
                xllm_ops.group_timestep_decay_norm_cub_bwd(
                    y_grad, mean_grad, var_grad, x, prev_count, bos_mask, cummean, cumrstd,
                    weight, None, num_groups, beta1, beta2
                )

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'{backend} timestep decay norm bwd: {delta:.2f}s')


def test_scan_cema(
    x, hx, p, q, gamma, y_grad, h_grad, bos_mask, epochs, backend
):
    cema_scan_fwd = xllm_ops.cema_cub_scan_fwd if backend == 'cub' else xllm_ops.cema_blelloch_scan_fwd
    cema_scan_bwd = xllm_ops.cema_cub_scan_bwd if backend == 'cub' else xllm_ops.cema_blelloch_scan_bwd

    x = x.contiguous()
    y_grad = y_grad.contiguous()
    with torch.no_grad():
        # B x D x L
        bsz, _, length = x.size()
        # warm up
        for _ in range(100):
            y, h, chunk_decay, chunk_gain = cema_scan_fwd(x, p, q, gamma, bos_mask, hx)
            x_grad, p_grad, q_grad, gamma_grad, hx_grad = cema_scan_bwd(
                y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask
            )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            y, h, chunk_decay, chunk_gain = cema_scan_fwd(x, p, q, gamma, bos_mask, hx)

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'{backend} scan cema fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            cema_scan_bwd(y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask)

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'{backend} scan cema bwd: {delta:.2f}s')


def test_causal_conv1d(
    x, hx, weight, bias, y_grad, final_state_grad, bos_mask, epochs, backend
):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            causal_conv1d_fwd(
                x, weight, bias, hx, bos_mask, output_final_state=True, activation=None, backend=backend
            )
            causal_conv1d_bwd(
                y_grad, final_state_grad, x, weight, bias, hx, bos_mask, activation=None, backend=backend
            )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            causal_conv1d_fwd(
                x, weight, bias, hx, bos_mask, output_final_state=True, activation=None, backend=backend
            )

        torch.cuda.synchronize()

        delta = timer() - start
        if hx is None:
            print(f'{backend} causal conv1d w.o. hx fwd: {delta:.2f}s')
        else:
            print(f'{backend} causal conv1d w. hx fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            causal_conv1d_bwd(
                y_grad, final_state_grad, x, weight, bias, hx, bos_mask, activation=None, backend=backend
            )

        torch.cuda.synchronize()

        delta = timer() - start
        if hx is None:
            print(f'{backend} causal conv1d w.o. hx bwd: {delta:.2f}s')
        else:
            print(f'{backend} causal conv1d w. hx bwd: {delta:.2f}s')


def test_sliding_chunk_attention(
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
        print(f'SCA-{backend}{suffix} fwd: {delta:.2f}s')

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
            print(f'SCA-{backend}{suffix} bwd: {delta:.2f}s ({delta1:.2f}s)')


def test_adaptive_working_memory(
    query, key, value, out_grad, mem_grad, lnt_grad, chunk_size, segment_idx, ortho, epochs
):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            out, _, mem, lnt = awk_fwd(
                query, key, value, None, None, None, None,
                chunk_size, segment_idx, None, ortho
            )
            awk_bwd(
                out_grad, mem_grad, lnt_grad, query, key, value, None, None,
                None, None, chunk_size, segment_idx, None, ortho
            )
        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            out, _, mem, lnt = awk_fwd(
                query, key, value, None, None, None, None,
                chunk_size, segment_idx, None, ortho,
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'awk (ortho={ortho}) fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            awk_bwd(
                out_grad, mem_grad, lnt_grad, query, key, value, None, None,
                None, None, chunk_size, segment_idx, None, ortho
            )
        torch.cuda.synchronize()

        delta = timer() - start
        print(f'awk (ortho={ortho}) bwd: {delta:.2f}s')


def test(B: int, L: int, D: int, W: int, H: int, HKV: int, S: int, V: int, chunk_size: int, avg_len: int, dtype: str):
    assert L % chunk_size == 0
    if avg_len == 0:
        avg_len = L
    bos_ratio = float(L - avg_len) / (L * avg_len)
    num_groups = D // 64
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        x = torch.randn(B, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")

        prev_count = torch.zeros(B, dtype=torch.int64).to("cuda")
        prev_mean = torch.zeros(B, num_groups, dtype=pt_dtype).to("cuda")
        prev_var = torch.zeros(B, num_groups, dtype=pt_dtype).to("cuda")
        norm_w = torch.randn(D, requires_grad=False, dtype=pt_dtype, device="cuda")
        norm_b = torch.randn(D, requires_grad=False, dtype=pt_dtype, device="cuda")

        # hx = torch.randn(B, D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        # alpha = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        # delta = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        # theta = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        # gamma = torch.randn(D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        # # D x N
        # alpha = torch.sigmoid(alpha)
        # delta = torch.sigmoid(delta)
        # # coeffs
        # p = alpha
        # q = torch.polar(1.0 - alpha * delta, theta)
        # scale = math.sqrt(1.0 / N)
        # gamma = gamma * scale

        conv_w = torch.randn(D, W, requires_grad=False, dtype=pt_dtype, device="cuda")
        init_states = torch.randn(B, W - 1, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        final_state_grad = torch.randn(B, W - 1, D, requires_grad=False, dtype=pt_dtype, device="cuda")

        query = torch.randn(B, L, H, S, requires_grad=False, dtype=pt_dtype, device="cuda")
        key = torch.randn(B, L, HKV, S, requires_grad=False, dtype=pt_dtype, device="cuda")
        value = torch.randn(B, L, HKV, V, requires_grad=False, dtype=pt_dtype, device="cuda")

        bos_mask = torch.rand(B, L, requires_grad=False, device='cuda') < bos_ratio
        segment_idx = torch.cumsum(bos_mask, dim=-1)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        value = F.silu(value) + 0.1

        mean_grad = torch.zeros(B, num_groups, dtype=pt_dtype).to("cuda")
        var_grad = torch.zeros(B, num_groups, dtype=pt_dtype).to("cuda")
        y_grad = torch.randn(B, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        # h_grad = torch.zeros(B, D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        out_grad = torch.randn(B, L, H, V, requires_grad=False, dtype=pt_dtype, device="cuda")
        mem_grad = torch.zeros(B, HKV, S, V, requires_grad=False, dtype=pt_dtype, device="cuda")
        lnt_grad = torch.zeros(B, HKV, S, requires_grad=False, dtype=pt_dtype, device="cuda")

    epochs = 1000
    print(f"B={B}, L={L}, D={D}, W={W}, H={H} ({HKV}), S={S}, V={V}, chunk={chunk_size}, AvgL={avg_len}, dtype={dtype}:")
    # test_decay_timestep_norm(
    #     x, bos_mask, prev_count, prev_mean, prev_var, norm_w, norm_b,
    #     y_grad, mean_grad, var_grad, num_groups, 0.999, 0.9999,
    #     1e-5, epochs, 'chunkwise'
    # )
    test_decay_timestep_norm(
        x, bos_mask, prev_count, prev_mean, prev_var, norm_w, norm_b,
        y_grad, mean_grad, var_grad, num_groups, 0.999, 0.9999,
        1e-5, epochs, 'cub'
    )
    ##################################################
    # test_scan_cema(x.transpose(1, 2), hx, p, q, gamma, y_grad.transpose(1, 2), h_grad, bos_mask, epochs, 'cub')
    # test_scan_cema(x.transpose(1, 2), hx, p, q, gamma, y_grad.transpose(1, 2), h_grad, bos_mask, epochs, 'blelloch')
    ###########################################################
    test_causal_conv1d(x, None, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, 'fla')
    test_causal_conv1d(x, None, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, 'triton')
    # test_causal_conv1d(x, init_states, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, 'fla')
    #####################################################
    # test_sliding_chunk_attention(query, key, value, out_grad, chunk_size, bos_mask, segment_idx, 'swift', epochs)
    if S <= 256 and V <= 256:
        test_sliding_chunk_attention(query, key, value, out_grad, chunk_size, bos_mask, segment_idx, False, 'xattn', epochs)
        test_sliding_chunk_attention(query, key, value, out_grad, chunk_size, bos_mask, segment_idx, True, 'xattn', epochs)
    test_adaptive_working_memory(query, key, value, out_grad, mem_grad, lnt_grad, chunk_size, segment_idx, False, epochs)
    test_adaptive_working_memory(query, key, value, out_grad, mem_grad, lnt_grad, chunk_size, segment_idx, True, epochs)
    print("*" * 70)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, D, H, K, S, V, C in [
        [4, 65536, 1536, 8, 8, 128, 256, 4096],
        # [4, 65536, 1536, 32, 16, 64, 64, 4096],
        # [4, 65536, 1536, 24, 12, 96, 96, 4096],
        [4, 65536, 1536, 16, 8, 128, 128, 4096],
        [4, 65536, 1536, 12, 6, 192, 192, 4096],
        [4, 65536, 1536, 8, 4, 256, 256, 4096],
        [1, 16384, 4096, 16, 4, 256, 2048, 2048],
        [1, 32768, 4096, 16, 4, 256, 2048, 4096],
        [1, 32768, 4096, 16, 16, 128, 256, 4096],
        [1, 65536, 2560, 32, 8, 128, 128, 4096],
    ]:
        for avg_len in [16000]:
            W = 4
            test(B, L, D, W, H, K, S, V, C, avg_len, dtype)


if __name__ == "__main__":
    fire.Fire(main)
