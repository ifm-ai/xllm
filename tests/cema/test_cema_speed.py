from timeit import default_timer as timer
import math
import numpy as np
import torch
import fire

import xllm_extension.ops as xllm_ops
from xllm.modules.fused_ops import (
    fused_fftconv_fwd,
    fused_fftconv_bwd,
)


def test_mega_cema(
    x, hx, p, q, gamma, y_grad, h_grad, epochs
):
    with torch.no_grad():
        # B x D x L
        bsz, _, length = x.size()
        # D x N x 1
        p = p.unsqueeze(2)
        q = q.unsqueeze(2)
        log_q = q.log()
        k_dtype = None
        # warm up
        for _ in range(100):
            k, b, v1 = xllm_ops.ema_parameters_fwd(p, log_q, gamma, hx, length)
            y, x_f, k_f = fused_fftconv_fwd(x, k)
            h, v2 = xllm_ops.ema_hidden_fwd(x, p, log_q, hx)
            k_dtype = k.dtype

            b_grad = y_grad.to(k_dtype)
            x_grad, k_grad = fused_fftconv_bwd(y_grad, x_f, k_f, k_dtype)

            p_grad, q_grad, gamma_grad, hx_grad = xllm_ops.ema_parameters_bwd(k_grad, b_grad, p, log_q, gamma, hx, v1)
            x_grad_h, p_grad_h, q_grad_h, hx_grax_h = xllm_ops.ema_hidden_bwd(h_grad, x, p, log_q, hx, v2)

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            k, b, v1 = xllm_ops.ema_parameters_fwd(p, log_q, gamma, hx, length)
            y, x_f, k_f = fused_fftconv_fwd(x, k)
            y = y + b.to(y)
            h, v2 = xllm_ops.ema_hidden_fwd(x, p, log_q, hx)

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'mega cema fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            b_grad = y_grad.to(k_dtype)
            x_grad, k_grad = fused_fftconv_bwd(y_grad, x_f, k_f, k_dtype)

            p_grad, q_grad, gamma_grad, hx_grad = xllm_ops.ema_parameters_bwd(k_grad, b_grad, p, log_q, gamma, hx, v1)
            x_grad_h, p_grad_h, q_grad_h, hx_grax_h = xllm_ops.ema_hidden_bwd(h_grad, x, p, log_q, hx, v2)
            x_grad = x_grad + x_grad_h
            p_grad = p_grad + p_grad_h
            q_grad = q_grad + q_grad_h
            hx_grad = hx_grad + hx_grax_h

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'mega cema bwd: {delta:.2f}s')


def test_scan_cema(
    x, hx, p, q, gamma, y_grad, h_grad, bos_mask, epochs, backend
):
    cema_scan_fwd = xllm_ops.cema_cub_scan_fwd if backend == 'cub' else xllm_ops.cema_blelloch_scan_fwd
    cema_scan_bwd = xllm_ops.cema_cub_scan_bwd if backend == 'cub' else xllm_ops.cema_blelloch_scan_bwd
    with torch.no_grad():
        # B x D x L
        bsz, _, length = x.size()
        # warm up
        for _ in range(100):
            y, h, chunk_decay, chunk_gain = cema_scan_fwd(x, p, q, gamma, bos_mask, hx)
            if backend == 'cub':
                y, h = xllm_ops.cema_cub_scan_fwd_recalc(x, p, q, gamma, bos_mask, chunk_decay, chunk_gain)

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

        # if backend =='cub':
        #     start = timer()
        #     for _ in range(epochs):
        #         y, h = xllm_ops.cema_cub_scan_fwd_recalc(x, p, q, gamma, bos_mask, chunk_decay, chunk_gain)
        #
        #     torch.cuda.synchronize()
        #
        #     delta = timer() - start
        #     print(f'{backend} scan cema recalc: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            x_grad, p_grad, q_grad, gamma_grad, hx_grad = cema_scan_bwd(
                y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask
            )

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'{backend} scan cema bwd: {delta:.2f}s')


def test(B: int, L: int, D: int, N: int, dtype: str):
    bos_ratio = 0.1
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        x = torch.randn(B, D, L, requires_grad=False, dtype=pt_dtype, device="cuda")
        hx = torch.randn(B, D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        alpha = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        delta = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        theta = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        gamma = torch.randn(D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        # D x N
        alpha = torch.sigmoid(alpha)
        delta = torch.sigmoid(delta)
        # coeffs
        p = alpha
        q = torch.polar(1.0 - alpha * delta, theta)
        scale = math.sqrt(1.0 / N)
        gamma = gamma * scale

    x = x.detach()
    hx = hx.detach()
    bos_mask = (torch.rand(B, L) < bos_ratio).to("cuda")
    p = p.detach()
    q = q.detach()
    gamma = gamma.detach()

    y_grad = torch.randn(B, D, L, requires_grad=False, dtype=pt_dtype, device="cuda")
    h_grad = torch.zeros(B, D, N, requires_grad=False, dtype=torch.complex64, device="cuda")

    epochs = 1000
    print(f"B={B}, L={L}, D={D}, N={N}, dtype={dtype}:")

    test_mega_cema(x, hx, p, q, gamma, y_grad, h_grad, epochs)
    test_scan_cema(x, hx, p, q, gamma, y_grad, h_grad, bos_mask, epochs, 'blelloch')
    test_scan_cema(x, hx, p, q, gamma, y_grad, h_grad, bos_mask, epochs, 'cub')
    print("*" * 70)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for L in [8192, 16384, 32768, 65536]:
        for D in [1024, 2048, 4096]:
            for B, N in [
                [1, 4],
                [1, 8],
                [2, 8],
                [4, 8],
                [8, 8],
                [1, 16],
                [2, 16],
                [4, 16],
                [8, 16],
            ]:
                test(B, L, D, N, dtype)


if __name__ == "__main__":
    fire.Fire(main)
