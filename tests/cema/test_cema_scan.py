import fire
import re
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import xllm_extension.ops as xllm_ops
import time
import math
import numpy as np
from contextlib import contextmanager

_c2r = torch.view_as_real


@contextmanager
def measure_memory_and_time(label="", unit="ms"):
    if unit == "ms":
        time_factor = 1000
    else:
        time_factor = 1

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before_mem = torch.cuda.memory_allocated()
    start_time = time.time()

    yield

    torch.cuda.synchronize()
    end_time = time.time()
    after_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()
    elapsed = end_time - start_time

    print(f"[{label}]")
    print(f"Time        :   {elapsed * time_factor:.2f} {unit}")
    print(f"Before      :   {before_mem / 1024 ** 2:.2f} MB")
    print(f"After       :   {after_mem / 1024 ** 2:.2f} MB")
    print(f"Increased   :   {(after_mem - before_mem) / 1024 ** 2:.2f} MB")
    print(f"Peak Mem    :   {peak_mem / 1024 ** 2:.2f} MB")
    print("-" * 40)


def manual_cema(
        x: Tensor,  # B x D x L
        hx: Tensor,  # B x D x N
        p: Tensor,  # D x N
        q: Tensor,  # D x N
        gamma: Tensor,  # D x N
        bos_mask: Tensor,  # B x L
):
    bsz, dim, length = x.size()

    y = []
    h = hx
    for t in range(length):
        if bos_mask is None:
            hx = h
        else:
            curr_mask = bos_mask[:, t]
            hx = h.masked_fill(curr_mask.view(bsz, 1, 1), 0 + 0j)

        # B x D x 1
        xt = x[:, :, t:t + 1]
        # (D x N) x (B x D x 1) -> B x D x N
        h = p * xt + q * hx
        # B x D
        yt = torch.einsum('bdn,dn->bd', h, gamma).real
        # B x D x 1
        y.append(yt.unsqueeze(-1))

    return torch.cat(y, dim=2), h


def scan_test(B: int, D: int, N: int, L: int, dtype: torch.dtype, multi_segment: bool, cub=True):
    print(f"Testing B={B}, D={D}, N={N}, L={L}, dtype={dtype}, multi_segment={multi_segment}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bos_ratio = 0.1
    with torch.no_grad():
        x = torch.randn(B, D, L, requires_grad=False, dtype=dtype, device="cuda")
        hx = torch.randn(B, D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        alpha = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        delta = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        theta = torch.randn(D, N, requires_grad=False, dtype=torch.float32, device="cuda")
        gamma = torch.randn(D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        # D x N x 1
        alpha = torch.sigmoid(alpha)
        delta = torch.sigmoid(delta)
        # coeffs
        p = alpha
        q = torch.polar(1.0 - alpha * delta, theta)
        scale = math.sqrt(1.0 / N)
        gamma = gamma * scale

        q = torch.complex(torch.real(q), torch.zeros_like(torch.real(q), dtype=torch.float32))
        gamma = torch.complex(torch.real(gamma), torch.zeros_like(torch.real(gamma), dtype=torch.float32))
        hx = torch.complex(torch.real(hx), torch.zeros_like(torch.real(hx), dtype=torch.float32))

    x = x.clone().detach().requires_grad_(True)
    hx = hx.clone().detach().requires_grad_(True)
    bos_mask = (torch.rand(B, L) < bos_ratio).to("cuda") if multi_segment else None
    p = p.clone().detach().requires_grad_(True)
    q = q.clone().detach().requires_grad_(True)
    gamma = gamma.clone().detach().requires_grad_(True)

    with measure_memory_and_time("Manual"):
        y_manual, h_manual = manual_cema(x.double(), hx.cdouble(), p.double(), q.cdouble(), gamma.cdouble(), bos_mask)

        y_manual_flat = y_manual.flatten()
        num_elem_y = y_manual_flat.shape[0]
        weight_y = torch.randn(num_elem_y, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(L)
        loss = y_manual_flat @ weight_y
        y_manual.retain_grad()
        loss.backward()
        y_grad = y_manual.grad.to(dtype)
        h_grad = torch.zeros(B, D, N, dtype=torch.complex64, device="cuda")
        x_grad = x.grad
        hx_grad = _c2r(hx.grad)
        p_grad = p.grad
        q_grad = _c2r(q.grad)
        gamma_grad = _c2r(gamma.grad)

    with torch.no_grad():
        L2 = L // 2
        L1 = L - L2
        with measure_memory_and_time("CUDA Kernel"):
            # y_kernel, h_kernel, chunk_decay, chunk_gain = xllm_ops.cema_scan_fwd(x, p, q, gamma, bos_mask if multi_segment else None, hx)
            # x_grad_kernel, p_grad_kernel, q_grad_kernel, gamma_grad_kernel, hx_grad_kernel = xllm_ops.cema_scan_bwd(
            #     y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask)

            # q_grad_kernel = _c2r(q_grad_kernel)
            # gamma_grad_kernel = _c2r(gamma_grad_kernel)
            # hx_grad_kernel = _c2r(hx_grad_kernel)

            if not cub:
                # if True:
                y_kernel_1, h_kernel_1, chunk_decay_1, chunk_gain_1 = xllm_ops.cema_blelloch_scan_fwd(x[:, :, :L1], p, q, gamma,
                                                                                                      bos_mask[:, :L1] if multi_segment else None, hx)
                y_kernel_2, h_kernel_2, chunk_decay_2, chunk_gain_2 = xllm_ops.cema_blelloch_scan_fwd(x[:, :, L1:], p, q, gamma,
                                                                                                      bos_mask[:, L1:] if multi_segment else None,
                                                                                                      h_kernel_1)
            else:
                y_kernel_1, h_kernel_1, chunk_decay_1, chunk_gain_1 = xllm_ops.cema_cub_scan_fwd(x[:, :, :L1], p, q, gamma,
                                                                                                 bos_mask[:, :L1] if multi_segment else None, hx)
                y_kernel_2, h_kernel_2, chunk_decay_2, chunk_gain_2 = xllm_ops.cema_cub_scan_fwd(x[:, :, L1:], p, q, gamma,
                                                                                                 bos_mask[:, L1:] if multi_segment else None,
                                                                                                 h_kernel_1)

            if not cub:
                # if True:
                x_grad_kernel_2, p_grad_kernel_2, q_grad_kernel_2, gamma_grad_kernel_2, hx_grad_kernel_2 = xllm_ops.cema_blelloch_scan_bwd(
                    y_grad[:, :, L1:], h_grad, chunk_decay_2, chunk_gain_2, x[:, :, L1:], p, q, gamma, bos_mask[:, L1:] if multi_segment else None)
                x_grad_kernel_1, p_grad_kernel_1, q_grad_kernel_1, gamma_grad_kernel_1, hx_grad_kernel_1 = xllm_ops.cema_blelloch_scan_bwd(
                    y_grad[:, :, :L1], hx_grad_kernel_2, chunk_decay_1, chunk_gain_1, x[:, :, :L1], p, q, gamma,
                    bos_mask[:, :L1] if multi_segment else None)
            else:
                x_grad_kernel_2, p_grad_kernel_2, q_grad_kernel_2, gamma_grad_kernel_2, hx_grad_kernel_2 = xllm_ops.cema_cub_scan_bwd(
                    y_grad[:, :, L1:], h_grad, chunk_decay_2, chunk_gain_2, x[:, :, L1:], p, q, gamma, bos_mask[:, L1:] if multi_segment else None)
                x_grad_kernel_1, p_grad_kernel_1, q_grad_kernel_1, gamma_grad_kernel_1, hx_grad_kernel_1 = xllm_ops.cema_cub_scan_bwd(
                    y_grad[:, :, :L1], hx_grad_kernel_2, chunk_decay_1, chunk_gain_1, x[:, :, :L1], p, q, gamma,
                    bos_mask[:, :L1] if multi_segment else None)

            h_kernel = h_kernel_2
            y_kernel = torch.cat([y_kernel_1, y_kernel_2], dim=2)
            x_grad_kernel = torch.cat([x_grad_kernel_1, x_grad_kernel_2], dim=2)
            p_grad_kernel = p_grad_kernel_1 + p_grad_kernel_2
            q_grad_kernel = _c2r(q_grad_kernel_1 + q_grad_kernel_2)
            gamma_grad_kernel = _c2r(gamma_grad_kernel_1 + gamma_grad_kernel_2)
            hx_grad_kernel = _c2r(hx_grad_kernel_1)

        atol = 1e-4 if dtype == torch.float32 else 1e-2
        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(h_kernel, h_manual.to(torch.complex64), rtol=rtol, atol=atol)
        torch.testing.assert_close(y_kernel, y_manual.to(dtype), rtol=rtol, atol=atol)

        atol = 1e-4 if dtype == torch.float32 else 1e-2
        rtol = 1e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(x_grad_kernel, x_grad.to(dtype), rtol=rtol, atol=atol)
        torch.testing.assert_close(hx_grad_kernel, hx_grad.to(torch.float32), rtol=rtol, atol=atol)

        atol = 2e-4 if dtype == torch.float32 else 1e-2
        rtol = 2e-4 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(p_grad_kernel, p_grad.to(torch.float32), rtol=rtol, atol=atol)
        torch.testing.assert_close(gamma_grad_kernel, gamma_grad.to(torch.float32), rtol=rtol, atol=atol)

        atol = 2e-3 if dtype == torch.float32 else 1e-2
        rtol = 1e-3 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(q_grad_kernel, q_grad.to(torch.float32), rtol=rtol, atol=atol)

        print("Passed test")
        print("-" * 40)


def main(seed: int):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B in [1, 2, 3, 4, 6, 8, 16, 32]:
        for L in [3, 4, 11, 32, 128, 255, 512, 733, 1024, 4096, 8192, 16384, 32768]:
            for D in [8, 128, 256, 4096, 8192]:
                for N in [4, 8, 16]:
                    for dtype in [torch.float32, torch.bfloat16]:
                        for multi_segment in [False, True]:
                            if B * L > 32768:
                                continue
                            if B * D * N * L > 16384 and dtype == torch.bfloat16:
                                continue
                            scan_test(B, D, N, L, dtype, multi_segment, cub=True)


if __name__ == "__main__":
    fire.Fire(main)
