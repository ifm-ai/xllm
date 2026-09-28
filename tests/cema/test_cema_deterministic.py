import math
import numpy as np
import torch
from torch import Tensor
import fire

import xllm_extension.ops as xllm_ops

_c2r = torch.view_as_real


def scan_cema_fwd(
    x: Tensor,  # B x D x L
    hx: Tensor,  # B x D x N
    p: Tensor,  # D x N
    q: Tensor,  # D x N
    gamma: Tensor,  # D x N
    bos_mask: Tensor,  # B x L
    backend: str
):
    if backend == 'cub':
        y, h, chunk_decay, chunk_gain = xllm_ops.cema_cub_scan_fwd(x, p, q, gamma, bos_mask, hx)
    else:
        y, h, chunk_decay, chunk_gain = xllm_ops.cema_blelloch_scan_fwd(x, p, q, gamma, bos_mask, hx)
    return y, h, chunk_decay, chunk_gain


def scan_cema_recalc(
    x: Tensor,  # B x D x L
    chunk_decay: Tensor,  # B x D x L/C x N
    chunk_gain: Tensor,  # B x D x L/C x N
    p: Tensor,  # D x N
    q: Tensor,  # D x N
    gamma: Tensor,  # D x N
    bos_mask: Tensor,  # B x L
    backend: str
):
    assert backend == 'cub'
    y, h = xllm_ops.cema_cub_scan_fwd_recalc(x, p, q, gamma, bos_mask, chunk_decay, chunk_gain)
    return y, h


def scan_cema_bwd(
    y_grad: Tensor,
    h_grad: Tensor,
    x: Tensor,  # B x D x L
    chunk_decay: Tensor,
    chunk_gain: Tensor,
    p: Tensor,  # D x N
    q: Tensor,  # D x N
    gamma: Tensor,  # D x N
    bos_mask: Tensor,  # B x L
    backend: str
):
    if backend == 'cub':
        x_grad, p_grad, q_grad, gamma_grad, hx_grad = xllm_ops.cema_cub_scan_bwd(
            y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask
        )
    else:
        x_grad, p_grad, q_grad, gamma_grad, hx_grad = xllm_ops.cema_blelloch_scan_bwd(
            y_grad, h_grad, chunk_decay, chunk_gain, x, p, q, gamma, bos_mask
        )
    return x_grad, hx_grad, p_grad, q_grad, gamma_grad


def test(B: int, L: int, D: int, N: int, dtype: str):
    with torch.no_grad():
        bos_ratio = 0.1
        pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

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

        bos_mask = (torch.rand(B, L) < bos_ratio).to("cuda")

        y_grad = torch.randn(B, D, L, requires_grad=False, dtype=pt_dtype, device="cuda")
        h_grad = torch.randn(B, D, N, dtype=torch.complex64, device="cuda")

        for backend in ['blelloch', 'cub']:
            # forward-1
            y1_scan, h1_scan, chunk1_decay, chunk1_gain = scan_cema_fwd(x, hx, p, q, gamma, bos_mask, backend)
            h1_scan = _c2r(h1_scan)

            # forward-2
            y2_scan, h2_scan, chunk2_decay, chunk2_gain = scan_cema_fwd(x, hx, p, q, gamma, bos_mask, backend)
            h2_scan = _c2r(h2_scan)

            print(f"y err in {dtype} {backend}")
            print(torch.abs(y1_scan - y2_scan).max())
            print(f"h err in {dtype} {backend}")
            print(torch.abs(h1_scan - h2_scan).max())

            # re-compute
            if backend == 'cub':
                y_recalc, h_recalc = scan_cema_recalc(
                    x, chunk1_decay, chunk1_gain, p, q, gamma, bos_mask, backend
                )
                h_recalc = _c2r(h_recalc)
                print(f"y_recalc err in {dtype} {backend}")
                print(torch.abs(y1_scan - y_recalc).max())
                print(f"h_recalc err in {dtype} {backend}")
                print(torch.abs(h1_scan - h_recalc).max())

            # backward-1
            x1_scan_grad, hx1_scan_grad, p1_scan_grad, q1_scan_grad, gamma1_scan_grad = scan_cema_bwd(
                y_grad, h_grad, x, chunk1_decay, chunk1_gain, p, q, gamma, bos_mask, backend
            )
            hx1_scan_grad = _c2r(hx1_scan_grad)
            gamma1_scan_grad = _c2r(gamma1_scan_grad)
            q1_scan_grad = _c2r(q1_scan_grad)


            # backward-2
            x2_scan_grad, hx2_scan_grad, p2_scan_grad, q2_scan_grad, gamma2_scan_grad = scan_cema_bwd(
                y_grad, h_grad, x, chunk2_decay, chunk2_gain, p, q, gamma, bos_mask, backend
            )

            hx2_scan_grad = _c2r(hx2_scan_grad)
            gamma2_scan_grad = _c2r(gamma2_scan_grad)
            q2_scan_grad = _c2r(q2_scan_grad)

            print(f"x_grad err in {dtype} {backend}")
            print(torch.abs(x1_scan_grad - x2_scan_grad).max())
            print(f"hx_grad err in {dtype} {backend}")
            print(torch.abs(hx1_scan_grad - hx2_scan_grad).max())
            print(f"p_grad err in {dtype} {backend}")
            print(torch.abs(p1_scan_grad - p2_scan_grad).max())
            print(f"q_grad err in {dtype} {backend}")
            print(torch.abs(q1_scan_grad - q2_scan_grad).max())
            print(f"gamma_grad err {dtype} {backend}")
            print(torch.abs(gamma1_scan_grad - gamma2_scan_grad).max())
            print('=' * 50)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, D, N in [
        [1, 16384, 4096, 16],
    ]:
        test(B, L, D, N, dtype)


if __name__ == "__main__":
    fire.Fire(main)
