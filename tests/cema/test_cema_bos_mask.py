import math
import numpy as np
import torch
from torch import Tensor
import fire

import xllm_extension.ops as xllm_ops
from xllm.modules.fused_ops import (
    fused_fftconv_fwd,
    fused_fftconv_bwd,
)

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


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
        xt = x[:, :, t:t+1]
        # (D x N) x (B x D x 1) -> B x D x N
        h = p * xt + q * hx
        # B x D
        yt = torch.einsum('bdn,dn->bd', h, gamma).real
        # B x D x 1
        y.append(yt.unsqueeze(-1))

    return torch.cat(y, dim=2), h


def mega_cema_fwd(
    x: Tensor,  # B x D x L
    hx: Tensor,  # B x D x N
    p: Tensor,  # D x N
    q: Tensor,  # D x N
    gamma: Tensor,  # D x N
):
    # B x D x L
    bsz, _, length = x.size()
    # D x N x 1
    p = p.unsqueeze(2)
    q = q.unsqueeze(2)
    log_q = q.log()
    k, b, v1 = xllm_ops.ema_parameters_fwd(p, log_q, gamma, hx, length)
    y, x_f, k_f = fused_fftconv_fwd(x, k)
    y = y + b.to(y)
    h, v2 = xllm_ops.ema_hidden_fwd(x, p, log_q, hx)

    return y, v1, x_f, k_f, h, v2, k.dtype


def mega_cema_bwd(
    y_grad: Tensor,
    h_grad: Tensor,
    x: Tensor,
    hx: Tensor,
    v1: Tensor,
    x_f: Tensor,
    k_f: Tensor,
    v2: Tensor,
    p: Tensor,  # D x N
    q: Tensor,  # D x N
    gamma: Tensor,  # D x N
    k_dtype: torch.dtype
):
    # D x N x 1
    p = p.unsqueeze(2)
    q = q.unsqueeze(2)
    log_q = q.log()

    b_grad = y_grad.to(k_dtype)
    x_grad, k_grad = fused_fftconv_bwd(y_grad, x_f, k_f, k_dtype)

    p_grad, q_grad, gamma_grad, hx_grad = xllm_ops.ema_parameters_bwd(k_grad, b_grad, p, log_q, gamma, hx, v1)

    x_grad_h, p_grad_h, q_grad_h, hx_grax_h = xllm_ops.ema_hidden_bwd(h_grad, x, p, log_q, hx, v2)
    x_grad = x_grad + x_grad_h
    p_grad = p_grad + p_grad_h
    q_grad = q_grad + q_grad_h
    assert hx_grad is not None and hx_grax_h is not None
    hx_grad = hx_grad + hx_grax_h

    p_grad = p_grad.squeeze(2)
    q_grad = q_grad.squeeze(2)

    return x_grad, hx_grad, p_grad, q_grad, gamma_grad


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


def test(B: int, L: int, D: int, N: int, multi_segment: bool, dtype: str):
    bos_ratio = 0.2
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        x = torch.randn(B, D, L, requires_grad=False, dtype=pt_dtype, device="cuda")
        hx = torch.randn(B, D, N, requires_grad=False, dtype=torch.complex64, device="cuda")
        alpha = torch.randn(D, N, requires_grad=False, dtype=pt_dtype, device="cuda")
        delta = torch.randn(D, N, requires_grad=False, dtype=pt_dtype, device="cuda")
        theta = torch.randn(D, N, requires_grad=False, dtype=pt_dtype, device="cuda")
        gamma = torch.randn(D, N, 2, requires_grad=False, dtype=pt_dtype, device="cuda")
        # D x N
        alpha = torch.sigmoid(alpha.float())
        delta = torch.sigmoid(delta.float())
        # coeffs
        p = alpha
        q = torch.polar(1.0 - alpha * delta, theta.float())
        scale = math.sqrt(1.0 / N)
        gamma = _r2c(gamma.float()) * scale

    x = x.clone().detach().requires_grad_(True)
    hx = hx.clone().detach().requires_grad_(True)
    bos_mask = (torch.rand(B, L) < bos_ratio).to("cuda") if multi_segment else None
    p = p.clone().detach().requires_grad_(True)
    q = q.clone().detach().requires_grad_(True)
    gamma = gamma.clone().detach().requires_grad_(True)

    y_manual, h_manual = manual_cema(x.double(), hx.cdouble(), p.double(), q.cdouble(), gamma.cdouble(), bos_mask)

    y_manual_flat = y_manual.flatten()
    num_elem_y = y_manual_flat.shape[0]
    weight_y = torch.randn(num_elem_y, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(B * L)
    loss = y_manual_flat @ weight_y
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(pt_dtype)
    h_grad = torch.zeros(B, D, N, dtype=torch.complex64, device="cuda")
    x_grad = x.grad
    hx_grad = _c2r(hx.grad)
    p_grad = p.grad
    q_grad = _c2r(q.grad)
    gamma_grad = _c2r(gamma.grad)

    with torch.no_grad():
        L2 = L // 2
        L1 = L - L2

        if not multi_segment:
            # forward
            y1_mega, v1_1, x_f_1, k_f_1, h1_mega, v2_1, k_dtype = mega_cema_fwd(
                x[:, :, :L1], hx, p, q, gamma,
            )

            y2_mega, v1_2, x_f_2, k_f_2, h2_mega, v2_2, k_dtype = mega_cema_fwd(
                x[:, :, L1:], h1_mega, p, q, gamma,
            )

            atol = {"fp32": 2e-4, "bf16": 2.5e-2, "fp16": 3e-3}[dtype]
            rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(torch.cat([y1_mega, y2_mega], dim=2), y_manual.to(pt_dtype), rtol=rtol, atol=atol)
            torch.testing.assert_close(h2_mega, h_manual.to(torch.complex64), rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, D={D}, N={N}, multiseg={multi_segment}, dtype={dtype}: pass mega fwd test")

            # backward
            x2_mega_grad, h1_mega_grad, p2_mega_grad, q2_mega_grad, gamma2_mega_grad = mega_cema_bwd(
                y_grad[:, :, L1:], h_grad, x[:, :, L1:], h1_mega, v1_2, x_f_2, k_f_2, v2_2, p, q, gamma, k_dtype
            )

            x1_mega_grad, hx_mega_grad, p1_mega_grad, q1_mega_grad, gamma1_mega_grad = mega_cema_bwd(
                y_grad[:, :, :L1], h1_mega_grad, x[:, :, :L1], hx, v1_1, x_f_1, k_f_1, v2_1, p, q, gamma, k_dtype
            )

            x_mega_grad = torch.cat([x1_mega_grad, x2_mega_grad], dim=2)
            hx_mega_grad = _c2r(hx_mega_grad)
            p_mega_grad = p1_mega_grad + p2_mega_grad
            gamma_mega_grad = _c2r(gamma1_mega_grad + gamma2_mega_grad)
            q_mega_grad = _c2r(q1_mega_grad + q2_mega_grad)

            atol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 6e-4}[dtype]
            rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(x_mega_grad, x_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(hx_mega_grad, hx_grad, rtol=rtol, atol=atol)

            atol = {"fp32": 5e-4, "bf16": 4e-2, "fp16": 4e-3}[dtype]
            rtol = {"fp32": 2e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(p_mega_grad, p_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(gamma_mega_grad, gamma_grad, rtol=rtol, atol=atol)

            atol = {"fp32": 3e-3, "bf16": 6e-1, "fp16": 1e-1}[dtype]
            rtol = {"fp32": 1e-3, "bf16": 1e-1, "fp16": 2e-2}[dtype]
            torch.testing.assert_close(q_mega_grad, q_grad, rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, D={D}, N={N}, multiseg={multi_segment}, dtype={dtype}: pass mega bwd test")

        for backend in ['cub', 'blelloch']:
            # scan forward
            y1_scan, h1_scan, chunk_decay_1, chunk_gain_1 = scan_cema_fwd(
                x[:, :, :L1], hx, p, q, gamma, bos_mask[:, :L1] if multi_segment else None, backend
            )

            y2_scan, h2_scan, chunk_decay_2, chunk_gain_2 = scan_cema_fwd(
                x[:, :, L1:], h1_scan, p, q, gamma, bos_mask[:, L1:] if multi_segment else None, backend
            )

            atol = {"fp32": 1e-4, "bf16": 5e-3, "fp16": 5e-4}[dtype]
            rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(torch.cat([y1_scan, y2_scan], dim=2), y_manual.to(pt_dtype), rtol=rtol, atol=atol)
            torch.testing.assert_close(h2_scan, h_manual.to(torch.complex64), rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, D={D}, N={N}, multiseg={multi_segment}, dtype={dtype}: pass {backend} scan fwd test")

            # scan re-compute
            if backend == 'cub':
                y1_recalc, h1_recalc = scan_cema_recalc(
                    x[:, :, :L1], chunk_decay_1, chunk_gain_1, p, q, gamma, bos_mask[:, :L1] if multi_segment else None, backend
                )

                y2_recalc, h2_recalc = scan_cema_recalc(
                    x[:, :, L1:], chunk_decay_2, chunk_gain_2, p, q, gamma, bos_mask[:, L1:] if multi_segment else None, backend
                )

                atol = 1e-8
                rtol = 1e-8
                torch.testing.assert_close(torch.cat([y1_recalc, y2_recalc], dim=2), torch.cat([y1_scan, y2_scan], dim=2), rtol=rtol, atol=atol)
                torch.testing.assert_close(h1_scan, h1_recalc, rtol=rtol, atol=atol)
                torch.testing.assert_close(h2_scan, h2_recalc, rtol=rtol, atol=atol)
                print(f"B={B}, L={L}, D={D}, N={N}, multiseg={multi_segment}, dtype={dtype}: pass {backend} scan recalc test")

            # scan backward
            x2_scan_grad, h1_scan_grad, p2_scan_grad, q2_scan_grad, gamma2_scan_grad = scan_cema_bwd(
                y_grad[:, :, L1:], h_grad, x[:, :, L1:], chunk_decay_2, chunk_gain_2, p, q, gamma,
                bos_mask[:, L1:] if multi_segment else None, backend
            )

            x1_scan_grad, hx_scan_grad, p1_scan_grad, q1_scan_grad, gamma1_scan_grad = scan_cema_bwd(
                y_grad[:, :, :L1], h1_scan_grad, x[:, :, :L1], chunk_decay_1, chunk_gain_1, p, q, gamma,
                bos_mask[:, :L1] if multi_segment else None, backend
            )

            x_scan_grad = torch.cat([x1_scan_grad, x2_scan_grad], dim=2)
            hx_scan_grad = _c2r(hx_scan_grad)
            p_scan_grad = p1_scan_grad + p2_scan_grad
            gamma_scan_grad = _c2r(gamma1_scan_grad + gamma2_scan_grad)
            q_scan_grad = _c2r(q1_scan_grad + q2_scan_grad)

            atol = {"fp32": 1e-4, "bf16": 6e-3, "fp16": 6e-4}[dtype]
            rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(x_scan_grad, x_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(hx_scan_grad, hx_grad, rtol=rtol, atol=atol)

            atol = {"fp32": 1e-4, "bf16": 4e-2, "fp16": 4e-3}[dtype]
            rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(p_scan_grad, p_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(gamma_scan_grad, gamma_grad, rtol=rtol, atol=atol)

            atol = {"fp32": 2e-4, "bf16": 2e-1, "fp16": 4e-2}[dtype]
            rtol = {"fp32": 2e-4, "bf16": 1e-1, "fp16": 2e-2}[dtype]
            torch.testing.assert_close(q_scan_grad, q_grad, rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, D={D}, N={N}, multiseg={multi_segment}, dtype={dtype}: pass {backend} scan bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B in [1, 2, 3, 4, 5, 8]:
        for L in [3, 4, 11, 32, 128, 255, 512, 733, 1024, 4096, 8192, 16384, 32768, 65536]:
            for D in [8, 64, 123, 256, 1024]:
                for N in [4, 8, 16]:
                    if B * L * D * N > 65536 * 1024 * 16:
                        continue
                    for multi_seg in [False, True]:
                        test(B, L, D, N, multi_seg, dtype)


if __name__ == "__main__":
    fire.Fire(main)
