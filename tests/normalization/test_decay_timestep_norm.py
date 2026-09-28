import math
import numpy as np
import torch
from torch import Tensor
import fire

import xllm_extension.ops as xllm_ops


def manual_normalize(
    x: Tensor,
    bos_mask: Tensor,
    prev_count: Tensor,
    prev_mean: Tensor,
    prev_var: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: int,
    beta1: float,
    beta2: float,
    eps: float,
):
    bsz, length, dim = x.size()
    assert dim % num_groups == 0
    count = prev_count  # B
    mean = prev_mean.unsqueeze(2)  # B x K x 1
    var = prev_var.unsqueeze(2)  # B x K x 1
    cummean = []
    cumrstd = []
    y = []
    for t in range(length):
        curr_mask = bos_mask[:, t]
        count = count.masked_fill(curr_mask, 0) + 1  # count = 1 if bos else prev_count + 1
        mean = mean.masked_fill(curr_mask.view(bsz, 1, 1), 0.0)  # B x K x 1
        var = var.masked_fill(curr_mask.view(bsz, 1, 1), 0.0)  # B x K x 1

        # B x D -> B x K x G
        xt = x[:, t].view(bsz, num_groups, -1)
        # B x K x 1
        vt, mt = torch.var_mean(xt, dim=-1, correction=0, keepdim=True)
        mean = beta1 * mean + (1.0 - beta1) * mt
        var = beta2 * var + (1.0 - beta2) * vt

        # B x K x 1
        correction1 = (1.0 - beta1 ** count).view(bsz, 1, 1)  # B x 1 x 1
        correction2 = (1.0 - beta2 ** count).view(bsz, 1, 1)  # B x 1 x 1
        curr_mean = mean / correction1
        curr_std = torch.sqrt(var / correction2 + eps)

        # B x K x G
        yt = (xt - curr_mean) / curr_std
        # B x D
        yt = yt.view(bsz, 1, dim) * weight + bias
        cummean.append(curr_mean.reshape(bsz, 1, -1))
        cumrstd.append((1.0 / curr_std).reshape(bsz, 1, -1))
        y.append(yt)

    return torch.cat(y, dim=1), torch.cat(cummean, dim=1), torch.cat(cumrstd, dim=1), mean.squeeze(2), var.squeeze(2)


def mega_normalize_fwd(
    x: Tensor,
    bos_mask: torch.Tensor,
    prev_count: torch.Tensor,
    prev_mean: torch.Tensor,
    prev_var: torch.Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: int,
    beta1: float,
    beta2: float,
    eps: float,
    backend: str,
):
    """"""
    if backend == 'chunkwise':
        y, count, mean, var, cummean, cumrstd = xllm_ops.group_timestep_decay_norm_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, weight, bias, None, num_groups, beta1, beta2, eps,
        )
    elif backend == 'cub':
        y, count, mean, var, cummean, cumrstd = xllm_ops.group_timestep_decay_norm_cub_fwd(
            x, bos_mask, prev_count, prev_mean, prev_var, weight, bias, None, num_groups, beta1, beta2, eps,
        )
    else:
        raise ValueError(f"Unknown backend {backend}")
    return y, count, mean, var, cummean, cumrstd


def mega_normalize_bwd(
    y_grad: Tensor,
    mean_grad: Tensor,
    var_grad: Tensor,
    x_or_y: Tensor,
    bos_mask: torch.Tensor,
    prev_count: torch.Tensor,
    cummean: Tensor,
    cumrstd: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_groups: int,
    beta1: float,
    beta2: float,
    eps: float,
    memory_efficient: bool,
    backend: str,
):
    if backend == 'chunkwise':
        x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad =  xllm_ops.group_timestep_decay_norm_bwd(
            y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd, weight, bias,
            None, num_groups, beta1, beta2, eps, memory_efficient,
        )
    elif backend == 'cub':
        x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad = xllm_ops.group_timestep_decay_norm_cub_bwd(
            y_grad, mean_grad, var_grad, x_or_y, prev_count, bos_mask, cummean, cumrstd, weight, 
            None, num_groups, beta1, beta2,
        )
    else:
        raise ValueError(f"Unknown backend {backend}")

    return x_grad, prev_mean_grad, prev_var_grad, gamma_grad, beta_grad


def test(B: int, L: int, H: int, num_groups: int, dtype: torch.dtype):
    eps = 1e-5
    beta1 = 0.999
    beta2 = 0.9999
    bos_ratio = 0.1

    with torch.no_grad():
        x = torch.randn(B, L, H, requires_grad=False, dtype=dtype, device="cuda")
        x = x + 0.1
        gamma = torch.randn(H, requires_grad=False, dtype=dtype, device="cuda")
        gamma = gamma.clamp(min=-0.9) + 1.0

    x = x.clone().detach().requires_grad_(True)
    gamma = gamma.clone().detach().requires_grad_(True)
    beta = torch.randn(H, requires_grad=True, dtype=dtype, device="cuda")

    bos_mask = (torch.rand(B, L) < bos_ratio).to("cuda")
    prev_count = torch.zeros(B, dtype=torch.int64).to("cuda")
    prev_mean = torch.zeros(B, num_groups, dtype=dtype).to("cuda")
    prev_var = torch.zeros(B, num_groups, dtype=dtype).to("cuda")

    y_manual, cummean, cumrstd, mean, var = manual_normalize(
        x.double(), bos_mask, prev_count, prev_mean.double(), prev_var.double(), gamma.double(), beta.double(),
        num_groups=num_groups, beta1=beta1, beta2=beta2, eps=eps,
    )

    y_manual_flat = y_manual.flatten()
    num_elem = y_manual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(L)
    loss = y_manual_flat @ weight
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(dtype)
    x_grad = x.grad
    gamma_grad = gamma.grad
    beta_grad = beta.grad

    with torch.no_grad():
        for backend in ['chunkwise', 'cub']:
            mean_grad = torch.zeros(B, num_groups, dtype=dtype).to("cuda")
            var_grad = torch.zeros(B, num_groups, dtype=dtype).to("cuda")

            # single segment fwd
            y_mega, count_mega, mean_mega, var_mega, cummean_mega, cumrstd_mega = mega_normalize_fwd(
                x, bos_mask, prev_count, prev_mean, prev_var, gamma, beta,
                num_groups=num_groups, beta1=beta1, beta2=beta2, eps=eps, backend=backend
            )

            atol = 1e-3
            rtol = 1e-2
            torch.testing.assert_close(y_mega, y_manual.to(dtype), rtol=rtol, atol=atol)
            torch.testing.assert_close(mean_mega, mean.to(dtype), rtol=rtol, atol=atol)
            torch.testing.assert_close(var_mega, var.to(dtype), rtol=rtol, atol=atol)
            if backend == 'chunkwise':
                torch.testing.assert_close(cummean_mega, cummean.to(dtype), rtol=rtol, atol=atol)
                torch.testing.assert_close(cumrstd_mega, cumrstd.to(dtype), rtol=rtol, atol=atol)
            else:
                torch.testing.assert_close(cummean_mega.transpose(1, 2), cummean.to(dtype), rtol=rtol, atol=atol)
                torch.testing.assert_close(cumrstd_mega.transpose(1, 2), cumrstd.to(dtype), rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, H={H}, num_groups={num_groups}, backend={backend}, dtype={dtype}: pass fwd test")

            # single segment bwd
            x_mega_grad, prev_mean_grad_mega, prev_var_grad_mega, gamma_mega_grad, beta_mega_grad = mega_normalize_bwd(
                y_grad, mean_grad, var_grad, x, bos_mask, prev_count, cummean_mega, cumrstd_mega, gamma, beta,
                num_groups, beta1, beta2, eps, False, backend
            )

            atol = 1e-3
            rtol = 1e-2
            torch.testing.assert_close(x_mega_grad, x_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(gamma_mega_grad, gamma_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(beta_mega_grad, beta_grad, rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, H={H}, num_groups={num_groups}, backend={backend}, dtype={dtype}: pass bwd test (mem_effn=False)")

            if backend == 'chunkwise':
                # single segment bwd
                x_mega_grad, prev_mean_grad_mega, prev_var_grad_mega, gamma_mega_grad, beta_mega_grad = mega_normalize_bwd(
                    y_grad, mean_grad, var_grad, y_mega, bos_mask, prev_count, cummean_mega, cumrstd_mega, gamma, beta,
                    num_groups, beta1, beta2, eps, True, backend
                )

                atol = 1e-3
                rtol = 1e-2
                torch.testing.assert_close(x_mega_grad, x_grad, rtol=rtol, atol=atol)
                torch.testing.assert_close(gamma_mega_grad, gamma_grad, rtol=rtol, atol=atol)
                torch.testing.assert_close(beta_mega_grad, beta_grad, rtol=rtol, atol=atol)
                print(f"B={B}, L={L}, H={H}, num_groups={num_groups}, backend={backend}, dtype={dtype}: pass bwd test (mem_effn=True)")

            #############################################################################################################
            # two-segment fwd
            L2 = L // 2
            L1 = L - L2
            y1_mega, count1_mega, mean1_mega, var1_mega, cummean1_mega, cumrstd1_mega = mega_normalize_fwd(
                x[:, :L1], bos_mask[:, :L1], prev_count, prev_mean, prev_var, gamma, beta,
                num_groups=num_groups, beta1=beta1, beta2=beta2, eps=eps, backend=backend,
            )
            y2_mega, count2_mega, mean2_mega, var2_mega, cummean2_mega, cumrstd2_mega = mega_normalize_fwd(
                x[:, L1:], bos_mask[:, L1:], count1_mega, mean1_mega, var1_mega, gamma, beta,
                num_groups=num_groups, beta1=beta1, beta2=beta2, eps=eps, backend=backend,
            )

            atol = 1e-3
            rtol = 1e-2
            cat_dim = 1 if backend == 'chunkwise' else 2
            torch.testing.assert_close(y_mega, torch.cat([y1_mega, y2_mega], dim=1), rtol=rtol, atol=atol)
            torch.testing.assert_close(cummean_mega, torch.cat([cummean1_mega, cummean2_mega], dim=cat_dim), rtol=rtol, atol=atol)
            torch.testing.assert_close(cumrstd_mega, torch.cat([cumrstd1_mega, cumrstd2_mega], dim=cat_dim), rtol=rtol, atol=atol)
            torch.testing.assert_close(count_mega, count2_mega, rtol=rtol, atol=atol)
            torch.testing.assert_close(mean_mega, mean2_mega, rtol=rtol, atol=atol)
            torch.testing.assert_close(var_mega, var2_mega, rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, H={H}, num_groups={num_groups}, backend={backend}, dtype={dtype}: pass 2-seg fwd test")

            # two-segment bwd
            x2_mega_grad, prev_mean2_grad_mega, prev_var2_grad_mega, gamma2_mega_grad, beta2_mega_grad = mega_normalize_bwd(
                y_grad[:, L1:], mean_grad, var_grad, x[:, L1:], bos_mask[:, L1:], count1_mega, cummean2_mega, cumrstd2_mega,
                gamma, beta, num_groups, beta1, beta2, eps, False, backend,
            )

            x1_mega_grad, prev_mean1_grad_mega, prev_var1_grad_mega, gamma1_mega_grad, beta1_mega_grad = mega_normalize_bwd(
                y_grad[:, :L1], prev_mean2_grad_mega, prev_var2_grad_mega, x[:, :L1], bos_mask[:, :L1], prev_count, cummean1_mega, cumrstd1_mega,
                gamma, beta, num_groups, beta1, beta2, eps, False, backend,
            )

            atol = 1e-4
            rtol = 1e-2
            torch.testing.assert_close(x_mega_grad, torch.cat([x1_mega_grad, x2_mega_grad], dim=1), rtol=rtol, atol=atol)
            torch.testing.assert_close(prev_mean_grad_mega, prev_mean1_grad_mega, rtol=rtol, atol=atol)
            torch.testing.assert_close(prev_var_grad_mega, prev_var1_grad_mega, rtol=rtol, atol=atol)
            torch.testing.assert_close(gamma_mega_grad, gamma1_mega_grad + gamma2_mega_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(beta_mega_grad, beta1_mega_grad + beta2_mega_grad, rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, H={H}, num_groups={num_groups}, backend={backend}, dtype={dtype}: pass 2-seg bwd test (mem_effn=False)")

            if backend == 'chunkwise':
                # two-segment bwd
                x2_mega_grad, prev_mean2_grad_mega, prev_var2_grad_mega, gamma2_mega_grad, beta2_mega_grad = mega_normalize_bwd(
                    y_grad[:, L1:], mean_grad, var_grad, y2_mega, bos_mask[:, L1:], count1_mega, cummean2_mega, cumrstd2_mega,
                    gamma, beta, num_groups, beta1, beta2, eps, True, backend,
                )

                x1_mega_grad, prev_mean1_grad_mega, prev_var1_grad_mega, gamma1_mega_grad, beta1_mega_grad = mega_normalize_bwd(
                    y_grad[:, :L1], prev_mean2_grad_mega, prev_var2_grad_mega, y1_mega, bos_mask[:, :L1], prev_count, cummean1_mega, cumrstd1_mega,
                    gamma, beta, num_groups, beta1, beta2, eps, True, backend
                )

                atol = 1e-4
                rtol = 1e-2
                torch.testing.assert_close(x_mega_grad, torch.cat([x1_mega_grad, x2_mega_grad], dim=1), rtol=rtol, atol=atol)
                torch.testing.assert_close(prev_mean_grad_mega, prev_mean1_grad_mega, rtol=rtol, atol=atol)
                torch.testing.assert_close(prev_var_grad_mega, prev_var1_grad_mega, rtol=rtol, atol=atol)
                torch.testing.assert_close(gamma_mega_grad, gamma1_mega_grad + gamma2_mega_grad, rtol=rtol, atol=atol)
                torch.testing.assert_close(beta_mega_grad, beta1_mega_grad + beta2_mega_grad, rtol=rtol, atol=atol)
                print(f"B={B}, L={L}, H={H}, num_groups={num_groups}, backend={backend}, dtype={dtype}: pass 2-seg bwd test (mem_effn=True)")


def main(seed: int):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, H, num_groups in [
        [2, 3, 6, 2],
        [16, 32, 128, 4],
        [16, 128, 256, 32],
        [16, 255, 256, 64],
        [16, 256, 256, 64],
        [4, 512, 256, 64],
        [1, 733, 256, 64],
        [1, 1024, 256, 64],
        [1, 32768, 4096, 128],
        [2, 32768, 4096, 64],
        [1, 65536, 4096, 128]
    ]:
        test(B, L, H, num_groups, torch.float32)


if __name__ == "__main__":
    fire.Fire(main)
