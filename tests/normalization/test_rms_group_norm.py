from typing import Tuple, Optional
import math
import numpy as np
import torch
from torch import Tensor
import fire

from xllm_extension.ops import (
    group_rms_norm_fwd,
    group_rms_norm_bwd,
    group_rms_norm_fwd_affine,
    group_rms_norm_bwd_affine,
)


def mannual_normaliz(
    x: Tensor,
    weight: Optional[torch.Tensor],
    num_groups: int,
    eps: float,
):
    assert x.dim() == 3
    bsz, seq_len, xdim = x.size()
    x = x.view(bsz, seq_len, num_groups, -1)

    v = torch.mean(x * x, dim=-1, keepdim=True)
    y = x / torch.sqrt(v + np.float64(eps))
    y = y.view(bsz, seq_len, xdim)
    if weight is not None:
        y = y * weight

    return y


def grouprmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_groups: int,
    eps: float,
    element_affine: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    bsz, seq_len, xdim = x.size()
    if element_affine:
        y, invvar = group_rms_norm_fwd_affine(x, xdim, num_groups, weight, eps)
    else:
        y, invvar = group_rms_norm_fwd(x, xdim, num_groups, eps)

    return y, invvar


def grouprmsnorm_bwd(
    y_grad: torch.Tensor,
    x_or_y: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    num_groups: int,
    memory_efficient: bool,
    element_affine: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, seq_len, xdim = x_or_y.size()
    if element_affine:
        x_grad, weight_grad = group_rms_norm_bwd_affine(
            y_grad, x_or_y, xdim, num_groups, rstd, weight, memory_efficient
        )
        return x_grad, weight_grad
    else:
        x_grad = group_rms_norm_bwd(
            y_grad, x_or_y, xdim, num_groups, rstd, memory_efficient
        )
        return x_grad, None


def test(B: int, L: int, H: int, num_groups: int, eps: float, element_affine: bool, memory_efficient: bool, dtype: torch.dtype):
    with torch.no_grad():
        x = torch.randn(B, L, H, requires_grad=False, dtype=dtype, device="cuda")
        x = x + 0.1
        gamma = torch.randn(H, requires_grad=False, dtype=dtype, device="cuda")
        gamma = gamma.clamp(min=-0.9) + 1.0

    x = x.clone().detach().requires_grad_(True)
    gamma = gamma.clone().detach().requires_grad_(True) if element_affine else None

    y_manual = mannual_normaliz(
        x.double(), gamma.double() if gamma is not None else None, num_groups, eps,
    )

    y_manual_flat = y_manual.flatten()
    num_elem = y_manual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(B * L)
    loss = y_manual_flat @ weight
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(dtype)
    x_grad = x.grad
    gamma_grad = gamma.grad if element_affine else None

    print(f"B={B}, L={L}, H={H}, G={num_groups}, eps={eps}, affine={element_affine}, mem_effn={memory_efficient}, dtype={dtype}:")

    atol = 1e-6 if dtype == torch.float32 else 1e-3
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    # group fwd
    y_group, rstd_group = grouprmsnorm_fwd(x, gamma, num_groups, eps, element_affine)
    torch.testing.assert_close(y_group, y_manual.to(dtype), rtol=rtol, atol=atol)
    print(f"group pass fwd test")

    # group bwd
    x_grad_group, gamma_grad_group = grouprmsnorm_bwd(
        y_grad, y_group if memory_efficient else x, rstd_group, gamma, num_groups, memory_efficient, element_affine
    )
    torch.testing.assert_close(x_grad_group, x_grad, rtol=rtol, atol=atol)
    if element_affine:
        atol = 4e-6 if dtype == torch.float32 else 5e-2
        rtol = 1e-5 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(gamma_grad_group, gamma_grad, rtol=rtol, atol=atol)
    print(f"group pass bwd test")


def main(seed: int):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, H, num_groups in [
        [2, 3, 3, 1],
        [2, 3, 5, 1],
        [2, 3, 8, 1],
        [2, 3, 32, 1],
        [2, 3, 64, 1],
        [2, 3, 1024, 8],
        [2, 3, 1024, 16],
        [2, 3, 8192, 1],
        [2, 3, 16384, 2],
        [2, 3, 32768, 4],
        [2, 3, 5120, 5],
        [16, 32, 1408, 4],
        [16, 32, 1416, 4],
        [16, 32, 5824, 4],
        [16, 32, 8736, 6],
        [2, 3, 8192, 2],
        [16, 32, 1412, 4],
        [16, 32, 1765, 5],
        [16, 128, 256, 32],
        [16, 155, 1024, 4],
        [16, 255, 256, 32],
        [16, 256, 256, 32],
        [4, 512, 256, 32],
        [1, 1024, 256, 32],
        [1, 1024, 262144, 32],
        [1, 32768, 4096, 4],
        [1, 32768, 4096, 8],
        [1, 32768, 4096, 16],
        [2, 32768, 4096, 16],
        [1, 32768, 4096, 32],
        [1, 65536, 8192, 32],
    ]:
        for dtype in [torch.float32, torch.bfloat16]:
            for eps in [1e-5, 1e-6]:
                for affine in [True, False]:
                    for memory_efficient in [False, True]:
                        if num_groups > 1 and H <= 8192:
                            test(B, L, H, 1, eps, affine, memory_efficient, dtype)
                        test(B, L, H, num_groups, eps, affine, memory_efficient, dtype)


if __name__ == "__main__":
    fire.Fire(main)
