from typing import Tuple, Optional
import importlib
from timeit import default_timer as timer
import numpy as np
import torch
import fire

try:
    import apex
    fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    APEX_ENABLED = False
except ImportError:
    APEX_ENABLED = False
    fused_layer_norm_cuda = None

from xllm_extension.ops import (
    group_layer_norm_fwd,
    group_layer_norm_bwd,
    group_layer_norm_fwd_affine,
    group_layer_norm_bwd_affine,
)


def apex_layernorm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    bsz, seq_len, xdim = x.size()
    if weight is not None:
        y, mean, invvar = fused_layer_norm_cuda.forward_affine(x, (xdim,), weight, bias, eps)
    else:
        y, mean, invvar = fused_layer_norm_cuda.forward(x, (xdim,), eps)

    return y, mean, invvar


def apex_layernorm_bwd(
    y_grad: torch.Tensor,
    x_or_y: torch.Tensor,
    mean: torch.Tensor,
    invvar: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    memory_efficient: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert x_or_y.dim() == 3
    bsz, seq_len, xdim = x_or_y.size()

    if weight is not None:
        x_grad, weight_grad, bias_grad = fused_layer_norm_cuda.backward_affine(
            y_grad, mean, invvar, x_or_y, (xdim,), weight, bias, eps, memory_efficient
        )
    else:
        x_grad = fused_layer_norm_cuda.backward(
            y_grad, mean, invvar, x_or_y, (xdim,), eps, memory_efficient
        )
        weight_grad = bias_grad = None

    return x_grad, weight_grad, bias_grad


def grouplayernorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    element_affine: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    bsz, seq_len, xdim = x.size()
    if element_affine:
        y, mean, rstd = group_layer_norm_fwd_affine(x, xdim, num_groups, weight, bias, eps)
    else:
        y, mean, rstd = group_layer_norm_fwd(x, xdim, num_groups, eps)
    return y, mean, rstd


def grouplayernorm_bwd(
    y_grad: torch.Tensor,
    x_or_y: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    memory_efficient: bool,
    element_affine: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, xdim = x_or_y.size()
    if element_affine:
        return group_layer_norm_bwd_affine(
            y_grad, x_or_y, xdim, num_groups, mean, rstd, weight, bias, memory_efficient
        )
    else:
        return group_layer_norm_bwd(
            y_grad, x_or_y, xdim, num_groups, mean, rstd, memory_efficient
        )


def test_apex_speed(x, y_grad, gamma, beta, memory_efficient, epochs):
    eps = 1e-6
    with torch.no_grad():
        # warm up
        for _ in range(100):
            y, mean, invvar = apex_layernorm_fwd(x, gamma, beta, eps)
            apex_layernorm_bwd(y_grad, y if memory_efficient else x, mean, invvar, gamma, beta, eps, memory_efficient)

        start = timer()
        for _ in range(epochs):
            y, mean, invvar = apex_layernorm_fwd(x, gamma, beta, eps)
            torch.cuda.synchronize()

        delta = timer() - start
        print(f'apex fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            apex_layernorm_bwd(y_grad, y if memory_efficient else x, mean, invvar, gamma, beta, eps, memory_efficient)
            torch.cuda.synchronize()

        delta = timer() - start
        print(f'apex bwd: {delta:.2f}s')


def test_group_speed(x, y_grad, gamma, beta, num_groups, element_affine, memory_efficient, epochs):
    eps = 1e-6
    with torch.no_grad():
        # warm up
        for _ in range(100):
            y, mean, rstd = grouplayernorm_fwd(x, gamma, beta, num_groups, eps, element_affine)
            grouplayernorm_bwd(
                y_grad, y if memory_efficient else x, mean, rstd, gamma, beta, num_groups, memory_efficient, element_affine
            )

        start = timer()
        for _ in range(epochs):
            y, mean, rstd = grouplayernorm_fwd(x, gamma, beta, num_groups, eps, element_affine)
            torch.cuda.synchronize()

        delta = timer() - start
        print(f'group-{num_groups} fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            grouplayernorm_bwd(
                y_grad, y if memory_efficient else x, mean, rstd, gamma, beta, num_groups, memory_efficient, element_affine
            )
            torch.cuda.synchronize()

        delta = timer() - start
        print(f'group-{num_groups} bwd: {delta:.2f}s')


def test(B: int, L: int, H: int, num_groups: int, element_affine: bool, memory_efficient: bool, dtype: torch.dtype):

    with torch.no_grad():
        x = torch.randn(B, L, H, requires_grad=False, dtype=dtype, device="cuda")
        x = x + 0.1
        gamma = torch.randn(H, requires_grad=False, dtype=dtype, device="cuda")
        gamma = gamma.clamp(min=-0.9) + 1.0
        y_grad = torch.randn(B, L, H, requires_grad=False, dtype=dtype, device="cuda")

    x = x.detach()
    gamma = gamma.clone().detach() if element_affine else None
    beta = torch.randn(H, requires_grad=True, dtype=dtype, device="cuda") if element_affine else None
    y_grad = y_grad.detach()

    epochs = 10000

    print(f"B={B}, L={L}, H={H}, G={num_groups}, affine={element_affine}, mem_effn={memory_efficient}, dtype={dtype}:")

    if APEX_ENABLED:
        test_apex_speed(x, y_grad, gamma, beta, memory_efficient, epochs)

    test_group_speed(x, y_grad, gamma, beta, 1, element_affine, memory_efficient, epochs)
    test_group_speed(x, y_grad, gamma, beta, num_groups, element_affine, memory_efficient, epochs)


def main(seed: int):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, H, num_groups in [
        [1024, 3, 1024, 2],
        [1024, 3, 1024, 8],
        [1024, 3, 1024, 32],
        [1024, 3, 8192, 4],
        [1024, 3, 5120, 5],
        [16, 32, 1412, 4],
        [1024, 32, 1765, 5],
        [512, 155, 1024, 4],
        [512, 155, 1024, 8],
        [4, 1024, 2048, 32],
        [4, 1024, 4096, 32],
        [1, 16384, 4096, 4],
        [1, 32768, 4096, 4],
        [1, 32768, 4096, 32],
        [2, 32768, 8192, 8],
        [1, 65536, 8192, 8],
        [1, 65536, 8192, 32],
    ]:
        for affine in [True, False]:
            for memory_efficient in [False, True]:
                for dtype in [torch.float32, torch.bfloat16]:
                    test(B, L, H, num_groups, affine, memory_efficient, dtype)


if __name__ == "__main__":
    fire.Fire(main)
