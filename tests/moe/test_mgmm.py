import numpy as np
import torch
from torch import Tensor
import fire

from xllm.modules.fused_ops import (
    multi_group_matmul_fwd,
    multi_group_matmul_bwd,
)


def manual_mgmm(
    inp1: Tensor,
    inp2: Tensor,
    group_sizes,
    transpose
):
    n_bsz = inp2.shape[0]
    assert len(group_sizes) == n_bsz

    inp1_list = torch.split(inp1, group_sizes, dim=0)
    out_list = []

    for i in range(len(inp1_list)):
        inp2_i = inp2[i].t() if transpose else inp2[i]
        out = torch.mm(inp1_list[i], inp2_i)
        out_list.append(out)

    out = torch.cat(out_list, dim=0)
    return out


def pytorch_mgmm_fwd(
    inp1: Tensor,
    inp2: Tensor,
    group_sizes,
    transpose,
    backend,
):
    return multi_group_matmul_fwd(inp1, inp2, group_sizes, transpose, backend)


def pytorch_mgmm_bwd(
    y_grad: Tensor,
    inp2: Tensor,
    inp1: Tensor,
    group_sizes,
    transpose,
    backend,
):
    return multi_group_matmul_bwd(y_grad, inp2, inp1, group_sizes, transpose, backend)


def test(bsz: int, n_experts: int, d_in: int, d_out: int, transpose: bool, dtype: str):
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    assert bsz > n_experts
    bounds = np.sort(np.random.choice(range(1, bsz), n_experts - 1, replace=True))
    group_sizes = [0 for _ in range(n_experts)]
    prev = 0
    for i in range(n_experts - 1):
        group_sizes[i] = bounds[i] - prev
        prev = bounds[i]
    group_sizes[n_experts - 1] = bsz - prev
    assert all([gs >= 0 for gs in group_sizes])
    assert sum(group_sizes) == bsz

    with torch.no_grad():
        inp1 = torch.randn(bsz, d_in, requires_grad=False, dtype=pt_dtype, device="cuda")
        if transpose:
            inp2 = torch.randn(n_experts, d_out, d_in, requires_grad=False, dtype=pt_dtype, device="cuda")
        else:
            inp2 = torch.randn(n_experts, d_in, d_out, requires_grad=False, dtype=pt_dtype, device="cuda")

    inp1 = inp1.clone().detach().requires_grad_(True)
    inp2 = inp2.clone().detach().requires_grad_(True)

    y_manual = manual_mgmm(inp1.double(), inp2.double(), group_sizes, transpose)
    y_manual_flat = y_manual.flatten()
    num_elem = y_manual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.double, device="cuda") / bsz
    loss = y_manual_flat @ weight
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(pt_dtype)
    inp1_grad = inp1.grad
    inp2_grad = inp2.grad

    # sequential mgmm
    with torch.no_grad():
        atol = {"fp32": 1.5e-3, "bf16": 1e-2, "fp16": 5e-3}[dtype]
        rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        y_seq = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'sequential')
        inp1_grad_seq, inp2_grad_seq = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'sequential')
        torch.testing.assert_close(y_seq, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass sequential fwd test")
        torch.testing.assert_close(inp1_grad_seq, inp1_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(inp2_grad_seq, inp2_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass sequential bwd test")

        # y_nest = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'nested')
        # inp1_grad_nest, inp2_grad_nest = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'nested')
        # torch.testing.assert_close(y_nest, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        # print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass nested fwd test")
        # torch.testing.assert_close(inp1_grad_nest, inp1_grad, rtol=rtol, atol=atol)
        # torch.testing.assert_close(inp2_grad_nest, inp2_grad, rtol=rtol, atol=atol)
        # print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass nested bwd test")
        #
        y_pt = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'torch')
        inp1_grad_pt, inp2_grad_pt = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'torch')
        torch.testing.assert_close(y_pt, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass pytorch fwd test")
        torch.testing.assert_close(inp1_grad_pt, inp1_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(inp2_grad_pt, inp2_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass pytorch bwd test")

        y_te = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'te')
        inp1_grad_te, inp2_grad_te = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'te')
        torch.testing.assert_close(y_te, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass te fwd test")
        torch.testing.assert_close(inp1_grad_te, inp1_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(inp2_grad_te, inp2_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass te bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, N, D1, D2 in [
        [5, 2, 4, 5],
        [13, 3, 7, 10],
        [128, 8, 16, 16],
        [253, 7, 21, 23],
        [512, 8, 32, 64],
        [1024, 16, 256, 1024],
        [2048, 32, 512, 256],
        [2048, 32, 512, 2048],
        [4834, 24, 4096, 14336],
        [8257, 64, 14336, 4096],
        [31456, 64, 4096, 8192]
    ]:
        test(B, N, D1, D2, False, dtype)
        test(B, N, D1, D2, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
