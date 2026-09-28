from timeit import default_timer as timer
import numpy as np
import torch
import fire

from xllm.modules.fused_ops import (
    multi_group_matmul_fwd,
    multi_group_matmul_bwd,
)


def gen_group_sizes(bsz, n_experts):
    bounds = np.sort(np.random.choice(range(1, bsz), n_experts - 1, replace=False))
    group_sizes = [0 for _ in range(n_experts)]
    prev = 0
    for i in range(n_experts - 1):
        group_sizes[i] = bounds[i] - prev
        prev = bounds[i]
    group_sizes[n_experts - 1] = bsz - prev
    assert all([gs > 0 for gs in group_sizes])
    assert sum(group_sizes) == bsz
    return group_sizes


def test_mgmm(
    inp1, inp2, y_grad, group_sizes, transpose, backend, epochs
):
    with torch.no_grad():
        for i in range(100):
            y = multi_group_matmul_fwd(inp1, inp2, group_sizes[i], transpose, backend)
            grad1, grad2 = multi_group_matmul_bwd(y_grad, inp2, inp1, group_sizes[i], transpose, backend)

        torch.cuda.synchronize()

        start = timer()
        for i in range(epochs):
            y = multi_group_matmul_fwd(inp1, inp2, group_sizes[i], transpose, backend)

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'mgmm-{backend} fwd: {delta:.2f}s')

        start = timer()
        for i in range(epochs):
            grad1, grad2 = multi_group_matmul_bwd(y_grad, inp2, inp1, group_sizes[i], transpose, backend)

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'mgmm-{backend} bwd: {delta:.2f}s')


def test(bsz: int, n_experts: int, d_in: int, d_out: int, transpose: bool, dtype: str):
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    assert bsz > n_experts

    with torch.no_grad():
        inp1 = torch.randn(bsz, d_in, requires_grad=False, dtype=pt_dtype, device="cuda")
        if transpose:
            inp2 = torch.randn(n_experts, d_out, d_in, requires_grad=False, dtype=pt_dtype, device="cuda")
        else:
            inp2 = torch.randn(n_experts, d_in, d_out, requires_grad=False, dtype=pt_dtype, device="cuda")
        y_grad = torch.randn(bsz, d_out, requires_grad=False, dtype=pt_dtype, device="cuda")

    epochs = 1000
    group_sizes = [gen_group_sizes(bsz, n_experts) for _ in range(epochs)]
    print(f"B={bsz}, N={n_experts}, D=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}:")
    test_mgmm(inp1, inp2, y_grad, group_sizes, transpose, 'sequential', epochs)
    test_mgmm(inp1, inp2, y_grad, group_sizes, transpose, 'torch', epochs)
    # test_mgmm(inp1, inp2, y_grad, group_sizes, transpose, 'nested', epochs)
    test_mgmm(inp1, inp2, y_grad, group_sizes, transpose, 'te', epochs)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, N, D1, D2 in [
        [1048576, 128, 2560, 768],
        [1048576, 64, 2560, 768],
        [1048576, 32, 2560, 768],
        [131072, 192, 6144, 1792],
        [131072, 96, 6144, 1792],
        [131072, 48, 6144, 1792],
        [131072, 24, 6144, 1792],
        [32768, 256, 7168, 2048],
        [32768, 128, 7168, 2048],
        [32768, 64, 7168, 2048],
        [32768, 32, 7168, 2048],
        [65536, 256, 7168, 2048],
        [65536, 128, 7168, 2048],
        [65536, 64, 7168, 2048],
        [65536, 32, 7168, 2048],
    ]:
        test(B, N, D1, D2, True, dtype)
        test(B, N, D2, D1, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
