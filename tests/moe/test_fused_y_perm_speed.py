from timeit import default_timer as timer
import numpy as np
import torch
import fire

from xllm.modules.moe.permute import fused_y_perm_bwd
from xllm.models.fused_blocks.utils import  (
    fused_permute_y_fwd,
    fused_permute_y_bwd_y_grad,
    fused_permute_y_bwd_scores_grad
)


def benchmark_permute_y(
    sorted_indices, y, routing_scores, out_grad, bsz, slen, topk, backend, epochs
):
    with torch.no_grad():
        for i in range(100):
            fused_permute_y_fwd(
                sorted_indices, y, routing_scores, bsz, slen, topk, backend
            )
            y_grad, inverse_indices = fused_permute_y_bwd_y_grad(
                out_grad, sorted_indices, routing_scores, bsz, slen, topk, backend
            )
            fused_permute_y_bwd_scores_grad(
                out_grad, inverse_indices, y, bsz, slen, topk, backend
            )
            if backend == 'triton':
                fused_y_perm_bwd(
                    out_grad, inverse_indices, y, routing_scores, bsz, slen, topk
                )

        torch.cuda.synchronize()

        start = timer()
        for i in range(epochs):
            fused_permute_y_fwd(sorted_indices, y, routing_scores, bsz, slen, topk, backend)

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'perm_y-{backend} fwd: {delta:.2f}s')

        start = timer()
        for i in range(epochs):
            fused_permute_y_bwd_y_grad(
                out_grad, sorted_indices, routing_scores, bsz, slen, topk, backend
            )
            fused_permute_y_bwd_scores_grad(
                out_grad, inverse_indices, y, bsz, slen, topk, backend
            )

        torch.cuda.synchronize()

        delta = timer() - start
        print(f'perm_y-{backend} bwd: {delta:.2f}s')

        if backend == 'triton':
            start = timer()
            for i in range(epochs):
                fused_y_perm_bwd(
                    out_grad, inverse_indices, y, routing_scores, bsz, slen, topk
                )

            torch.cuda.synchronize()

            delta = timer() - start
            print(f'perm_y-fused bwd: {delta:.2f}s')


def run_benchmark_case(B, L, K, D, dtype):
    N = B * L * K
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        sorted_indices = torch.randperm(N, device="cuda")
        y = torch.randn(N, D, device="cuda", dtype=pt_dtype)
        routing_scores = torch.randn(B, L, K, device="cuda", dtype=torch.float32).softmax(dim=-1).to(pt_dtype)
        out_grad = torch.randn(B, L, D, dtype=pt_dtype, device="cuda")

    epochs = 1000
    print(f"B={B}, L={L}, K={K}, D={D}, dtype={dtype}:")
    benchmark_permute_y(sorted_indices, y, routing_scores, out_grad, B, L, K, 'torch', epochs)
    benchmark_permute_y(sorted_indices, y, routing_scores, out_grad, B, L, K, 'triton', epochs)

def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L, K, D in [
        [16, 8192, 8, 2560],
        [32, 8192, 8, 1280],
        [64, 8192, 8, 640],
        [2, 8192, 8, 6144],
        [4, 8192, 8, 3072],
        [8, 8192, 8, 1536],
        [16, 8192, 8, 768],
        [4, 8192, 8, 7168],
        [8, 8192, 8, 3584],
        [16, 8192, 8, 1792],
        [32, 8192, 8, 896],
    ]:
        run_benchmark_case(B, L, K, D, dtype)


if __name__ == "__main__":
    fire.Fire(main)
