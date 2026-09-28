from timeit import default_timer as timer
import math
import numpy as np
import torch
import torch.nn.functional as F
import fire

from xllm.modules.fused_ops import (
    causal_conv1d_fwd,
    causal_conv1d_bwd,
)


def test_causal_conv1d(
    x, hx, weight, bias, y_grad, final_state_grad, bos_mask, epochs, activation, backend
):
    with torch.no_grad():
        # warm up
        for _ in range(100):
            causal_conv1d_fwd(
                x, weight, bias, hx, bos_mask, output_final_state=True, activation=activation, backend=backend
            )
            causal_conv1d_bwd(
                y_grad, final_state_grad, x, weight, bias, hx, bos_mask, activation=activation, backend=backend
            )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            causal_conv1d_fwd(
                x, weight, bias, hx, bos_mask, output_final_state=True, activation=activation, backend=backend
            )

        torch.cuda.synchronize()

        delta = timer() - start
        if hx is None:
            print(f'{backend} causal conv1d w.o. hx fwd: {delta:.2f}s')
        else:
            print(f'{backend} causal conv1d w. hx fwd: {delta:.2f}s')

        start = timer()
        for _ in range(epochs):
            causal_conv1d_bwd(
                y_grad, final_state_grad, x, weight, bias, hx, bos_mask, activation=activation, backend=backend
            )

        torch.cuda.synchronize()

        delta = timer() - start
        if hx is None:
            print(f'{backend} causal conv1d w.o. hx bwd: {delta:.2f}s')
        else:
            print(f'{backend} causal conv1d w. hx bwd: {delta:.2f}s')


def test(B: int, L: int, D: int, W: int, avg_len: int, activation: str|None, dtype: str):
    if avg_len == 0 or avg_len > L:
        avg_len = L
    bos_ratio = float(L - avg_len) / (L * avg_len)
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        x = torch.randn(B, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        y_grad = torch.randn(B, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        conv_w = torch.randn(D, W, requires_grad=False, dtype=pt_dtype, device="cuda")
        init_states = torch.randn(B, W - 1, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        final_state_grad = torch.randn(B, W - 1, D, requires_grad=False, dtype=pt_dtype, device="cuda")

        bos_mask = torch.rand(B, L, requires_grad=False, device='cuda') < bos_ratio

    epochs = 1000
    print(f"B={B}, L={L}, D={D}, W={W}, AvgL={avg_len}, act={activation}, dtype={dtype}:")
    test_causal_conv1d(x, None, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, activation, 'fla')
    test_causal_conv1d(x, None, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, activation, 'triton')

    test_causal_conv1d(x, init_states, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, activation, 'fla')
    test_causal_conv1d(x, init_states, conv_w, None, y_grad, final_state_grad, bos_mask, epochs, activation, 'triton')
    print("*" * 70)


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B, L, D in [
        [1, 16384, 4096],
        [1, 32768, 4096],
        [8, 32768, 2048],
        [1, 65536, 2560],
        [4, 65536, 2048],
    ]:
        for avg_len in [10, 100, 1000, 2000, 5000, 10000, 16000, 32000, 0]:
            W = 4
            test(B, L, D, W, avg_len, None, dtype)
            test(B, L, D, W, avg_len, 'silu', dtype)


if __name__ == "__main__":
    fire.Fire(main)
