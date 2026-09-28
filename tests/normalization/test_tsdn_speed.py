from timeit import default_timer as timer

import fire
import torch

import xllm_extension.ops as xllm_ops


def benchmark_timestep_decay_norm(
    x,
    bos_mask,
    prev_count,
    prev_mean,
    prev_var,
    weight,
    bias,
    y_grad,
    mean_grad,
    var_grad,
    padding_mask,
    num_groups,
    beta1,
    beta2,
    eps,
    warmup,
    epochs,
):
    with torch.no_grad():
        for _ in range(warmup):
            y, count, mean, var, cummean, cumrstd = xllm_ops.group_timestep_decay_norm_fwd(
                x,
                bos_mask,
                prev_count,
                prev_mean,
                prev_var,
                weight,
                bias,
                padding_mask,
                num_groups,
                beta1,
                beta2,
                eps,
            )
            xllm_ops.group_timestep_decay_norm_bwd(
                y_grad,
                mean_grad,
                var_grad,
                x,
                prev_count,
                bos_mask,
                cummean,
                cumrstd,
                weight,
                bias,
                padding_mask,
                num_groups,
                beta1,
                beta2,
                eps,
                False,
            )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            y, count, mean, var, cummean, cumrstd = xllm_ops.group_timestep_decay_norm_fwd(
                x,
                bos_mask,
                prev_count,
                prev_mean,
                prev_var,
                weight,
                bias,
                padding_mask,
                num_groups,
                beta1,
                beta2,
                eps,
            )
        torch.cuda.synchronize()
        fwd_delta = timer() - start

        start = timer()
        for _ in range(epochs):
            xllm_ops.group_timestep_decay_norm_bwd(
                y_grad,
                mean_grad,
                var_grad,
                x,
                prev_count,
                bos_mask,
                cummean,
                cumrstd,
                weight,
                bias,
                padding_mask,
                num_groups,
                beta1,
                beta2,
                eps,
                False,
            )
        torch.cuda.synchronize()
        bwd_delta = timer() - start

    return {"fwd": fwd_delta, "bwd": bwd_delta}


def benchmark_timestep_decay_norm_cub(
    x,
    bos_mask,
    prev_count,
    prev_mean,
    prev_var,
    weight,
    bias,
    y_grad,
    mean_grad,
    var_grad,
    padding_mask,
    num_groups,
    beta1,
    beta2,
    eps,
    warmup,
    epochs,
):
    with torch.no_grad():
        for _ in range(warmup):
            y, count, mean, var, cummean, cumrstd = xllm_ops.group_timestep_decay_norm_cub_fwd(
                x,
                bos_mask,
                prev_count,
                prev_mean,
                prev_var,
                weight,
                bias,
                padding_mask,
                num_groups,
                beta1,
                beta2,
                eps,
            )
            xllm_ops.group_timestep_decay_norm_cub_bwd(
                y_grad,
                mean_grad,
                var_grad,
                x,
                prev_count,
                bos_mask,
                cummean,
                cumrstd,
                weight,
                padding_mask,
                num_groups,
                beta1,
                beta2,
            )

        torch.cuda.synchronize()

        start = timer()
        for _ in range(epochs):
            y, count, mean, var, cummean, cumrstd = xllm_ops.group_timestep_decay_norm_cub_fwd(
                x,
                bos_mask,
                prev_count,
                prev_mean,
                prev_var,
                weight,
                bias,
                padding_mask,
                num_groups,
                beta1,
                beta2,
                eps,
            )
        torch.cuda.synchronize()
        fwd_delta = timer() - start

        start = timer()
        for _ in range(epochs):
            xllm_ops.group_timestep_decay_norm_cub_bwd(
                y_grad,
                mean_grad,
                var_grad,
                x,
                prev_count,
                bos_mask,
                cummean,
                cumrstd,
                weight,
                padding_mask,
                num_groups,
                beta1,
                beta2,
            )
        torch.cuda.synchronize()
        bwd_delta = timer() - start

    return {"fwd": fwd_delta, "bwd": bwd_delta}


def print_benchmark(name, result, epochs):
    print(f"{name} fwd: {result['fwd']:.2f}s")
    print(f"{name} bwd: {result['bwd']:.2f}s")


def print_speedup(baseline, candidate):
    print(f"fwd speedup: {baseline['fwd'] / candidate['fwd']:.3f}x")
    print(f"bwd speedup: {baseline['bwd'] / candidate['bwd']:.3f}x")


def test(
    B: int,
    L: int,
    H: int,
    dtype: str,
    epochs: int,
    warmup: int,
    bos_ratio: float,
    padding_ratio: float,
):
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    assert H % 64 == 0, "H must be divisible by 64"
    assert epochs > 0, "epochs must be positive"
    assert warmup >= 0, "warmup must be non-negative"
    assert dtype in {"fp32", "bf16", "fp16"}, "dtype must be one of: fp32, bf16, fp16"
    assert 0.0 <= bos_ratio <= 1.0, "bos_ratio must be in [0, 1]"
    assert 0.0 <= padding_ratio <= 1.0, "padding_ratio must be in [0, 1]"

    num_groups = H // 64
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

    with torch.no_grad():
        x = torch.randn(B, L, H, requires_grad=False, dtype=pt_dtype, device="cuda")
        bos_mask = torch.rand(B, L, requires_grad=False, device="cuda") < bos_ratio
        padding_mask = None
        if padding_ratio > 0.0:
            padding_mask = torch.rand(B, L, requires_grad=False, device="cuda") < padding_ratio
        prev_count = torch.zeros(B, dtype=torch.int64, device="cuda")
        prev_mean = torch.zeros(B, num_groups, dtype=pt_dtype, device="cuda")
        prev_var = torch.zeros(B, num_groups, dtype=pt_dtype, device="cuda")
        weight = torch.randn(H, requires_grad=False, dtype=pt_dtype, device="cuda")
        bias = torch.randn(H, requires_grad=False, dtype=pt_dtype, device="cuda")
        y_grad = torch.randn(B, L, H, requires_grad=False, dtype=pt_dtype, device="cuda")
        mean_grad = torch.zeros(B, num_groups, dtype=pt_dtype, device="cuda")
        var_grad = torch.zeros(B, num_groups, dtype=pt_dtype, device="cuda")

    print(
        f"B={B}, L={L}, H={H}, num_groups={num_groups}, dtype={dtype}, "
        f"warmup={warmup}, epochs={epochs}, bos_ratio={bos_ratio}, "
        f"padding_ratio={padding_ratio}:"
    )
    baseline = benchmark_timestep_decay_norm(
        x,
        bos_mask,
        prev_count,
        prev_mean,
        prev_var,
        weight,
        bias,
        y_grad,
        mean_grad,
        var_grad,
        padding_mask,
        num_groups,
        0.999,
        0.9999,
        1e-5,
        warmup,
        epochs,
    )
    candidate = benchmark_timestep_decay_norm_cub(
        x,
        bos_mask,
        prev_count,
        prev_mean,
        prev_var,
        weight,
        bias,
        y_grad,
        mean_grad,
        var_grad,
        padding_mask,
        num_groups,
        0.999,
        0.9999,
        1e-5,
        warmup,
        epochs,
    )
    print_benchmark("timestep_decay_norm", baseline, epochs)
    print_benchmark("timestep_decay_norm_cub", candidate, epochs)
    print_speedup(baseline, candidate)
    print("*" * 70)


def main(
    seed: int = 1,
    dtype: str = "bf16",
    epochs: int = 1000,
    warmup: int = 100,
    bos_ratio: float = 0.01,
    padding_ratio: float = 0.0,
):
    print(f"Initializing random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for B, L, H in [
        [8, 128, 1024],
        [8, 255, 1024],
        [8, 512, 1024],
        [8, 1024, 1024],
        [8, 32768, 2048],
        [4, 65536, 2048],
        [1, 16384, 4096],
        [1, 32768, 4096],
        [1, 65536, 4096],
    ]:
        test(B, L, H, dtype, epochs, warmup, bos_ratio, padding_ratio)


if __name__ == "__main__":
    fire.Fire(main)