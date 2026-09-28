"""Compare the current extension with a saved pre-change extension on one GPU.

python -m tests.normalization.benchmark_group_norm_regression \
    --baseline /path/to/old/xllm_extension.cpython-311-x86_64-linux-gnu.so \
    --output /tmp/group_norm_performance.json

Uses CUDA graphs to measure device execution, alternating baseline/current order
between rounds. Outputs and gradients must also match the baseline exactly.
Run without other GPU workloads. This does not measure Python/host dispatch time.
"""

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys
import types

import torch


def load_baseline(path):
    package = types.ModuleType("group_norm_baseline")
    package.__path__ = []
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.xllm_extension", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ops


def operations(ops, norm, x, dy, gamma, beta, groups, affine, memory_efficient):
    channels = x.shape[-1]
    eps = 1e-5
    if norm == "layer":
        if affine:
            def forward():
                return ops.group_layer_norm_fwd_affine(x, channels, groups, gamma, beta, eps)
        else:
            def forward():
                return ops.group_layer_norm_fwd(x, channels, groups, eps)
        y, mean, rstd = forward()
        x_or_y = y if memory_efficient else x
        if affine:
            def backward():
                return ops.group_layer_norm_bwd_affine(
                    dy, x_or_y, channels, groups, mean, rstd, gamma, beta, memory_efficient)
        else:
            def backward():
                return (ops.group_layer_norm_bwd(
                    dy, x_or_y, channels, groups, mean, rstd, memory_efficient),)
    else:
        if affine:
            def forward():
                return ops.group_rms_norm_fwd_affine(x, channels, groups, gamma, eps)
        else:
            def forward():
                return ops.group_rms_norm_fwd(x, channels, groups, eps)
        y, rstd = forward()
        x_or_y = y if memory_efficient else x
        if affine:
            def backward():
                return ops.group_rms_norm_bwd_affine(
                    dy, x_or_y, channels, groups, rstd, gamma, memory_efficient)
        else:
            def backward():
                return (ops.group_rms_norm_bwd(
                    dy, x_or_y, channels, groups, rstd, memory_efficient),)
    return {"forward": forward, "backward": backward}


def capture(function, unroll):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(unroll):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def measure(graph, unroll, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (unroll * replays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[65, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--rows", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--unroll", type=int, default=16)
    parser.add_argument("--replays", type=int, default=4)
    args = parser.parse_args()
    if any(size < 65 or size > 8192 for size in args.sizes):
        parser.error("the pre-change extension requires sizes in [65, 8192]")
    if min(*args.rows, args.rounds, args.unroll, args.replays) < 1:
        parser.error("rows, rounds, unroll, and replays must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(20260916)
    import xllm_extension

    baseline = load_baseline(args.baseline.resolve())
    current = xllm_extension.ops
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "baseline": str(args.baseline.resolve()), "current": xllm_extension.__file__,
        "rounds": args.rounds, "unroll": args.unroll, "replays": args.replays,
        "timing": "CUDA graph replay, alternating baseline/current order; microseconds per operation",
        "results": [],
    }
    cases = list(itertools.product(("layer", "rms"), args.sizes, args.rows,
                                   ("float32", "bfloat16"), ((False, False), (True, False), (True, True))))
    print(f"GPU: {report['gpu']}; cases: {len(cases)}", flush=True)
    with torch.no_grad():
        for index, (norm, size, rows, dtype_name, (affine, memory_efficient)) in enumerate(cases, 1):
            groups = min(32, max(1, 4096 // size))
            channels = groups * size
            dtype = getattr(torch, dtype_name)
            x = torch.randn(1, rows, channels, device="cuda", dtype=dtype)
            dy = torch.randn_like(x)
            gamma = 0.5 + torch.rand(channels, device="cuda", dtype=dtype)
            beta = 0.25 * torch.randn_like(gamma)
            old = operations(baseline, norm, x, dy, gamma, beta, groups, affine, memory_efficient)
            new = operations(current, norm, x, dy, gamma, beta, groups, affine, memory_efficient)
            row = {"norm": norm, "size": size, "rows": rows, "groups": groups,
                   "dtype": dtype_name, "affine": affine, "memory_efficient": memory_efficient}
            for phase in ("forward", "backward"):
                for expected, actual in zip(old[phase](), new[phase]()):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                graphs = {"baseline": capture(old[phase], args.unroll),
                          "current": capture(new[phase], args.unroll)}
                for graph in graphs.values():
                    for _ in range(5):
                        graph.replay()
                torch.cuda.synchronize()
                samples = {name: [] for name in graphs}
                for round_index in range(args.rounds):
                    order = ("baseline", "current") if round_index % 2 == 0 else ("current", "baseline")
                    for name in order:
                        samples[name].append(measure(graphs[name], args.unroll, args.replays))
                old_us = statistics.median(samples["baseline"])
                new_us = statistics.median(samples["current"])
                row[phase] = {"baseline_us": old_us, "current_us": new_us,
                              "ratio": new_us / old_us, "samples": samples}
                del graphs, graph
            report["results"].append(row)
            if index % 24 == 0:
                print(f"Completed {index}/{len(cases)}", flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for phase in ("forward", "backward"):
        ratios = [row[phase]["ratio"] for row in report["results"]]
        print(f"{phase}: median current/baseline={statistics.median(ratios):.4f}, "
              f"range=[{min(ratios):.4f}, {max(ratios):.4f}]")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
