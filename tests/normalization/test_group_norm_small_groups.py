"""Check small group norms against an independent, pure PyTorch FP64 reference.

Run from the repository root:
    python -m tests.normalization.test_group_norm_small_groups --output /tmp/group_norm.json

Each input, affine parameter, and upstream gradient is rounded to the tested
dtype FIRST. The reference then uses independent FP64 leaves and FP64 autograd,
so neither its forward nor its gradients are rounded back to that dtype. The
reference defaults to CPU; --reference-device cuda uses pure PyTorch FP64 on GPU.
With --rounds, each configuration advances its own seeded generator without
reseeding. --summary-only retains all failures and per-case/round summaries.
Numerical agreement alone does not prove memory safety; the CLI also supports
small, targeted runs under compute-sanitizer.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time

import torch


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}
# Explicit elementwise thresholds; raw errors are recorded as well.
TOLERANCES = {
    torch.float64: (1e-9, 1e-8),
    torch.float32: (1e-5, 1e-4),
    torch.float16: (3e-3, 1e-2),
    torch.bfloat16: (2e-2, 5e-2),
}


def reference(norm, x, gamma, beta, dy, groups, eps, device="cpu"):
    """Return FP64 outputs, statistics, and gradients using only torch ops."""
    x64 = x.detach().to(device=device, dtype=torch.float64).requires_grad_(True)
    gamma64 = gamma.detach().to(device=device, dtype=torch.float64).requires_grad_(True) if gamma is not None else None
    beta64 = beta.detach().to(device=device, dtype=torch.float64).requires_grad_(True) if beta is not None else None
    grouped = x64.reshape(*x.shape[:-1], groups, -1)
    expected = {}
    if norm == "layer":
        mean = grouped.mean(dim=-1, keepdim=True)
        centered = grouped - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        normalized = centered * rstd
        expected["mean"] = mean.detach().reshape(-1, groups)
    else:
        rstd = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        normalized = grouped * rstd
    y = normalized.reshape_as(x64)
    leaves, names = [x64], ["dx"]
    if gamma64 is not None:
        y = y * gamma64
        leaves.append(gamma64)
        names.append("dgamma")
    if beta64 is not None:
        y = y + beta64
        leaves.append(beta64)
        names.append("dbeta")
    gradients = torch.autograd.grad(y, leaves, grad_outputs=dy.to(device=device, dtype=torch.float64))
    expected.update(zip(names, gradients))
    expected["y"] = y.detach()
    expected["rstd"] = rstd.detach().reshape(-1, groups)
    return expected


def error_metrics(actual, expected):
    atol, rtol = TOLERANCES[actual.dtype]
    actual64 = actual.detach().to(device=expected.device, dtype=torch.float64)
    difference = (actual64 - expected).abs()
    finite = torch.isfinite(actual64) & torch.isfinite(expected)
    allowed = atol + rtol * expected.abs()
    mismatches = (~finite) | (difference > allowed)

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else None

    # Transfer scalar metrics together when the reference is on the GPU.
    max_abs, error_norm, reference_norm, max_scaled, mismatch_count, nonfinite = torch.stack((
        difference.max(), torch.linalg.vector_norm(difference), torch.linalg.vector_norm(expected),
        (difference / allowed).max(), mismatches.sum(), (~finite).sum(),
    )).tolist()
    return {
        "passed": mismatch_count == 0,
        "max_abs": number(max_abs),
        "rel_l2": number(error_norm / reference_norm)
        if reference_norm > 0 else None,
        "max_scaled_error": number(max_scaled),
        "mismatches": int(mismatch_count),
        "elements": actual.numel(),
        "nonfinite": int(nonfinite),
        "atol": atol,
        "rtol": rtol,
    }


def run_case(ops, case, batch_size, seq_len, seed, result, forward_only=False,
             generator=None, reference_device="cpu", fingerprint=False):
    norm, size, groups, dtype_name, affine, memory_efficient, eps = case.values()
    dtype = DTYPES[dtype_name]
    if generator is None:
        generator = torch.Generator(device=reference_device).manual_seed(seed)
    channels = groups * size
    shape = (batch_size, seq_len, channels)
    x = (torch.randn(shape, generator=generator, dtype=torch.float64, device=reference_device) + 0.1).to(dtype)
    dy = (torch.randn(shape, generator=generator, dtype=torch.float64, device=reference_device)
          / math.sqrt(batch_size * seq_len)).to(dtype)
    gamma = (0.5 + torch.rand(channels, generator=generator, dtype=torch.float64, device=reference_device)).to(dtype) if affine else None
    beta = (0.25 * torch.randn(channels, generator=generator, dtype=torch.float64, device=reference_device)).to(dtype) if affine and norm == "layer" else None
    if fingerprint:
        sample = x.flatten()[:16].tolist()
        result["input_fingerprint"] = hashlib.sha256(struct.pack(f"{len(sample)}d", *sample)).hexdigest()
    expected = reference(norm, x, gamma, beta, dy, groups, eps, device=reference_device)
    x, dy = x.cuda(), dy.cuda()
    gamma = gamma.cuda() if gamma is not None else None
    beta = beta.cuda() if beta is not None else None

    result["stage"] = "forward"
    if norm == "layer":
        if affine:
            y, mean, rstd = ops.group_layer_norm_fwd_affine(x, channels, groups, gamma, beta, eps)
        else:
            y, mean, rstd = ops.group_layer_norm_fwd(x, channels, groups, eps)
        actual = {"y": y, "mean": mean, "rstd": rstd}
    else:
        if affine:
            y, rstd = ops.group_rms_norm_fwd_affine(x, channels, groups, gamma, eps)
        else:
            y, rstd = ops.group_rms_norm_fwd(x, channels, groups, eps)
        actual = {"y": y, "rstd": rstd}
    torch.cuda.synchronize()
    result["metrics"] = {name: error_metrics(value, expected[name]) for name, value in actual.items()}
    if forward_only:
        result["stage"] = "complete"
        result["status"] = "PASS" if all(m["passed"] for m in result["metrics"].values()) else "FAIL"
        return

    result["stage"] = "backward"
    x_or_y = y if memory_efficient else x
    if norm == "layer":
        if affine:
            dx, dgamma, dbeta = ops.group_layer_norm_bwd_affine(
                dy, x_or_y, channels, groups, mean, rstd, gamma, beta, memory_efficient)
            actual = {"dx": dx, "dgamma": dgamma, "dbeta": dbeta}
        else:
            dx = ops.group_layer_norm_bwd(dy, x_or_y, channels, groups, mean, rstd, memory_efficient)
            actual = {"dx": dx}
    else:
        if affine:
            dx, dgamma = ops.group_rms_norm_bwd_affine(
                dy, x_or_y, channels, groups, rstd, gamma, memory_efficient)
            actual = {"dx": dx, "dgamma": dgamma}
        else:
            dx = ops.group_rms_norm_bwd(dy, x_or_y, channels, groups, rstd, memory_efficient)
            actual = {"dx": dx}
    torch.cuda.synchronize()
    result["metrics"].update({name: error_metrics(value, expected[name]) for name, value in actual.items()})
    result["stage"] = "complete"
    result["status"] = "PASS" if all(m["passed"] for m in result["metrics"].values()) else "FAIL"


def isolated_case(case, args):
    """Use a fresh CUDA process for each case, including after illegal accesses."""
    with tempfile.TemporaryDirectory(prefix="group-norm-case-") as directory:
        output = Path(directory) / "result.json"
        command = [sys.executable, "-m", "tests.normalization.test_group_norm_small_groups",
                   "--output", str(output), "--batch-size", str(args.batch_size),
                   "--seq-len", str(args.seq_len), "--seed", str(args.seed),
                   "--reference-device", args.reference_device, "--shard-index", "0"]
        flags = {"norm": "--norms", "size": "--sizes", "groups": "--groups",
                 "dtype": "--dtypes", "affine": "--affine",
                 "memory_efficient": "--memory-efficient", "eps": "--eps"}
        for name, flag in flags.items():
            command.extend([flag, str(case[name])])
        if args.forward_only:
            command.append("--forward-only")
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=180,
                                     env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return {"case": case, "stage": "process", "metrics": {},
                    "status": "ERROR", "error": "isolated case timed out after 180 seconds"}
        if output.exists():
            return json.loads(output.read_text())["results"][0]
        return {"case": case, "stage": "process", "metrics": {}, "status": "ERROR",
                "error": f"exit={process.returncode}: {process.stdout} {process.stderr}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norms", nargs="+", choices=("layer", "rms"), default=["layer", "rms"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[*range(32, 65), 65, 128],
                        help="elements per group; defaults to every size 32–64 plus 65 and 128")
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 4, 32])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["float32", "float16", "bfloat16"])
    parser.add_argument("--affine", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--memory-efficient", nargs="+", type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument("--eps", nargs="+", type=float, default=[1e-5, 1e-6])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--summary-only", action="store_true", help="retain failures and aggregate metrics instead of every passing result")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--isolate", action="store_true", help="run every case in a fresh process to survive CUDA errors")
    parser.add_argument("--workers", type=int, default=4, help="concurrent processes with --isolate")
    args = parser.parse_args()
    if any(size < 1 or size > 8192 for size in args.sizes):
        parser.error("sizes must be in [1, 8192]")
    if any(group < 1 or group > 32 for group in args.groups):
        parser.error("groups must be in [1, 32]")
    if args.batch_size < 1 or args.seq_len < 1 or args.workers < 1 or any(eps <= 0 for eps in args.eps):
        parser.error("batch-size, seq-len, workers, and eps must be positive")
    if args.rounds < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("rounds and num-shards must be positive; shard-index must be in [0, num-shards)")
    if args.isolate and args.rounds != 1:
        parser.error("multi-round runs require a persistent process; omit --isolate")
    if args.output:
        args.output = Path(str(args.output).replace("{shard}", str(args.shard_index)))
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no kernel correctness tests were run.")
    import xllm_extension

    keys = ("norm", "size", "groups", "dtype", "affine", "memory_efficient", "eps")
    cases = [dict(zip(keys, values)) for values in itertools.product(
        args.norms, args.sizes, args.groups, args.dtypes, args.affine, args.memory_efficient, args.eps)]
    total_cases = len(cases)
    # Rotate assignment across contiguous blocks to distribute expensive modes.
    cases = [case for index, case in enumerate(cases)
             if (index // args.num_shards + index % args.num_shards) % args.num_shards == args.shard_index]
    generators = {tuple(case.values()): torch.Generator(device=args.reference_device).manual_seed(args.seed)
                  for case in cases} if args.rounds > 1 else {}
    summaries = {tuple(case.values()): {"case": case, "counts": {status: 0 for status in ("PASS", "FAIL", "ERROR")},
                                      "distinct_initializations": 0, "metrics": {}} for case in cases}
    fingerprints = {key: set() for key in summaries}
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "extension": xllm_extension.__file__,
        "reference": f"pure PyTorch FP64 on {args.reference_device}, including independent FP64 autograd leaves",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "rounds": args.rounds,
        "reference_device": args.reference_device,
        "generator_policy": "one seeded generator per configuration; advance continuously between rounds",
        "fingerprint": "SHA-256 of the first 16 quantized input elements, packed as FP64",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "global_cases_per_round": total_cases,
        "cases_per_round": len(cases),
        "summary_only": args.summary_only,
        "forward_only": args.forward_only,
        "isolated": args.isolate,
        "requested": len(cases) * args.rounds,
        "counts": {status: 0 for status in ("PASS", "FAIL", "ERROR")},
        "completed": 0,
        "attempted": 0,
        "not_run": len(cases) * args.rounds,
        "round_results": [],
        "case_summaries": list(summaries.values()),
        "results": [],
    }
    print(f"GPU: {report['gpu']}; shard {args.shard_index}/{args.num_shards}; "
          f"requested cases: {report['requested']}", flush=True)

    def execute(case):
        if args.isolate:
            return isolated_case(case, args)
        result = {"case": case, "stage": "setup", "metrics": {}}
        try:
            run_case(xllm_extension.ops, case, args.batch_size, args.seq_len, args.seed, result,
                     forward_only=args.forward_only, generator=generators.get(tuple(case.values())),
                     reference_device=args.reference_device, fingerprint=args.rounds > 1)
        except RuntimeError as error:
            result.update(status="ERROR", error=str(error))
        return result

    started = time.perf_counter()

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        if args.output:
            # Keep the last complete checkpoint readable while a new one is written.
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.output)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for round_index in range(1, args.rounds + 1):
            round_started = time.perf_counter()
            round_counts = {status: 0 for status in report["counts"]}
            fatal = False
            results = pool.map(execute, cases) if args.isolate else map(execute, cases)
            for index, result in enumerate(results, 1):
                result["round"] = round_index
                key = tuple(result["case"].values())
                summary = summaries[key]
                fingerprint = result.get("input_fingerprint")
                if fingerprint:
                    if fingerprint in fingerprints[key]:
                        result.update(status="FAIL", error="Repeated input initialization")
                    fingerprints[key].add(fingerprint)
                    summary["distinct_initializations"] = len(fingerprints[key])
                status = result["status"]
                report["counts"][status] += 1
                report["attempted"] += 1
                report["not_run"] -= 1
                report["completed"] += result["stage"] == "complete"
                round_counts[status] += 1
                summary["counts"][status] += 1
                for name, metric in result["metrics"].items():
                    worst = summary["metrics"].setdefault(name, {"max_abs": 0.0, "max_scaled_error": 0.0,
                                                                "nonfinite": 0, "mismatches": 0})
                    for field in ("max_abs", "max_scaled_error"):
                        if metric[field] is not None:
                            worst[field] = max(worst[field], metric[field])
                    for field in ("nonfinite", "mismatches"):
                        worst[field] += metric[field]
                if not args.summary_only or status != "PASS":
                    report["results"].append(result)
                if status != "PASS":
                    failed = [name for name, metric in result["metrics"].items() if not metric["passed"]]
                    error = result.get("error", "").splitlines()
                    print(f"{status} round={round_index} {result['case']} stage={result['stage']} failed={failed}"
                          f" {error[0] if error else ''}", flush=True)
                if args.rounds == 1 and index % 100 == 0:
                    print(f"Completed {index}/{len(cases)}", flush=True)
                # A CUDA error can poison the context; never reuse it for later cases.
                if status == "ERROR" and not args.isolate:
                    fatal = True
                    break
            report["round_results"].append({"round": round_index, "counts": round_counts,
                                             "elapsed_seconds": time.perf_counter() - round_started})
            checkpoint()
            print(f"Shard {args.shard_index}: round {round_index}/{args.rounds}: {round_counts}", flush=True)
            if fatal:
                break
    if args.output:
        print(f"Report: {args.output}", flush=True)
    print(f"Summary: {report['counts']}; not run: {report['not_run']}", flush=True)
    return int(report["counts"]["FAIL"] > 0 or report["counts"]["ERROR"] > 0 or report["not_run"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
