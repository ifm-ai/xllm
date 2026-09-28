#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise AssertionError(f"Missing JSONL file: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index(records: List[Dict[str, Any]], key: str) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    for record in records:
        value = int(record[key])
        if value in indexed:
            raise AssertionError(f"Duplicate {key}={value}")
        indexed[value] = record
    return indexed


def _compare_optimizer_metrics(
    baseline: Path,
    resumed: Path,
    checkpoint_step: int,
    final_step: int,
) -> None:
    baseline_records = _index(_read_jsonl(baseline / "metrics.optim.jsonl"), "step")
    resumed_records = _index(_read_jsonl(resumed / "metrics.optim.jsonl"), "step")
    fields = ("lr", "avg_loss", "max_loss", "g_norm", "clip_cumulative")
    for step in range(checkpoint_step + 1, final_step + 1):
        expected = baseline_records[step]
        actual = resumed_records[step]
        for field in fields:
            if expected[field] != actual[field]:
                raise AssertionError(
                    f"Optimizer metric mismatch at step {step} for {field}: "
                    f"{expected[field]!r} != {actual[field]!r}"
                )


def _compare_data_sources(
    baseline: Path,
    resumed: Path,
    checkpoint_step: int,
    final_step: int,
) -> None:
    relative_path = Path("data_source_logs/rank_00000.jsonl")
    baseline_records = _index(_read_jsonl(baseline / relative_path), "step")
    resumed_records = _index(_read_jsonl(resumed / relative_path), "step")
    for step in range(checkpoint_step, final_step):
        if baseline_records[step] != resumed_records[step]:
            raise AssertionError(f"Data source mismatch at zero-based training step {step}")


def _check_eval(resumed: Path, final_step: int) -> None:
    eval_records = _read_jsonl(resumed / "metrics.eval.jsonl")
    matching = [
        record
        for record in eval_records
        if int(record["global_step"]) == final_step
    ]
    if len(matching) != 1:
        raise AssertionError(
            f"Expected one eval record for step {final_step}, found {len(matching)}"
        )
    record = matching[0]
    if not any(key.startswith("ppl/") for key in record):
        raise AssertionError("PPL metric is missing from eval output")
    if not any(key.startswith("task/copa/") for key in record):
        raise AssertionError("COPA metric is missing from eval output")
    if not (resumed / "eval" / str(final_step) / "copa.pkl").is_file():
        raise AssertionError("COPA eval artifact is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=200)
    parser.add_argument("--final-step", type=int, default=300)
    args = parser.parse_args()

    baseline = args.run_root / "baseline"
    resumed = args.run_root / "resume"
    _compare_optimizer_metrics(
        baseline,
        resumed,
        args.checkpoint_step,
        args.final_step,
    )
    _compare_data_sources(
        baseline,
        resumed,
        args.checkpoint_step,
        args.final_step,
    )
    _check_eval(resumed, args.final_step)
    print(
        f"PASS: resume steps {args.checkpoint_step + 1}-{args.final_step} "
        "and synchronous eval match expectations"
    )


if __name__ == "__main__":
    main()
