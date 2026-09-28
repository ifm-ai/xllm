#!/usr/bin/env python3
"""
Distributed data-loader test for xllm.data module.

Simulates the training data flow without model computation.
Supports source logging and a save/reload reproducibility check.

Launch (single GPU):
    torchrun --nproc_per_node=1 xllm/data/test.py \\
        --data_dir /path/to/dir:1.0:text:text --tokenizer_path /path/to/tok

Launch (multiple GPUs):
    torchrun --nproc_per_node=2 xllm/data/test.py \\
        --data_dir /path/to/dir:1.0:text:text --tokenizer_path /path/to/tok

Reproducibility check (save at step 100, compare 10 steps):
    torchrun --nproc_per_node=2 xllm/data/test.py \\
        --data_dir /path/to/dir:1.0:text:text --tokenizer_path /path/to/tok \\
        --log_sources --checkpoint_step 100 --compare_steps 10
"""

import os
import sys
import argparse
import pickle
from timeit import default_timer as timer
from typing import IO, List, Optional
import numpy as np
import torch
import torch.distributed as dist

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xllm.config import TokenizerConf
from xllm.data.dataset_streamer.tokenizer.build import build_tokenizer
from xllm.data.dataloader import MultiSourceDataLoader
from xllm.data.data_types import Batch

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SEQ_LEN         = 20480
DEFAULT_BATCH_SIZE      = 4
DEFAULT_NUM_BUFFERED    = 640
DEFAULT_NUM_WORKERS     = 1
DEFAULT_NUM_STEPS       = 400
DEFAULT_LOG_FREQ        = 10
DEFAULT_CHECKPOINT_STEP = 200
DEFAULT_COMPARE_STEPS   = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1

def log(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


def write_batch_sources(f: IO, step: int, batch: Batch) -> None:
    """Write one TSV line per source entry in the batch to f."""
    if batch.src_infos is None:
        return
    for seq_idx, seq_sources in enumerate(batch.src_infos):
        if not seq_sources:
            continue
        for src in seq_sources:
            f.write(
                f"{step}\t{seq_idx}\t{src.filename}\t{src.line_num}"
                f"\t{src.is_truncation}\t{src.dataset}\n"
            )
    f.flush()


def read_source_log(path: str, step_lo: int, step_hi: int) -> List[tuple]:
    """
    Read a source log file and return records for steps in [step_lo, step_hi].
    Each record is (seq_idx, filename, line_num, is_truncation, dataset) — no step.
    """
    records = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            step = int(parts[0])
            if step_lo <= step <= step_hi:
                records.append(tuple(parts[1:]))  # drop step column
    return records


def compare_source_logs(
    path_a: str, step_lo_a: int, step_hi_a: int,
    path_b: str, step_lo_b: int, step_hi_b: int,
    rank: int,
) -> bool:
    """
    Compare the [step_lo_a, step_hi_a] window of path_a against the
    [step_lo_b, step_hi_b] window of path_b (step column is ignored).
    """
    a = read_source_log(path_a, step_lo_a, step_hi_a)
    b = read_source_log(path_b, step_lo_b, step_hi_b)

    label_a = f"run1[{step_lo_a}..{step_hi_a}]"
    label_b = f"run2[{step_lo_b}..{step_hi_b}]"

    if a == b:
        log(rank, f"[MATCH] rank{rank}: {label_a} == {label_b}  ({len(a)} records)")
        return True
    else:
        mismatches = sum(1 for x, y in zip(a, b) if x != y)
        extra = abs(len(a) - len(b))
        log(rank, f"[MISMATCH] rank{rank}: {label_a} vs {label_b} — "
                  f"{mismatches} differing records, length diff={extra}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                log(rank, f"  first diff at record {i}:")
                log(rank, f"    run1: {x}")
                log(rank, f"    run2: {y}")
                break
        return False


def build_loader(args, rank: int, world_size: int) -> MultiSourceDataLoader:
    tokenizer_cfg = TokenizerConf()
    tokenizer_cfg.type = "huggingface"
    tokenizer_cfg.path = args.tokenizer_path
    tokenizer = build_tokenizer(tokenizer_cfg)
    return MultiSourceDataLoader(
        tokenizer=tokenizer,
        data_mix_str=args.data_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_buffered_seq=args.num_buffered,
        world_rank=rank,
        world_size=world_size,
        packing_type=args.packing_type,
        num_workers=args.num_workers,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",         type=str, required=True,
                        help="Data mix string: path:weight:json_key:source_format[,path:weight:json_key:source_format...]")
    parser.add_argument("--tokenizer_path",   type=str, required=True)
    parser.add_argument("--seq_len",          type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch_size",       type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_buffered",     type=int, default=DEFAULT_NUM_BUFFERED)
    parser.add_argument("--num_workers",      type=int, default=DEFAULT_NUM_WORKERS,
                        help="Tokenizer worker threads during buffer refill")
    parser.add_argument("--num_steps",        type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--log_freq",         type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--packing_type",     type=str, default="bestfit",
                        choices=["simple", "bestfit"])
    parser.add_argument("--log_sources",      action="store_true",
                        help="Log per-batch data sources to rank{N}_run1.log / run2.log")
    parser.add_argument("--checkpoint_step",  type=int, default=DEFAULT_CHECKPOINT_STEP,
                        help="Step at which to save checkpoint")
    parser.add_argument("--compare_steps",    type=int, default=DEFAULT_COMPARE_STEPS,
                        help="How many steps after checkpoint to use for comparison")
    parser.add_argument("--output_dir",       type=str, default=".",
                        help="Directory for checkpoint and log files")
    args = parser.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank       = get_rank()
    world_size = get_world_size()

    os.makedirs(args.output_dir, exist_ok=True)
    # Each rank owns its own checkpoint and log files.
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_rank{rank}.pkl")
    log_run1_path   = os.path.join(args.output_dir, f"rank{rank}_run1.log")
    log_run2_path   = os.path.join(args.output_dir, f"rank{rank}_run2.log")

    log(rank, "=" * 60)
    log(rank, f"world_size={world_size}  seq_len={args.seq_len}  "
              f"batch_size={args.batch_size}  packing={args.packing_type}")
    log(rank, f"num_steps={args.num_steps}  checkpoint_step={args.checkpoint_step}  "
              f"compare_steps={args.compare_steps}  log_sources={args.log_sources}")
    log(rank, "=" * 60)

    data_loader = build_loader(args, rank, world_size)
    log(rank, "Data loader built.")

    global_batch_size = world_size * args.batch_size
    padding_ratios: List[float] = []
    truncation_ratios: List[float] = []

    # ── Phase 1: run num_steps steps ──────────────────────────────────────────
    # Save checkpoint at checkpoint_step.
    # Log source info for ALL steps so we can slice any window for comparison.
    log(rank, "Phase 1: running...")
    t_start = t0 = timer()
    last_tokens = total_tokens = 0

    src_log1: Optional[IO] = None
    if args.log_sources:
        src_log1 = open(log_run1_path, "w")
        src_log1.write("step\tseq_idx\tfilename\tline_num\tis_truncation\tdataset\n")

    do_repro_check = (
        args.log_sources
        and args.checkpoint_step < args.num_steps
        and args.compare_steps > 0
    )
    repro_ok = True

    for step in range(1, args.num_steps + 1):
        t1 = timer()
        batch = next(data_loader)
        data_load_time = timer() - t1

        if src_log1 is not None:
            write_batch_sources(src_log1, step, batch)

        if step == args.checkpoint_step:
            log(rank, f"Saving checkpoint -> {checkpoint_path}")
            with open(checkpoint_path, "wb") as f:
                pickle.dump(data_loader.get_state(), f)

        batch_tokens  = args.batch_size * args.seq_len
        last_tokens  += batch_tokens
        total_tokens += batch_tokens
        if args.packing_type == "bestfit":
            padding_ratios.append(batch.padding_ratio)
            truncation_ratios.append(batch.truncation_ratio)

        if step % args.log_freq == 0 or step == args.num_steps:
            delta = timer() - t0
            wps   = last_tokens * world_size / delta
            n_gb  = float(global_batch_size) * args.seq_len / 1e9 * step
            pinfo = ""
            if args.packing_type == "bestfit" and padding_ratios:
                pinfo = (f"  pad={np.mean(padding_ratios)*100:.2f}%"
                         f"  trunc={np.mean(truncation_ratios)*100:.2f}%")
            log(rank, f"step={step:6d}  tokens={n_gb:.3f}B  "
                      f"wps={wps:,.0f}  load={data_load_time*1e3:.1f}ms{pinfo}")
            last_tokens = 0
            padding_ratios.clear()
            truncation_ratios.clear()
            t0 = timer()

    if src_log1 is not None:
        src_log1.close()

    log(rank, f"Phase 1 done in {timer()-t_start:.1f}s  ({total_tokens:,} tokens on this rank)")

    # ── Phase 2: reload + run compare_steps, compare against run1 window ──────
    if do_repro_check:
        compare_start = args.checkpoint_step + 1
        compare_end   = args.checkpoint_step + args.compare_steps

        log(rank, f"Phase 2: reload from step {args.checkpoint_step}, "
                  f"run {args.compare_steps} steps...")
        with open(checkpoint_path, "rb") as f:
            data_loader.set_state(pickle.load(f))

        src_log2 = open(log_run2_path, "w")
        src_log2.write("step\tseq_idx\tfilename\tline_num\tis_truncation\tdataset\n")

        for step in range(1, args.compare_steps + 1):
            batch = next(data_loader)
            write_batch_sources(src_log2, step, batch)

        src_log2.close()
        log(rank, "Phase 2 done.")

        # Compare: run1[checkpoint_step+1 .. checkpoint_step+compare_steps]
        #      vs: run2[1 .. compare_steps]
        # Both represent the same logical training steps after the checkpoint.
        repro_ok = compare_source_logs(
            path_a=log_run1_path, step_lo_a=compare_start, step_hi_a=compare_end,
            path_b=log_run2_path, step_lo_b=1,             step_hi_b=args.compare_steps,
            rank=rank,
        )

        log(rank, f"Logs: {log_run1_path}  {log_run2_path}")

    # Stop all background threads before tearing down the process group.
    # Without this, daemon threads blocked on buffer.get() cause SIGABRT
    # when NCCL/gloo shuts down under them.
    data_loader.close()

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if repro_ok else 1


if __name__ == "__main__":
    sys.exit(main())
