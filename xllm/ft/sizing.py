"""
Sizing gate for the in-memory copies. It runs in two stages: /dev/shm and RAM at
startup, from the local parameter count; GPU after the first steps, from the
measured peak. A shortfall is printed with the numbers that caused it.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from xllm.ft.node import NodeResources, discover_node

logger = logging.getLogger()

SLOT_BYTES_PER_PARAM = 12  # fp32 params + exp_avg + exp_avg_sq
SLOTS_PER_RANK = 4  # own A/B + peer A/B
PINNED_BYTES_PER_PARAM = 24  # two pinned host staging buffers per rank
MODEL_STAGING_BYTES_PER_PARAM = 4  # DCP model-state CPU staging at a checkpoint
OPTIM_STAGING_BYTES_PER_PARAM = 8  # DCP optimizer-state CPU staging, only with dcp_for_optimizer
GPU_STAGING_BYTES_PER_PARAM = 24  # persistent GPU snapshot buffers
USABLE_FRACTION = 0.8  # share of /dev/shm and RAM the copies may use
GPU_PEAK_STEPS = 10  # steps whose memory peak is measured before the GPU check


@dataclass
class SizingCheck:
    what: str
    needed: int
    available: int

    @property
    def ok(self) -> bool:
        return self.needed <= self.available

    def __str__(self) -> str:
        verdict = "ok" if self.ok else f"SHORT by {(self.needed - self.available) / 1e9:.1f} GB"
        return f"ft sizing {self.what}: need {self.needed / 1e9:.1f} GB, have {self.available / 1e9:.1f} GB -> {verdict}"


def local_param_count(model: torch.nn.Module) -> int:
    count = 0
    for p in model.parameters():
        count += p.to_local().numel() if hasattr(p, "to_local") else p.numel()
    return count


def check_host(local_params: int, node: NodeResources, dcp_for_optimizer: bool) -> List[SizingCheck]:
    """Per-node checks at startup; local_params is the largest per-rank count on the node."""
    q, g = local_params, node.gpus
    shm = SLOT_BYTES_PER_PARAM * SLOTS_PER_RANK * q * g
    staging = PINNED_BYTES_PER_PARAM + MODEL_STAGING_BYTES_PER_PARAM
    if dcp_for_optimizer:
        staging += OPTIM_STAGING_BYTES_PER_PARAM
    return [
        SizingCheck("/dev/shm", shm, int(USABLE_FRACTION * node.shm_bytes)),
        SizingCheck("RAM", shm + staging * q * g, int(USABLE_FRACTION * node.ram_bytes)),
    ]


def check_gpu(peak_reserved: int, local_params: int, gpu_bytes: int) -> SizingCheck:
    return SizingCheck("GPU", peak_reserved + GPU_STAGING_BYTES_PER_PARAM * local_params, gpu_bytes)


class GpuPeakRecorder:
    """
    Records the reserved-memory peak of the first steps. Call record(step) just before
    the loop resets the peak statistics, so each call sees the previous iteration's peak.
    A sample is taken only when the step advanced since the last call: the first call
    (the model build) and iterations that skipped the step (loss-scale overflow) do not count.
    """

    def __init__(self, local_params: int, steps: int = GPU_PEAK_STEPS):
        self.local_params = local_params
        self.steps = steps
        self.peaks: List[int] = []
        self.last_step: Optional[int] = None
        self.verdict: Optional[SizingCheck] = None
        self.allowed = False

    @property
    def copies_allowed(self) -> bool:
        return self.allowed

    def record(self, step: int):
        if self.verdict is not None:
            return
        advanced = self.last_step is not None and step != self.last_step
        self.last_step = step
        if not advanced:
            return
        self.peaks.append(torch.cuda.max_memory_reserved())
        if len(self.peaks) == self.steps:
            gpu_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            self.verdict = check_gpu(max(self.peaks), self.local_params, gpu_bytes)
            # copies are collective: they go on only if every rank fits
            nccl = torch.distributed.is_initialized() and torch.distributed.get_backend() == "nccl"
            ok = torch.tensor([int(self.verdict.ok)], device="cuda" if nccl else "cpu")  # the backend's, on every rank
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
            self.allowed = bool(ok.item())
            log = logger.info if self.allowed else logger.error
            note = "" if self.allowed else "; in-memory copies stay off" + ("" if not self.verdict.ok else " (another rank is short)")
            log(f"{self.verdict}{note}")


def startup_gate(model: torch.nn.Module, slot_dir: str, dcp_for_optimizer: bool) -> GpuPeakRecorder:
    """Stage 1: refuse ft before training if the copies cannot fit on this node. Collective."""
    q = torch.tensor([local_param_count(model)], dtype=torch.int64, device="cuda")
    torch.distributed.all_reduce(q, op=torch.distributed.ReduceOp.MAX)
    checks = check_host(int(q.item()), discover_node(slot_dir), dcp_for_optimizer)
    for check in checks:
        logger.info(str(check))
    short = [str(c) for c in checks if not c.ok]
    # every rank must take the same decision, or the ranks that go on hang in their next collective
    any_short = torch.tensor([int(bool(short))], device="cuda")
    torch.distributed.all_reduce(any_short, op=torch.distributed.ReduceOp.MAX)
    if any_short.item():
        raise RuntimeError("ft sizing gate refused ft: " + ("; ".join(short) if short else "another node is short"))
    return GpuPeakRecorder(int(q.item()))
