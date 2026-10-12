from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from xllm.metrics import MetricLogger


_ACTIVE_COLLECTOR: Optional["HealthMetricCollector"] = None
_LocalStats = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


def get_active_health_collector() -> Optional["HealthMetricCollector"]:
    return _ACTIVE_COLLECTOR


def _pack_extrema(values: list[torch.Tensor]) -> torch.Tensor:
    by_dtype = {}
    for index, value in enumerate(values):
        by_dtype.setdefault(value.dtype, []).append(index)
    if len(by_dtype) == 1:
        return torch.stack(values).float()

    # Mixed-dtype CUDA stacks can fall back to individual scalar copies.
    packed = values[0].new_empty(len(values), dtype=torch.float32)
    for indices in by_dtype.values():
        group = torch.stack([values[index] for index in indices]).float()
        positions = torch.tensor(indices, device=packed.device)
        packed.index_copy_(0, positions, group)
    return packed


class HealthMetricCollector:
    """Collect health summaries on sampled training steps."""

    def __init__(
        self,
        enabled: bool,
        freq: int,
        num_layers: int,
    ):
        self.enabled = enabled
        self.freq = freq
        self.num_layers = num_layers
        self.step: Optional[int] = None
        self.recording = False
        self.decoder_layer_activation_records: Dict[int, _LocalStats] = {}

    def begin_step(self, step: int) -> None:
        global _ACTIVE_COLLECTOR
        self.step = step
        self.decoder_layer_activation_records = {}
        self.recording = self.enabled and step % self.freq == 0
        _ACTIVE_COLLECTOR = self if self.recording else None

    def end_forward(self) -> None:
        global _ACTIVE_COLLECTOR
        _ACTIVE_COLLECTOR = None

    @staticmethod
    def _local_stats(x: torch.Tensor) -> _LocalStats:
        x = x.detach()
        min_, max_ = torch.aminmax(x)
        return x.sum(dtype=torch.float32), min_, max_, x.numel()

    @torch.no_grad()
    def record_activation(self, layer_idx: int, x: torch.Tensor) -> None:
        if not self.recording:
            return
        if layer_idx in self.decoder_layer_activation_records:
            raise RuntimeError(f"Duplicate activation statistics for layer {layer_idx}")
        if x.numel() == 0:
            raise RuntimeError(f"Empty activation tensor for layer {layer_idx}")

        self.decoder_layer_activation_records[layer_idx] = self._local_stats(x)

    def flush(self, mlogger: "MetricLogger", step: int) -> None:
        if not self.recording:
            return

        self.recording = False
        layers = list(range(self.num_layers))
        if step != self.step or sorted(self.decoder_layer_activation_records) != layers:
            raise RuntimeError(
                f"Expected {self.num_layers} activation records at step {self.step}, "
                f"got layers {sorted(self.decoder_layer_activation_records)} at step {step}"
            )

        entries = []
        for layer_idx in layers:
            layer = f"layer_{layer_idx:03d}"
            entries.append((
                f"activation_stats/{layer}",
                ("mean", "min", "max"),
                self.decoder_layer_activation_records[layer_idx],
            ))

        sums = torch.stack([stats[0] for _, _, stats in entries]).double()
        counts = torch.tensor(
            [stats[3] for _, _, stats in entries],
            dtype=torch.float64,
            device=sums.device,
        )
        sum_count = torch.stack((sums, counts))
        mins = _pack_extrema([stats[1] for _, _, stats in entries])
        maxs = _pack_extrema([stats[2] for _, _, stats in entries])
        self.decoder_layer_activation_records = {}

        if dist.is_available() and dist.is_initialized():
            dist.reduce(sum_count, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(mins, dst=0, op=dist.ReduceOp.MIN)
            dist.reduce(maxs, dst=0, op=dist.ReduceOp.MAX)
            if dist.get_rank() != 0:
                return

        num_stat_entries = len(entries)
        means = sum_count[0, :num_stat_entries] / sum_count[1, :num_stat_entries]
        values = torch.stack((means.float(), mins, maxs), dim=1).cpu().tolist()
        metrics = {}
        for (prefix, names, _), stats in zip(entries, values):
            metrics.update({
                f"{prefix}/{name}": value for name, value in zip(names, stats)
            })
        mlogger.log(metrics, step)
