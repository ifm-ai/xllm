import math
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

if TYPE_CHECKING:
    from xllm.metrics import MetricLogger


_ACTIVE_COLLECTOR: Optional["HealthMetricCollector"] = None
_LocalStats = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]

_MATRIX_PARAMETER_PATHS = (
    ("attention.wq.weight", "attention/q_proj"),
    ("attention.wk.weight", "attention/k_proj"),
    ("attention.wv.weight", "attention/v_proj"),
    ("attention.wo.weight", "attention/o_proj"),
    ("nffn.fc1.weight", "mlp/gate_proj"),
    ("nffn.fc3.weight", "mlp/up_proj"),
    ("nffn.fc2.weight", "mlp/down_proj"),
)


def _matrix_metric_path(layer_idx: int, fqn: str) -> Optional[str]:
    for suffix, matrix_path in _MATRIX_PARAMETER_PATHS:
        if fqn.endswith(suffix):
            return f"layer_{layer_idx:03d}/{matrix_path}"
    return None


def _matrix_norm_replica_count(model: Any) -> int:
    fsdp_size = int(model.world_size)
    data_parallel_size = int(model.data_parallel_world_size)
    if fsdp_size <= 0 or data_parallel_size % fsdp_size != 0:
        raise RuntimeError(
            "Invalid FSDP topology for matrix norms: "
            f"data_parallel_size={data_parallel_size}, fsdp_size={fsdp_size}"
        )
    replica_count = data_parallel_size // fsdp_size * int(model.context_parallel_world_size)
    if dist.is_available() and dist.is_initialized():
        model_parallel_size = int(model.model_parallel_world_size)
        expected_world_size = fsdp_size * replica_count * model_parallel_size
        if expected_world_size != dist.get_world_size():
            raise RuntimeError(
                "Matrix norm topology does not match the distributed world: "
                f"expected={expected_world_size}, actual={dist.get_world_size()}"
            )
    return replica_count


def _local_squared_l2(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(
        x.detach().reshape(-1), ord=2, dtype=torch.float32
    ).square()


def _dtensor_squared_l2_contribution(x: DTensor) -> torch.Tensor:
    replica_count = 1
    for mesh_dim, placement in enumerate(x.placements):
        if isinstance(placement, Replicate):
            replica_count *= x.device_mesh.size(mesh_dim)
        elif not isinstance(placement, Shard):
            raise RuntimeError(
                f"Unsupported DTensor placement for matrix norms: {placement}"
            )
    squared_l2 = _local_squared_l2(x.to_local())
    return squared_l2 if replica_count == 1 else squared_l2 / replica_count


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
        enable_query_key_stats: bool = False,
        enable_weight_norms: bool = False,
        enable_gradient_norms: bool = False,
    ):
        self.enabled = enabled
        self.freq = freq
        self.num_layers = num_layers
        self.enable_query_key_stats = enable_query_key_stats
        self.enable_weight_norms = enable_weight_norms
        self.enable_gradient_norms = enable_gradient_norms
        self.step: Optional[int] = None
        self.recording = False
        self.decoder_layer_activation_records: Dict[int, _LocalStats] = {}
        self.query_key_stats_records: Dict[int, Dict[str, _LocalStats]] = {}
        self.matrix_squared_l2_records: Dict[str, torch.Tensor] = {}

    def begin_step(self, step: int) -> None:
        global _ACTIVE_COLLECTOR
        self.step = step
        self.decoder_layer_activation_records = {}
        self.query_key_stats_records = {}
        self.matrix_squared_l2_records = {}
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

    @torch.no_grad()
    def record_query_key_stats(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor) -> None:
        if not self.recording or not self.enable_query_key_stats:
            return
        if layer_idx in self.query_key_stats_records:
            raise RuntimeError(f"Duplicate query/key statistics for layer {layer_idx}")
        if q.numel() == 0 or k.numel() == 0:
            raise RuntimeError(f"Empty query/key tensor for layer {layer_idx}")
        if q.ndim == 0 or k.ndim == 0:
            raise RuntimeError(f"Query/key tensors must have a head dimension for layer {layer_idx}")

        q_vector_l2 = torch.linalg.vector_norm(q.detach(), dim=-1, dtype=torch.float32)
        k_vector_l2 = torch.linalg.vector_norm(k.detach(), dim=-1, dtype=torch.float32)
        self.query_key_stats_records[layer_idx] = {
            "q": self._local_stats(q),
            "q_vector_l2": self._local_stats(q_vector_l2),
            "k": self._local_stats(k),
            "k_vector_l2": self._local_stats(k_vector_l2),
        }

    def _record_fsdp2_matrix_norms(
        self,
        model: Any,
        gradient_scale: float,
    ) -> None:
        layers = getattr(model, "layers", None)
        if layers is None or len(layers) != self.num_layers:
            raise RuntimeError("FSDP2 Transformer layers are unavailable for matrix norms")

        for layer_idx, layer in enumerate(layers):
            num_layer_matrices = 0
            for fqn, weight in layer.named_parameters():
                metric_path = _matrix_metric_path(layer_idx, fqn)
                if metric_path is None:
                    continue
                num_layer_matrices += 1
                if not isinstance(weight, DTensor):
                    raise RuntimeError(
                        f"FSDP2 parameter is not a DTensor: layer {layer_idx}, {fqn}"
                    )

                if self.enable_weight_norms:
                    metric_name = f"weight_norms/{metric_path}/l2"
                    if metric_name in self.matrix_squared_l2_records:
                        raise RuntimeError(f"Duplicate matrix norm metric: {metric_name}")
                    self.matrix_squared_l2_records[metric_name] = (
                        _dtensor_squared_l2_contribution(weight)
                    )
                if self.enable_gradient_norms:
                    gradient = weight.grad
                    if not isinstance(gradient, DTensor):
                        raise RuntimeError(
                            "FSDP2 gradient is unavailable or is not a DTensor: "
                            f"layer {layer_idx}, {fqn}"
                        )
                    metric_name = f"gradient_norms/{metric_path}/l2"
                    if metric_name in self.matrix_squared_l2_records:
                        raise RuntimeError(f"Duplicate matrix norm metric: {metric_name}")
                    squared_l2 = _dtensor_squared_l2_contribution(gradient)
                    self.matrix_squared_l2_records[metric_name] = (
                        squared_l2 if gradient_scale == 1 else squared_l2 / gradient_scale ** 2
                    )

            if num_layer_matrices != len(_MATRIX_PARAMETER_PATHS):
                raise RuntimeError(
                    "Expected seven matrix parameters for layer "
                    f"{layer_idx}, found {num_layer_matrices}"
                )

        num_enabled_norms = (
            int(self.enable_weight_norms) + int(self.enable_gradient_norms)
        )
        expected_records = (
            self.num_layers * len(_MATRIX_PARAMETER_PATHS) * num_enabled_norms
        )
        if len(self.matrix_squared_l2_records) != expected_records:
            raise RuntimeError(
                f"Expected {expected_records} matrix norm records, "
                f"found {len(self.matrix_squared_l2_records)}"
            )

    @torch.no_grad()
    def record_matrix_norms(self, model: Any, gradient_scale: float = 1.0) -> None:
        if not self.recording or not (
            self.enable_weight_norms or self.enable_gradient_norms
        ):
            return
        if self.enable_gradient_norms and (
            gradient_scale <= 0.0 or not math.isfinite(gradient_scale)
        ):
            raise ValueError(f"gradient_scale must be finite and positive, got {gradient_scale}")
        if self.matrix_squared_l2_records:
            raise RuntimeError("Duplicate matrix norm records")

        handles = getattr(model, "_all_handles", None)
        if handles is None:
            self._record_fsdp2_matrix_norms(model, gradient_scale)
            return

        replica_count = float(_matrix_norm_replica_count(model))
        gradient_squared_l2_divisor = replica_count * gradient_scale ** 2
        for handle in handles:
            module = getattr(handle, "_fully_sharded_module", None)
            layer_idx = getattr(module, "layer_id", None)
            if layer_idx is None:
                continue

            flat_param = handle.flat_param
            fqns = getattr(flat_param, "_fqns", None)
            shard_infos = getattr(flat_param, "_shard_param_infos", None)
            if fqns is None or shard_infos is None or len(fqns) != len(shard_infos):
                raise RuntimeError(
                    f"FSDP1 parameter shard metadata is unavailable for layer {layer_idx}"
                )
            weight_shard = getattr(flat_param, "_local_shard", None)
            if self.enable_weight_norms and weight_shard is None:
                raise RuntimeError(
                    f"FSDP1 local parameter shard is unavailable for layer {layer_idx}"
                )
            weight_shard = (
                None if weight_shard is None else weight_shard.detach().reshape(-1)
            )
            gradient_shard = flat_param.grad
            if self.enable_gradient_norms and gradient_shard is None:
                raise RuntimeError(
                    f"FSDP1 gradient shard is unavailable for layer {layer_idx}"
                )
            gradient_shard = (
                None if gradient_shard is None else gradient_shard.detach().reshape(-1)
            )

            def local_squared_l2(
                shard: torch.Tensor,
                shard_info: Any,
            ) -> torch.Tensor:
                if not shard_info.in_shard:
                    return shard.new_zeros((), dtype=torch.float32)
                start = int(shard_info.offset_in_shard)
                end = start + int(shard_info.numel_in_shard)
                if start < 0 or end > shard.numel():
                    raise RuntimeError(
                        f"FSDP1 shard metadata points outside layer {layer_idx}: "
                        f"[{start}, {end}) vs {shard.numel()}"
                    )
                return _local_squared_l2(shard[start:end])

            num_layer_matrices = 0
            for fqn, shard_info in zip(fqns, shard_infos):
                metric_path = _matrix_metric_path(int(layer_idx), fqn)
                if metric_path is None:
                    continue
                num_layer_matrices += 1

                if self.enable_weight_norms:
                    assert weight_shard is not None
                    metric_name = f"weight_norms/{metric_path}/l2"
                    if metric_name in self.matrix_squared_l2_records:
                        raise RuntimeError(f"Duplicate matrix norm metric: {metric_name}")
                    squared_l2 = local_squared_l2(weight_shard, shard_info)
                    self.matrix_squared_l2_records[metric_name] = (
                        squared_l2 if replica_count == 1 else squared_l2 / replica_count
                    )
                if self.enable_gradient_norms:
                    assert gradient_shard is not None
                    metric_name = f"gradient_norms/{metric_path}/l2"
                    if metric_name in self.matrix_squared_l2_records:
                        raise RuntimeError(f"Duplicate matrix norm metric: {metric_name}")
                    squared_l2 = local_squared_l2(gradient_shard, shard_info)
                    self.matrix_squared_l2_records[metric_name] = (
                        squared_l2 if gradient_squared_l2_divisor == 1
                        else squared_l2 / gradient_squared_l2_divisor
                    )

            if num_layer_matrices != len(_MATRIX_PARAMETER_PATHS):
                raise RuntimeError(
                    "Expected seven matrix parameters for layer "
                    f"{layer_idx}, found {num_layer_matrices}"
                )

        num_enabled_norms = (
            int(self.enable_weight_norms) + int(self.enable_gradient_norms)
        )
        expected_records = (
            self.num_layers * len(_MATRIX_PARAMETER_PATHS) * num_enabled_norms
        )
        if len(self.matrix_squared_l2_records) != expected_records:
            raise RuntimeError(
                f"Expected {expected_records} matrix norm records, "
                f"found {len(self.matrix_squared_l2_records)}"
            )

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
        if self.enable_query_key_stats and sorted(self.query_key_stats_records) != layers:
            raise RuntimeError(
                f"Expected {self.num_layers} query/key records at step {self.step}, "
                f"got layers {sorted(self.query_key_stats_records)} at step {step}"
            )
        num_enabled_norms = (
            int(self.enable_weight_norms) + int(self.enable_gradient_norms)
        )
        expected_norms = (
            self.num_layers * len(_MATRIX_PARAMETER_PATHS) * num_enabled_norms
        )
        if (
            num_enabled_norms
            and len(self.matrix_squared_l2_records) != expected_norms
        ):
            raise RuntimeError(
                f"Expected {len(_MATRIX_PARAMETER_PATHS) * num_enabled_norms} matrix norm "
                f"records per layer at step {self.step}, "
                f"got {len(self.matrix_squared_l2_records)} records at step {step}"
            )

        entries = []
        for layer_idx in layers:
            layer = f"layer_{layer_idx:03d}"
            entries.append((
                f"activation_stats/{layer}",
                ("mean", "min", "max"),
                self.decoder_layer_activation_records[layer_idx],
            ))
            if self.enable_query_key_stats:
                for name in ("q", "k"):
                    prefix = f"{name}_stats/{layer}"
                    entries.append((
                        prefix,
                        ("mean", "min", "max"),
                        self.query_key_stats_records[layer_idx][name],
                    ))
                    entries.append((
                        prefix,
                        ("vector_l2_mean", "vector_l2_min", "vector_l2_max"),
                        self.query_key_stats_records[layer_idx][f"{name}_vector_l2"],
                    ))

        sums = torch.stack([stats[0] for _, _, stats in entries]).double()
        counts = torch.tensor(
            [stats[3] for _, _, stats in entries],
            dtype=torch.float64,
            device=sums.device,
        )
        matrix_metric_names = sorted(self.matrix_squared_l2_records)
        matrix_squared_l2_start = sums.numel()
        if matrix_metric_names:
            matrix_squared_l2_values = torch.stack([
                self.matrix_squared_l2_records[name] for name in matrix_metric_names
            ]).double()
            sums = torch.cat((sums, matrix_squared_l2_values))
        num_sum_values = sums.numel()
        sum_packet = torch.cat((sums, counts))
        mins = _pack_extrema([stats[1] for _, _, stats in entries])
        maxs = _pack_extrema([stats[2] for _, _, stats in entries])
        self.decoder_layer_activation_records = {}
        self.query_key_stats_records = {}
        self.matrix_squared_l2_records = {}

        if dist.is_available() and dist.is_initialized():
            dist.reduce(sum_packet, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(mins, dst=0, op=dist.ReduceOp.MIN)
            dist.reduce(maxs, dst=0, op=dist.ReduceOp.MAX)
            if dist.get_rank() != 0:
                return

        reduced_sums = sum_packet[:num_sum_values]
        reduced_counts = sum_packet[num_sum_values:]
        num_stat_entries = len(entries)
        means = reduced_sums[:num_stat_entries] / reduced_counts[:num_stat_entries]
        values = torch.stack((means.float(), mins, maxs), dim=1).cpu().tolist()
        metrics = {}
        for (prefix, names, _), stats in zip(entries, values):
            metrics.update({
                f"{prefix}/{name}": value for name, value in zip(names, stats)
            })
        if matrix_metric_names:
            matrix_norm_values = (
                reduced_sums[matrix_squared_l2_start:]
                .clamp_min(0)
                .sqrt()
                .cpu()
                .tolist()
            )
            metrics.update(dict(zip(matrix_metric_names, matrix_norm_values)))
        mlogger.log(metrics, step)
