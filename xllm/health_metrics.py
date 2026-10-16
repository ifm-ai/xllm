import math
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

if TYPE_CHECKING:
    from xllm.metrics import MetricLogger

logger = getLogger()

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

_GEKKO_MATRIX_PARAMETER_PATHS = (
    ("gda.wq.weight", "attention/q_proj"),
    ("gda.wk.weight", "attention/k_proj"),
    ("gda.wv.weight", "attention/v_proj"),
    ("gda.wo.weight", "attention/o_proj"),
    ("gda.wr.weight", "attention/output_gate"),
    ("gda.q_conv.weight", "attention/q_conv"),
    ("gda.k_conv.weight", "attention/k_conv"),
    ("gda.v_conv.weight", "attention/v_conv"),
    ("gda.query_norm.weight", "attention/query_norm"),
    ("gda.key_norm.weight", "attention/key_norm"),
    ("gda.timenorm.weight", "attention/time_norm/weight"),
    ("gda.timenorm.bias", "attention/time_norm/bias"),
    ("nffn.fc1.weight", "mlp/gate_proj"),
    ("nffn.fc3.weight", "mlp/up_proj"),
    ("nffn.fc2.weight", "mlp/down_proj"),
)

# Logged only when the layer has them (bias terms and an affine MLP norm).
_GEKKO_OPTIONAL_MATRIX_PARAMETER_PATHS = (
    ("gda.wv.bias", "attention/v_proj/bias"),
    ("gda.wr.bias", "attention/output_gate/bias"),
    ("nffn.norm.weight", "mlp/norm/weight"),
    ("nffn.norm.bias", "mlp/norm/bias"),
)

_MatrixParameterPaths = Tuple[Tuple[str, str], ...]


def _matrix_parameter_paths(layer: Any, fqns: list[str]) -> _MatrixParameterPaths:
    """Resolve the norm parameter set of one decoder layer from its parameter names."""
    if not hasattr(layer, "gda"):
        return _MATRIX_PARAMETER_PATHS
    return _GEKKO_MATRIX_PARAMETER_PATHS + tuple(
        (suffix, matrix_path)
        for suffix, matrix_path in _GEKKO_OPTIONAL_MATRIX_PARAMETER_PATHS
        if any(fqn.endswith(suffix) for fqn in fqns)
    )


def _matrix_metric_path(
    layer_idx: int, fqn: str, paths: _MatrixParameterPaths,
) -> Optional[str]:
    for suffix, matrix_path in paths:
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
        enable_logits_stats: bool = False,
        enable_gekko_stats: bool = False,
    ):
        self.enabled = enabled
        self.freq = freq
        self.num_layers = num_layers
        self.enable_query_key_stats = enable_query_key_stats or enable_gekko_stats
        self.enable_weight_norms = enable_weight_norms
        self.enable_gradient_norms = enable_gradient_norms
        self.enable_logits_stats = enable_logits_stats
        self.enable_gekko_stats = enable_gekko_stats
        self.step: Optional[int] = None
        self.recording = False
        self.decoder_layer_activation_records: Dict[int, _LocalStats] = {}
        self.query_key_stats_records: Dict[int, Dict[str, _LocalStats]] = {}
        self.gekko_stats_records: Dict[int, Dict[str, _LocalStats]] = {}
        self.matrix_squared_l2_records: Dict[str, torch.Tensor] = {}
        self.expected_matrix_norm_records: Optional[int] = None
        self.logits_stats_record: Optional[Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        ]] = None

    def begin_step(self, step: int) -> None:
        global _ACTIVE_COLLECTOR
        self.step = step
        self.decoder_layer_activation_records = {}
        self.query_key_stats_records = {}
        self.gekko_stats_records = {}
        self.matrix_squared_l2_records = {}
        self.expected_matrix_norm_records = None
        self.logits_stats_record = None
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

    @torch.no_grad()
    def record_gekko_query_key_stats(
        self, layer_idx: int, query: torch.Tensor, key: torch.Tensor,
        sca_query: torch.Tensor, sca_key_l2: Optional[torch.Tensor],
    ) -> None:
        """Record Q/K; key vector norms are captured before in-place partial RoPE."""
        self.record_query_key_stats(layer_idx, query, key)
        if not self.recording or not self.enable_gekko_stats:
            return
        if layer_idx in self.gekko_stats_records:
            raise RuntimeError(f"Duplicate Gekko queries for layer {layer_idx}")
        assert sca_key_l2 is not None
        sca_query_l2 = torch.linalg.vector_norm(sca_query.detach(), dim=-1, dtype=torch.float32)
        layer_stats: Dict[str, _LocalStats] = {
            "sca_query_l2": self._local_stats(sca_query_l2),
            "sca_key_l2": self._local_stats(sca_key_l2),
        }
        # Some CP ranks have no update (for example, a single cold-start chunk).
        zero = query.new_zeros((), dtype=torch.float32)
        layer_stats["awk_update_ratio"] = (zero, zero + float("inf"), zero - float("inf"), 0)
        self.gekko_stats_records[layer_idx] = layer_stats

    @torch.no_grad()
    def record_gekko_update_ratio(self, layer_idx: int, ratio: torch.Tensor) -> None:
        if not self.recording or not self.enable_gekko_stats:
            return
        layer_stats = self.gekko_stats_records[layer_idx]
        total, minimum, maximum, count = layer_stats["awk_update_ratio"]
        update_sum, update_min, update_max, update_count = self._local_stats(ratio)
        layer_stats["awk_update_ratio"] = (
            total + update_sum, torch.minimum(minimum, update_min),
            torch.maximum(maximum, update_max), count + update_count,
        )

    @torch.no_grad()
    def record_gekko_branches(
        self, layer_idx: int, sca: torch.Tensor, awk: torch.Tensor,
        output_gate: torch.Tensor, unavailable_history_mask: torch.Tensor,
    ) -> None:
        if not self.recording or not self.enable_gekko_stats:
            return
        layer_stats = self.gekko_stats_records[layer_idx]
        if "output_gate" in layer_stats:
            raise RuntimeError(f"Duplicate Gekko branches for layer {layer_idx}")
        # Reduce squared coordinates globally before deriving branch RMS/energy share.
        for name, branch in (("sca_squared", sca), ("awk_squared", awk)):
            squared_l2 = _local_squared_l2(branch)
            # Only sum/count are used; avoid a full FP32 activation copy for RMS.
            layer_stats[name] = (squared_l2, squared_l2, squared_l2, branch.numel())
        layer_stats["output_gate"] = self._local_stats(output_gate)
        for name, mask in (
            ("gate_near_zero", output_gate.detach().abs().lt(1e-3)),
            ("history_available", ~unavailable_history_mask),
        ):
            # Only sum/count are consumed for fractions, not coordinate extrema.
            count = mask.sum(dtype=torch.float32)
            layer_stats[name] = (count, count, count, mask.numel())

    def _record_fsdp2_matrix_norms(
        self,
        model: Any,
        gradient_scale: float,
    ) -> None:
        layers = getattr(model, "layers", None)
        if layers is None or len(layers) != self.num_layers:
            raise RuntimeError("FSDP2 decoder layers are unavailable for matrix norms")

        num_matrices = 0
        for layer_idx, layer in enumerate(layers):
            named_parameters = list(layer.named_parameters())
            paths = _matrix_parameter_paths(layer, [fqn for fqn, _ in named_parameters])
            num_matrices += len(paths)
            num_layer_matrices = 0
            for fqn, weight in named_parameters:
                metric_path = _matrix_metric_path(layer_idx, fqn, paths)
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

            if num_layer_matrices != len(paths):
                raise RuntimeError(
                    f"Expected {len(paths)} matrix parameters for layer "
                    f"{layer_idx}, found {num_layer_matrices}"
                )

        self._check_matrix_norm_records(num_matrices)

    def _check_matrix_norm_records(self, num_matrices: int) -> None:
        num_enabled_norms = (
            int(self.enable_weight_norms) + int(self.enable_gradient_norms)
        )
        expected_records = num_matrices * num_enabled_norms
        if len(self.matrix_squared_l2_records) != expected_records:
            raise RuntimeError(
                f"Expected {expected_records} matrix norm records, "
                f"found {len(self.matrix_squared_l2_records)}"
            )
        self.expected_matrix_norm_records = expected_records

    @torch.no_grad()
    def record_logits_stats(
        self,
        logits_sum: torch.Tensor,
        logits_count: torch.Tensor,
        logits_minimum: torch.Tensor,
        logits_maximum: torch.Tensor,
    ) -> None:
        if not self.recording or not self.enable_logits_stats:
            return
        if self.logits_stats_record is not None:
            raise RuntimeError("Duplicate logit statistics")
        if any(value.numel() != 1 for value in (
            logits_sum, logits_count, logits_minimum, logits_maximum,
        )):
            raise RuntimeError("Logit statistics must be scalars")
        self.logits_stats_record = (
            logits_sum.detach().float(),
            logits_count.detach().double(),
            logits_minimum.detach().float(),
            logits_maximum.detach().float(),
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
        num_matrices = 0
        seen_layers = set()
        for handle in handles:
            module = getattr(handle, "_fully_sharded_module", None)
            layer_idx = getattr(module, "layer_id", None)
            if layer_idx is None:
                continue
            layer_idx = int(layer_idx)
            if layer_idx in seen_layers:
                raise RuntimeError(f"Duplicate FSDP1 decoder layer {layer_idx}")
            seen_layers.add(layer_idx)

            flat_param = handle.flat_param
            fqns = getattr(flat_param, "_fqns", None)
            shard_infos = getattr(flat_param, "_shard_param_infos", None)
            if fqns is None or shard_infos is None or len(fqns) != len(shard_infos):
                raise RuntimeError(
                    f"FSDP1 parameter shard metadata is unavailable for layer {layer_idx}"
                )
            paths = _matrix_parameter_paths(module, list(fqns))
            num_matrices += len(paths)
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
                metric_path = _matrix_metric_path(layer_idx, fqn, paths)
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

            if num_layer_matrices != len(paths):
                raise RuntimeError(
                    f"Expected {len(paths)} matrix parameters for layer "
                    f"{layer_idx}, found {num_layer_matrices}"
                )

        if seen_layers != set(range(self.num_layers)):
            raise RuntimeError(
                f"Expected FSDP1 decoder layers 0..{self.num_layers - 1}, "
                f"found {sorted(seen_layers)}"
            )
        self._check_matrix_norm_records(num_matrices)

    def _clear_records(self) -> None:
        self.decoder_layer_activation_records = {}
        self.query_key_stats_records = {}
        self.gekko_stats_records = {}
        self.matrix_squared_l2_records = {}
        self.expected_matrix_norm_records = None
        self.logits_stats_record = None

    def flush(self, mlogger: "MetricLogger", step: int) -> None:
        if not self.recording:
            return

        self.recording = False
        layers = list(range(self.num_layers))
        problems = []
        if step != self.step or sorted(self.decoder_layer_activation_records) != layers:
            problems.append(
                f"Expected {self.num_layers} activation records at step {self.step}, "
                f"got layers {sorted(self.decoder_layer_activation_records)} at step {step}"
            )
        if self.enable_query_key_stats and sorted(self.query_key_stats_records) != layers:
            problems.append(
                f"Expected {self.num_layers} query/key records at step {self.step}, "
                f"got layers {sorted(self.query_key_stats_records)} at step {step}"
            )
        if self.enable_logits_stats and self.logits_stats_record is None:
            problems.append(f"Missing logit statistics at step {step}")
        if (self.enable_weight_norms or self.enable_gradient_norms) and (
            self.expected_matrix_norm_records is None
            or len(self.matrix_squared_l2_records) != self.expected_matrix_norm_records
        ):
            problems.append(
                f"Expected {self.expected_matrix_norm_records} matrix norm records "
                f"at step {self.step}, "
                f"got {len(self.matrix_squared_l2_records)} records at step {step}"
            )
        if self.enable_gekko_stats:
            if sorted(self.gekko_stats_records) != layers:
                problems.append(f"Missing Gekko statistics at step {step}")
            else:
                problems.extend(
                    f"Missing Gekko branch statistics for layer {layer_idx}"
                    for layer_idx in layers
                    if "output_gate" not in self.gekko_stats_records[layer_idx]
                )

        # Every rank must skip together, or the reductions below would hang.
        skip_step = bool(problems)
        if dist.is_available() and dist.is_initialized():
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device("cpu")
            )
            invalid = torch.tensor([float(skip_step)], device=device)
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
            skip_step = invalid.item() > 0
        if skip_step:
            for problem in problems:
                logger.warning(f"Skipping health metrics: {problem}")
            if not problems:
                logger.warning(
                    f"Skipping health metrics at step {step}: another rank reported a problem"
                )
            self._clear_records()
            return

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

        if self.enable_gekko_stats:
            for layer_idx in layers:
                layer_stats = self.gekko_stats_records[layer_idx]
                for name, stats in layer_stats.items():
                    entries.append((
                        f"gekko_stats/layer_{layer_idx:03d}/{name}",
                        ("mean", "min", "max"), stats,
                    ))

        sums = torch.stack([stats[0] for _, _, stats in entries]).double()
        counts = torch.tensor(
            [stats[3] for _, _, stats in entries],
            dtype=torch.float64,
            device=sums.device,
        )
        logits_index = None
        logits_extrema = None
        if self.logits_stats_record is not None:
            logits_index = len(entries)
            logits_sum, logits_count, logits_minimum, logits_maximum = (
                self.logits_stats_record
            )
            sums = torch.cat((sums, logits_sum.reshape(1).double()))
            counts = torch.cat((counts, logits_count.reshape(1)))
            logits_extrema = logits_minimum, logits_maximum
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
        if logits_extrema is not None:
            logits_minimum, logits_maximum = logits_extrema
            mins = torch.cat((mins, logits_minimum.reshape(1)))
            maxs = torch.cat((maxs, logits_maximum.reshape(1)))
        self._clear_records()

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
        values = torch.stack((
            means.float(), mins[:num_stat_entries], maxs[:num_stat_entries],
        ), dim=1).cpu().tolist()
        metrics = {}
        for (prefix, names, _), stats in zip(entries, values):
            metrics.update({
                f"{prefix}/{name}": value for name, value in zip(names, stats)
            })
        if self.enable_gekko_stats:
            entry_counts = dict(zip(
                (prefix for prefix, _, _ in entries),
                reduced_counts[:num_stat_entries].cpu().tolist(),
            ))
            for layer_idx in layers:
                prefix = f"gekko_stats/layer_{layer_idx:03d}"
                sca_squared = metrics[f"{prefix}/sca_squared/mean"]
                awk_squared = metrics[f"{prefix}/awk_squared/mean"]
                metrics[f"{prefix}/sca_rms"] = math.sqrt(sca_squared)
                metrics[f"{prefix}/awk_rms"] = math.sqrt(awk_squared)
                total_energy = sca_squared + awk_squared
                metrics[f"{prefix}/awk_energy_fraction"] = (
                    awk_squared / total_energy if total_energy > 0 else float("nan")
                )
                for name in ("sca_squared", "awk_squared", "gate_near_zero", "history_available"):
                    for stat in ("min", "max"):
                        del metrics[f"{prefix}/{name}/{stat}"]
                del metrics[f"{prefix}/sca_squared/mean"], metrics[f"{prefix}/awk_squared/mean"]
                for name, metric_name in (
                    ("gate_near_zero", "output_gate/near_zero_fraction"),
                    ("history_available", "awk_history_available_fraction"),
                ):
                    metrics[f"{prefix}/{metric_name}"] = metrics.pop(f"{prefix}/{name}/mean")
                ratio_count = entry_counts[f"{prefix}/awk_update_ratio"]
                metrics[f"{prefix}/awk_update_ratio/count"] = ratio_count
                if ratio_count == 0:
                    for stat in ("mean", "min", "max"):
                        metrics[f"{prefix}/awk_update_ratio/{stat}"] = float("nan")

        if logits_index is not None:
            logit_sum, logit_count, logit_minimum, logit_maximum = torch.stack((
                reduced_sums[logits_index], reduced_counts[logits_index],
                mins[logits_index], maxs[logits_index],
            )).cpu().tolist()
            if logit_count <= 0:
                logger.warning(f"No valid logits at step {step}; logging NaN logit statistics")
                logit_sum = logit_minimum = logit_maximum = float("nan")
                logit_count = 1.0
            metrics["logits_stats/mean"] = logit_sum / logit_count
            metrics["logits_stats/min"] = logit_minimum
            metrics["logits_stats/max"] = logit_maximum
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
