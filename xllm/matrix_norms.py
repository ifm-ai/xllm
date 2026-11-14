"""Logical matrix slices inside xLLM's packed expert and value parameters."""

from typing import NamedTuple, Optional

import torch
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


class MatrixSlice(NamedTuple):
    path: str
    dim: Optional[int] = None
    start: int = 0
    stop: int = 0


def matrix_slices(layer, layer_idx, fqn, shape, dense_path):
    prefix = f"layer_{layer_idx:03d}"
    if dense_path is not None:
        return [MatrixSlice(dense_path)]
    if fqn in ("attention.wr.weight", "mova.wr.weight"):
        return [MatrixSlice(f"{prefix}/attention/output_gate")]
    if fqn in ("moe.router.weight", "mova.router.weight"):
        family = fqn.split(".")[0]
        return [MatrixSlice(f"{prefix}/{family}/router")]
    if fqn in ("mova.wq.weight", "mova.wk.weight", "mova.wo.weight"):
        projection = {"wq": "q_proj", "wk": "k_proj", "wo": "o_proj"}[fqn.split(".")[1]]
        return [MatrixSlice(f"{prefix}/attention/{projection}")]
    packed = fqn == "mova.wv.weight" or fqn.startswith("moe.experts.weight") or fqn in (
        "moe.fc1.weight", "moe.fc2.weight", "moe.fc3.weight",
    )
    if packed and (shape is None or len(shape) != 2):
        raise RuntimeError(f"Missing 2D parameter shape for {prefix}/{fqn}: {shape}")
    if fqn == "mova.wv.weight":
        count = layer.mova.n_values
        return _equal_slices(prefix, "mova/values", "v_proj", count, shape, 0)
    if fqn.startswith("moe.experts.weight"):
        projection = {"weight1": "gate_proj", "weight2": "down_proj", "weight3": "up_proj"}.get(fqn.split(".")[-1])
        if projection is None:
            return []
        moe = layer.moe
        rows = _slice_width(shape[0], moe.n_local_experts)
        slices = []
        for expert_idx in range(moe.n_experts):
            local_idx = expert_idx - moe.expert_start_idx
            start = local_idx * rows if 0 <= local_idx < moe.n_local_experts else 0
            stop = start + rows if 0 <= local_idx < moe.n_local_experts else 0
            slices.append(MatrixSlice(
                f"{prefix}/moe/experts/expert_{expert_idx:03d}/{projection}", 0, start, stop
            ))
        return slices
    if fqn in ("moe.fc1.weight", "moe.fc2.weight", "moe.fc3.weight"):
        projection = {"fc1": "gate_proj", "fc2": "down_proj", "fc3": "up_proj"}[fqn.split(".")[1]]
        return _equal_slices(
            prefix, "moe/shared_experts", projection, layer.moe.n_shared_experts,
            shape, 1 if projection == "down_proj" else 0,
        )
    return []


def _slice_width(size, count):
    if count <= 0 or size % count:
        raise RuntimeError(f"Cannot split dimension {size} into {count} matrices")
    return size // count


def _equal_slices(prefix, family, projection, count, shape, dim):
    width = _slice_width(shape[dim], count)
    name = "value" if family == "mova/values" else "expert"
    return [MatrixSlice(
        f"{prefix}/{family}/{name}_{index:03d}/{projection}",
        dim, index * width, (index + 1) * width,
    ) for index in range(count)]


def matrix_counts(layer, dense_count):
    """Expected packed parameters and logical matrices, including optional gates."""
    attention = getattr(layer, "mova", getattr(layer, "attention", None))
    gate = int(getattr(attention, "wr", None) is not None)
    moe = getattr(layer, "moe", None)
    if moe is None:
        return dense_count + gate, dense_count + gate
    values = getattr(getattr(layer, "mova", None), "n_values", 0)
    attention_parameters = 5 if values else 4
    shared = moe.n_shared_experts
    parameters = attention_parameters + gate + 4 + (3 if shared else 0)
    attention_matrices = 4 + values if values else 4
    matrices = attention_matrices + gate + 1 + 3 * (moe.n_experts + shared)
    return parameters, matrices


def squared_l2(x):
    return torch.linalg.vector_norm(x.detach().reshape(-1), dtype=torch.float32).square()


def flat_slice_squared_l2(local_values, param_offset, shape, selection):
    """Intersect a contiguous FSDP1 parameter fragment with a logical matrix."""
    if selection.dim is None:
        return squared_l2(local_values)
    if len(shape) != 2:
        raise RuntimeError(f"Packed matrix must be 2D, got {shape}")
    rows, columns = shape
    if param_offset < 0 or param_offset + local_values.numel() > rows * columns:
        raise RuntimeError("FSDP1 fragment is outside its original parameter")
    if selection.dim == 0:
        start = max(selection.start * columns - param_offset, 0)
        stop = min(selection.stop * columns - param_offset, local_values.numel())
        return squared_l2(local_values[start:max(start, stop)])

    # A column slice has full interior rows and at most two partial boundary rows.
    end = param_offset + local_values.numel()
    first_full = (param_offset + columns - 1) // columns
    last_full = end // columns
    result = local_values.new_zeros((), dtype=torch.float32)
    if last_full > first_full:
        middle = local_values[first_full * columns - param_offset:last_full * columns - param_offset]
        result = squared_l2(middle.view(-1, columns)[:, selection.start:selection.stop])
    for row in {param_offset // columns, (end - 1) // columns}:
        if first_full <= row < last_full or end == param_offset:
            continue
        start = max(row * columns + selection.start, param_offset)
        stop = min(row * columns + selection.stop, end)
        if stop > start:
            result = result + squared_l2(local_values[start - param_offset:stop - param_offset])
    return result


def dtensor_slice(x, selection):
    local = x.to_local()
    if selection.dim is None:
        return local
    local_shape, offsets = compute_local_shape_and_global_offset(x.shape, x.device_mesh, x.placements)
    if tuple(local.shape) != local_shape:
        raise RuntimeError("DTensor local shape does not match its shard metadata")
    dim = selection.dim
    start = min(max(selection.start - offsets[dim], 0), local.shape[dim])
    stop = min(max(selection.stop - offsets[dim], 0), local.shape[dim])
    return local.narrow(dim, start, max(stop - start, 0))


def _partition_offsets(local, selections, offsets):
    dim = selections[0].dim
    boundaries = [0]
    covered = 0
    for selection in selections:
        if selection.dim != dim:
            raise RuntimeError("Packed matrix slices must share a dimension")
        start = min(max(selection.start - offsets[dim], 0), local.shape[dim])
        stop = min(max(selection.stop - offsets[dim], 0), local.shape[dim])
        if stop < start or (stop > start and start != covered):
            raise RuntimeError("Packed matrix slices must form a disjoint partition")
        boundaries.append(boundaries[-1] + stop - start)
        if stop > start:
            covered = stop
    if covered != local.shape[dim]:
        raise RuntimeError("Packed matrix slices do not cover the local shard")
    return torch.tensor(boundaries, device=local.device, dtype=torch.int64)


def partitioned_squared_l2(local, selections, offsets, partitions=None):
    """Reduce a packed local matrix once, including partial and unowned slices."""
    dim = selections[0].dim
    key = (local.device, tuple(local.shape), offsets[dim],
           tuple((s.dim, s.start, s.stop) for s in selections))
    boundaries = None if partitions is None else partitions.get(key)
    if boundaries is None:
        boundaries = _partition_offsets(local, selections, offsets)
        if partitions is not None:
            partitions[key] = boundaries
    if local.numel() == 0:
        return local.new_zeros(len(selections), dtype=torch.float32)

    axis_squares = torch.linalg.vector_norm(local.detach(), dim=1 - dim, dtype=torch.float32).square()
    # The host-side partition check makes GPU validation/synchronization unnecessary.
    return torch.segment_reduce(axis_squares, "sum", offsets=boundaries, unsafe=True)


def dtensor_squared_l2(x, selections, partitions):
    local = x.to_local()
    if len(selections) == 1 and selections[0].dim is None:
        return squared_l2(local).reshape(1)
    local_shape, offsets = compute_local_shape_and_global_offset(x.shape, x.device_mesh, x.placements)
    if tuple(local.shape) != local_shape:
        raise RuntimeError("DTensor local shape does not match its shard metadata")
    return partitioned_squared_l2(local, selections, offsets, partitions)
