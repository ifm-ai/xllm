"""Weight and gradient norms of MoE / MoVA layers: one norm per routed expert, shared expert and value.

Ported from xllm-lab #262. The first tests check the packed-slice reductions directly; the synthetic
FSDP1 test checks expert ownership across tensor-parallel ranks; the real-model tests check every logged
norm of tiny MoE and MoVA Transformers, sharded with FSDP1 and FSDP2, against the unsharded matrices.
"""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import tests.health.helpers  # noqa: F401  (installs the CPU stubs before any xllm import)
import xllm.distributed.initialize as parallel_state
from tests.health.helpers import TOKENS, logged_metrics, shard, tiny_transformer
from xllm.health_metrics import _MATRIX_PARAMETER_PATHS, HealthMetricCollector, _matrix_metric_path
from xllm.matrix_norms import MatrixSlice, flat_slice_squared_l2, matrix_slices, partitioned_squared_l2


# --------------------------------------------------------------------------------------------------
# Packed-slice reductions


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_norms_match_individual_slices_for_every_shard_boundary(dim, dtype):
    matrix = (torch.arange(35).reshape(5, 7) / 11).to(dtype)
    width = matrix.shape[dim]
    selections = [MatrixSlice(str(i), dim, i, i + 1) for i in range(width)]
    # Empty global expert IDs can precede/follow this TP rank's owned experts.
    selections = [MatrixSlice("unowned_before", dim, 0, 0), *selections,
                  MatrixSlice("unowned_after", dim, 0, 0)]
    for start in range(width + 1):
        for stop in range(start, width + 1):
            local = matrix.narrow(dim, start, stop - start)
            offsets = (start, 0) if dim == 0 else (0, start)
            actual = partitioned_squared_l2(local, selections, offsets)
            expected = []
            for selection in selections:
                left = min(max(selection.start - start, 0), local.shape[dim])
                right = min(max(selection.stop - start, 0), local.shape[dim])
                values = local.narrow(dim, left, right - left)
                expected.append(values.float().square().sum())
            torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)


def test_packed_norms_reject_overlapping_or_incomplete_partitions():
    matrix = torch.ones(4, 3)
    for bounds in ([(0, 3), (2, 4)], [(0, 1), (2, 4)], [(0, 3)]):
        selections = [MatrixSlice(str(i), 0, start, stop) for i, (start, stop) in enumerate(bounds)]
        with pytest.raises(RuntimeError, match="partition|cover"):
            partitioned_squared_l2(matrix, selections, (0, 0))


def test_partition_cache_tracks_layout_but_not_tensor_values():
    partitions = {}
    selections = [MatrixSlice("a", 0, 0, 3), MatrixSlice("b", 0, 3, 6)]
    first = partitioned_squared_l2(torch.ones(3, 4), selections, (0, 0), partitions)
    changed = partitioned_squared_l2(torch.full((3, 4), 2.), selections, (0, 0), partitions)
    shifted = partitioned_squared_l2(torch.ones(3, 4), selections, (3, 0), partitions)
    torch.testing.assert_close(first, torch.tensor([12., 0.]))
    torch.testing.assert_close(changed, first * 4)
    torch.testing.assert_close(shifted, torch.tensor([0., 12.]))
    assert len(partitions) == 2


@pytest.mark.parametrize("dim", [0, 1])
def test_flat_fragments_intersect_rows_and_columns(dim):
    matrix = torch.arange(35, dtype=torch.float32).view(5, 7)
    selection = MatrixSlice("test", dim, 1, 3)
    expected = matrix[1:3] if dim == 0 else matrix[:, 1:3]
    # Every possible split includes cuts inside rows and empty end shards.
    for split in range(matrix.numel() + 1):
        flat = matrix.flatten()
        actual = flat_slice_squared_l2(flat[:split], 0, matrix.shape, selection)
        actual += flat_slice_squared_l2(flat[split:], split, matrix.shape, selection)
        torch.testing.assert_close(actual, expected.square().sum())


# --------------------------------------------------------------------------------------------------
# Synthetic FSDP1 shards over tensor-parallel, context-parallel and replicated ranks


def routed_layer(tp_rank, mova):
    moe = SimpleNamespace(n_experts=4, n_local_experts=2, expert_start_idx=2 * tp_rank, n_shared_experts=2)
    layer = SimpleNamespace(layer_id=0, moe=moe)
    attention_name = "mova" if mova else "attention"
    setattr(layer, attention_name, SimpleNamespace(wr=object(), n_values=3))
    shapes = {f"{attention_name}.w{letter}.weight": (4, 4) for letter in "qko"}
    shapes.update({
        f"{attention_name}.wv.weight": (6, 4) if mova else (4, 4),
        f"{attention_name}.wr.weight": (4, 4),
        "moe.router.weight": (4, 4),
        "moe.experts.weight1": (6, 4),
        "moe.experts.weight2": (8, 3),
        "moe.experts.weight3": (6, 4),
        "moe.fc1.weight": (6, 4),
        "moe.fc2.weight": (4, 6),
        "moe.fc3.weight": (6, 4),
    })
    if mova:
        shapes["mova.router.weight"] = (3, 4)
    weights = {
        name: torch.arange(1, 1 + rows * columns, dtype=torch.float32).reshape(rows, columns) + tp_rank
        for name, (rows, columns) in shapes.items()
    }
    return layer, weights


def flat_handle(layer, weights, shard_rank, shards):
    full = torch.cat([weight.flatten() for weight in weights.values()])
    width = (full.numel() + shards - 1) // shards
    start, end = shard_rank * width, min((shard_rank + 1) * width, full.numel())
    local = full[start:end]
    infos, offset = [], 0
    for weight in weights.values():
        left, right = max(start, offset), min(end, offset + weight.numel())
        infos.append(SimpleNamespace(
            in_shard=right > left, offset_in_shard=max(left - start, 0),
            numel_in_shard=max(right - left, 0), intra_param_start_idx=max(left - offset, 0),
        ))
        offset += weight.numel()
    return SimpleNamespace(_fully_sharded_module=layer, flat_param=SimpleNamespace(
        _fqns=list(weights), _shapes=[weight.shape for weight in weights.values()],
        _shard_param_infos=infos, _local_shard=local, grad=2 * local,
    ))


@pytest.mark.parametrize("mova", [False, True])
def test_expert_ids_and_norms_survive_fsdp_tp_cp_and_replication(mova):
    actual, expected = {}, {}
    for tp_rank in range(2):
        layer, weights = routed_layer(tp_rank, mova)
        for name, weight in weights.items():
            dense_path = _matrix_metric_path(0, name, _MATRIX_PARAMETER_PATHS)
            for selection in matrix_slices(layer, 0, name, weight.shape, dense_path):
                selected = weight if selection.dim is None else weight.narrow(
                    selection.dim, selection.start, selection.stop - selection.start)
                expected[selection.path] = expected.get(selection.path, 0) + selected.square().sum()
        for shard_rank in range(3):
            for replica in range(4):
                model = SimpleNamespace(
                    _all_handles=[flat_handle(layer, weights, shard_rank, 3)],
                    world_size=3, data_parallel_world_size=6, context_parallel_world_size=2,
                )
                collector = HealthMetricCollector(True, 1, 1, enable_weight_norms=True, enable_gradient_norms=True)
                collector.begin_step(1)
                collector.record_matrix_norms(model, gradient_scale=2)
                for name, value in collector.matrix_squared_l2_records.items():
                    actual[name] = actual.get(name, 0) + value
    for family in ("weight_norms", "gradient_norms"):
        for path, value in expected.items():
            torch.testing.assert_close(actual[f"{family}/{path}/l2"], value)
        # Expert 2 belongs to TP rank 1, not rank 0's local expert 0.
        _, tp1_weights = routed_layer(1, mova)
        torch.testing.assert_close(
            actual[f"{family}/layer_000/moe/experts/expert_002/gate_proj/l2"],
            tp1_weights["moe.experts.weight1"][:3].square().sum(),
        )
    assert len(actual) == 2 * len(expected)


# --------------------------------------------------------------------------------------------------
# Real tiny MoE / MoVA Transformers, sharded with FSDP1 and FSDP2

MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=8, num_dense_layers=1)
CONFIGS = {
    "moe": MOE,
    "moe_shared": MOE | dict(num_shared_experts=2),
    "moe_gate": MOE | dict(apply_attn_gate=True),
    "mova": MOE | dict(num_values=4, num_activated_values=2),
    "mova_shared_gate": MOE | dict(num_values=4, num_activated_values=2, num_shared_experts=1,
                                   apply_attn_gate=True),
}


@pytest.fixture
def routed_group(single_process_group, monkeypatch):
    # The router's backward all-reduces its loads over this group.
    monkeypatch.setattr(parallel_state, "_HYBRID_SHARD_DATA_PARALLEL_GROUP", dist.group.WORLD)
    monkeypatch.setattr(parallel_state, "_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE", 1)


def _chunks(matrix, count, dim):
    return matrix.chunk(count, dim=dim)


def reference_matrices(layer, layer_idx, use_grad):
    """Every logged matrix of one unsharded layer, sliced by hand (independently of matrix_slices)."""
    get = (lambda p: p.grad) if use_grad else (lambda p: p.detach())
    prefix = f"layer_{layer_idx:03d}"
    out = {}
    attention = getattr(layer, "mova", None) or layer.attention
    for letter, name in zip("qko", ("q_proj", "k_proj", "o_proj")):
        out[f"{prefix}/attention/{name}"] = get(getattr(attention, f"w{letter}").weight)
    if attention.wr is not None:
        out[f"{prefix}/attention/output_gate"] = get(attention.wr.weight)
    if hasattr(layer, "mova"):
        for i, value in enumerate(_chunks(get(attention.wv.weight), attention.n_values, 0)):
            out[f"{prefix}/mova/values/value_{i:03d}/v_proj"] = value
        out[f"{prefix}/mova/router"] = get(attention.router.weight)
    else:
        out[f"{prefix}/attention/v_proj"] = get(attention.wv.weight)
    moe = getattr(layer, "moe", None)
    if moe is None:
        for fc, name in (("fc1", "gate_proj"), ("fc3", "up_proj"), ("fc2", "down_proj")):
            out[f"{prefix}/mlp/{name}"] = get(getattr(layer.nffn, fc).weight)
        return out
    out[f"{prefix}/moe/router"] = get(moe.router.weight)
    for weight, name in (("weight1", "gate_proj"), ("weight2", "down_proj"), ("weight3", "up_proj")):
        packed = get(getattr(moe.experts, weight))
        for e, expert in enumerate(_chunks(packed, moe.n_experts, 0)):
            out[f"{prefix}/moe/experts/expert_{e:03d}/{name}"] = expert
    if moe.n_shared_experts:
        for fc, name, dim in (("fc1", "gate_proj", 0), ("fc3", "up_proj", 0), ("fc2", "down_proj", 1)):
            packed = get(getattr(moe, fc).weight)
            for e, expert in enumerate(_chunks(packed, moe.n_shared_experts, dim)):
                out[f"{prefix}/moe/shared_experts/expert_{e:03d}/{name}"] = expert
    return out


def expected_norms(model, family, use_grad, scale=1.0):
    return {
        f"{family}/{path}/l2": matrix.float().norm().item() * scale
        for i, layer in enumerate(model.layers)
        for path, matrix in reference_matrices(layer, i, use_grad).items()
    }


def only(metrics, family):
    return {name: value for name, value in metrics.items() if name.startswith(f"{family}/")}


@pytest.mark.usefixtures("routed_group")
@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("config", CONFIGS)
def test_routed_weight_norms_match_unsharded_matrices(backend, config):
    model = tiny_transformer(3, ddp_backend=backend, **CONFIGS[config])
    with torch.no_grad():
        # Distinct values per expert, so a swapped or mis-sliced expert changes the norm.
        for i, p in enumerate(model.parameters()):
            p.copy_(torch.randn_like(p) * (1 + 0.1 * i))
    expected = expected_norms(model, "weight_norms", use_grad=False)

    logged = only(logged_metrics(shard(model, backend), enable_weight_norms=True), "weight_norms")

    assert logged == pytest.approx(expected, rel=1e-5)
    assert all(name.startswith("weight_norms/layer_") for name in logged)  # no embedding or LM head


SCALES = {"unscaled": (1.0, 1.0), "loss_scaled_and_unscaled": (4.0, 4.0), "fraction": (1.0, 0.5)}


@pytest.mark.usefixtures("routed_group")
@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("scales", SCALES)
def test_routed_gradient_norms_match_unsharded_gradients(backend, config, scales):
    loss_scale, gradient_scale = SCALES[scales]
    model = tiny_transformer(3, ddp_backend=backend, **CONFIGS[config])
    reference = copy.deepcopy(model)
    logits, _, _ = reference.eval()(TOKENS, multi_segments=False)
    logits.float().square().mean().backward()
    expected = expected_norms(reference, "gradient_norms", use_grad=True, scale=loss_scale / gradient_scale)
    assert any(value > 0 for name, value in expected.items() if "/experts/" in name)

    logged = only(logged_metrics(
        shard(model, backend), loss_scale=loss_scale, gradient_scale=gradient_scale,
        enable_gradient_norms=True,
    ), "gradient_norms")

    assert logged == pytest.approx(expected, rel=1e-4, abs=1e-7)


@pytest.mark.usefixtures("routed_group")
@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
def test_routed_weight_and_gradient_norms_together_log_what_each_logs_alone(backend):
    def run(**flags):
        model = tiny_transformer(3, ddp_backend=backend, **CONFIGS["mova_shared_gate"])
        return logged_metrics(shard(model, backend), loss_scale=2.0, **flags)

    both = run(enable_weight_norms=True, enable_gradient_norms=True)
    weights = only(run(enable_weight_norms=True), "weight_norms")
    gradients = only(run(enable_gradient_norms=True), "gradient_norms")

    assert only(both, "weight_norms") == pytest.approx(weights, rel=1e-6)
    assert only(both, "gradient_norms") == pytest.approx(gradients, rel=1e-6)
    # 3 layers: 1 dense (4 attention + gate + 3 MLP), 2 MoVA (3 attention + gate + 4 values + value router
    # + MoE router + 3 x (4 routed + 1 shared) experts).
    assert len(weights) == len(gradients) == 8 + 2 * (3 + 1 + 4 + 1 + 1 + 3 * 5)
