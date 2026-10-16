import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard
from torch.utils.checkpoint import checkpoint

import xllm.health_metrics as health_metrics
import xllm.metrics as metric_logging


class RecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))


MATRIX_FQNS = [
    "attention.wq.weight",
    "attention.wk.weight",
    "attention.wv.weight",
    "attention.wo.weight",
    "nffn.fc1.weight",
    "nffn.fc3.weight",
    "nffn.fc2.weight",
]


def make_fsdp1_model(
    weight_values,
    gradient_values=None,
    shard_infos=None,
    fsdp_size=1,
    replica_count=1,
    context_parallel_size=1,
):
    if shard_infos is None:
        shard_infos = [
            SimpleNamespace(in_shard=True, offset_in_shard=i, numel_in_shard=1)
            for i in range(len(MATRIX_FQNS))
        ]
    flat_param = SimpleNamespace(
        _fqns=MATRIX_FQNS,
        _shard_param_infos=shard_infos,
        _local_shard=torch.tensor(weight_values, dtype=torch.float32),
        grad=(
            None if gradient_values is None
            else torch.tensor(gradient_values, dtype=torch.float32)
        ),
    )
    handle = SimpleNamespace(
        _fully_sharded_module=SimpleNamespace(layer_id=0),
        flat_param=flat_param,
    )
    return SimpleNamespace(
        _all_handles=[handle],
        world_size=fsdp_size,
        data_parallel_world_size=fsdp_size * replica_count,
        context_parallel_world_size=context_parallel_size,
    )


class FakeDeviceMesh:
    def __init__(self, shape):
        self.shape = shape

    def size(self, mesh_dim):
        return self.shape[mesh_dim]


class FakeDTensor:
    def __init__(self, local_values, placements, mesh_shape):
        self.local_tensor = torch.as_tensor(local_values, dtype=torch.float32)
        self.placements = placements
        self.device_mesh = FakeDeviceMesh(mesh_shape)
        self.grad = None

    def to_local(self):
        return self.local_tensor


def make_fsdp2_model(
    weight_values,
    gradient_values=None,
    placements=(Shard(0),),
    mesh_shape=(1,),
):
    parameters = []
    for index, (fqn, values) in enumerate(zip(MATRIX_FQNS, weight_values)):
        weight = FakeDTensor(values, placements, mesh_shape)
        if gradient_values is not None:
            weight.grad = FakeDTensor(gradient_values[index], placements, mesh_shape)
        parameters.append((fqn, weight))
    layer = SimpleNamespace(named_parameters=lambda: iter(parameters))
    return SimpleNamespace(layers=[layer])


def test_activation_stats_follow_frequency():
    collector = health_metrics.HealthMetricCollector(enabled=True, freq=2, num_layers=2)
    logger = RecordingLogger()

    collector.begin_step(1)
    assert health_metrics.get_active_health_collector() is None
    collector.flush(logger, 1)
    assert logger.calls == []

    collector.begin_step(2)
    active = health_metrics.get_active_health_collector()
    assert active is collector
    active.record_activation(0, torch.tensor([-2.0, 0.0, 4.0]))
    active.record_activation(1, torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
    collector.end_forward()
    collector.flush(logger, 2)

    metrics, step = logger.calls[0]
    assert step == 2
    assert metrics["activation_stats/layer_000/mean"] == pytest.approx(2.0 / 3.0)
    assert metrics["activation_stats/layer_000/min"] == -2.0
    assert metrics["activation_stats/layer_000/max"] == 4.0
    assert metrics["activation_stats/layer_001/mean"] == 4.0


def test_query_key_coordinate_and_vector_stats():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_query_key_stats=True
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0]))
    collector.record_query_key_stats(
        0,
        torch.tensor([[[[3.0, 4.0]], [[0.0, 2.0]]]]),
        torch.tensor([[[[-3.0, 4.0]]]]),
    )
    collector.end_forward()
    collector.flush(logger, 1)

    metrics, _ = logger.calls[0]
    assert metrics["q_stats/layer_000/mean"] == 2.25
    assert metrics["q_stats/layer_000/min"] == 0.0
    assert metrics["q_stats/layer_000/max"] == 4.0
    assert metrics["q_stats/layer_000/vector_l2_mean"] == 3.5
    assert metrics["q_stats/layer_000/vector_l2_min"] == 2.0
    assert metrics["q_stats/layer_000/vector_l2_max"] == 5.0
    assert metrics["k_stats/layer_000/mean"] == 0.5
    assert metrics["k_stats/layer_000/min"] == -3.0
    assert metrics["k_stats/layer_000/max"] == 4.0
    assert metrics["k_stats/layer_000/vector_l2_mean"] == 5.0
    assert metrics["k_stats/layer_000/vector_l2_min"] == 5.0
    assert metrics["k_stats/layer_000/vector_l2_max"] == 5.0


def test_logit_stats_share_existing_reductions():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_logits_stats=True
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0]))
    collector.record_logits_stats(
        torch.tensor(4.0), torch.tensor(2), torch.tensor(-2.0), torch.tensor(5.0)
    )
    collector.end_forward()
    reductions = []

    def reduce(tensor, dst, op):
        assert dst == 0
        reductions.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            tensor[1] += 8.0
            tensor[3] += 2.0
        elif op == torch.distributed.ReduceOp.MIN:
            tensor[-1] = -7.0
        elif op == torch.distributed.ReduceOp.MAX:
            tensor[-1] = 9.0

    with patch.object(health_metrics.dist, "is_initialized", return_value=True), \
         patch.object(health_metrics.dist, "all_reduce"), \
         patch.object(health_metrics.dist, "get_rank", return_value=0), \
         patch.object(health_metrics.dist, "reduce", side_effect=reduce):
        collector.flush(logger, 1)

    assert reductions == [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ]
    assert logger.calls[0][0]["logits_stats/mean"] == 3.0
    assert logger.calls[0][0]["logits_stats/min"] == -7.0
    assert logger.calls[0][0]["logits_stats/max"] == 9.0


def test_weight_norms_reconstruct_all_seven_matrices():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_weight_norms=True
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0]))
    collector.record_matrix_norms(make_fsdp1_model(
        [3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0]
    ))
    collector.end_forward()
    collector.flush(logger, 1)

    metrics, _ = logger.calls[0]
    assert metrics["weight_norms/layer_000/attention/q_proj/l2"] == 3.0
    assert metrics["weight_norms/layer_000/attention/k_proj/l2"] == 4.0
    assert metrics["weight_norms/layer_000/attention/v_proj/l2"] == 5.0
    assert metrics["weight_norms/layer_000/attention/o_proj/l2"] == 6.0
    assert metrics["weight_norms/layer_000/mlp/gate_proj/l2"] == 7.0
    assert metrics["weight_norms/layer_000/mlp/up_proj/l2"] == 8.0
    assert metrics["weight_norms/layer_000/mlp/down_proj/l2"] == 9.0


def test_gradient_norms_reconstruct_all_seven_matrices_and_unscale():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_gradient_norms=True
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0]))
    collector.record_matrix_norms(make_fsdp1_model(
        [0.0] * 7,
        gradient_values=[6.0, -8.0, 10.0, -12.0, 14.0, -16.0, 18.0],
    ), gradient_scale=2.0)
    collector.end_forward()
    collector.flush(logger, 1)

    metrics, _ = logger.calls[0]
    expected = {
        "attention/q_proj": 3.0,
        "attention/k_proj": 4.0,
        "attention/v_proj": 5.0,
        "attention/o_proj": 6.0,
        "mlp/gate_proj": 7.0,
        "mlp/up_proj": 8.0,
        "mlp/down_proj": 9.0,
    }
    for matrix, norm in expected.items():
        assert metrics[f"gradient_norms/layer_000/{matrix}/l2"] == norm


def test_weight_norms_use_exact_local_shard_and_replica_correction():
    shard_infos = [
        SimpleNamespace(in_shard=True, offset_in_shard=0, numel_in_shard=2),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=True, offset_in_shard=2, numel_in_shard=1),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
    ]
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_weight_norms=True
    )
    collector.begin_step(1)
    collector.record_matrix_norms(make_fsdp1_model(
        [3.0, 4.0, 12.0],
        shard_infos=shard_infos,
        fsdp_size=2,
        replica_count=2,
        context_parallel_size=2,
    ))

    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/attention/q_proj/l2"
    ].item() == pytest.approx(25.0 / 4.0)
    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/mlp/up_proj/l2"
    ].item() == pytest.approx(144.0 / 4.0)
    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/attention/k_proj/l2"
    ].item() == 0.0
    collector.end_forward()


def test_fsdp2_weight_norms_use_per_parameter_dtensor_shards():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_weight_norms=True
    )
    collector.begin_step(1)

    with patch.object(health_metrics, "DTensor", FakeDTensor):
        collector.record_matrix_norms(make_fsdp2_model(
            [[3.0, 4.0], [-5.0], [6.0], [-7.0], [8.0], [-9.0], [10.0]],
            placements=(Shard(0),),
            mesh_shape=(2,),
        ))

    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/attention/q_proj/l2"
    ].item() == 25.0
    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/mlp/down_proj/l2"
    ].item() == 100.0
    collector.end_forward()


def test_fsdp2_weight_norms_correct_hybrid_and_context_replicas():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_weight_norms=True
    )
    collector.begin_step(1)

    with patch.object(health_metrics, "DTensor", FakeDTensor):
        collector.record_matrix_norms(make_fsdp2_model(
            [[3.0, 4.0], [-5.0], [6.0], [-7.0], [8.0], [-9.0], [10.0]],
            placements=(Replicate(), Shard(0)),
            mesh_shape=(4, 2),
        ))

    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/attention/q_proj/l2"
    ].item() == pytest.approx(25.0 / 4.0)
    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/mlp/down_proj/l2"
    ].item() == pytest.approx(100.0 / 4.0)
    collector.end_forward()


def test_gradient_norms_use_exact_local_shard_and_replica_correction():
    shard_infos = [
        SimpleNamespace(in_shard=True, offset_in_shard=0, numel_in_shard=2),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
        SimpleNamespace(in_shard=True, offset_in_shard=2, numel_in_shard=1),
        SimpleNamespace(in_shard=False, offset_in_shard=0, numel_in_shard=0),
    ]
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_gradient_norms=True
    )
    collector.begin_step(1)
    collector.record_matrix_norms(make_fsdp1_model(
        [0.0, 0.0, 0.0],
        gradient_values=[6.0, 8.0, 24.0],
        shard_infos=shard_infos,
        fsdp_size=2,
        replica_count=2,
        context_parallel_size=2,
    ), gradient_scale=2.0)

    assert collector.matrix_squared_l2_records[
        "gradient_norms/layer_000/attention/q_proj/l2"
    ].item() == pytest.approx(100.0 / (4.0 * 2.0 ** 2))
    assert collector.matrix_squared_l2_records[
        "gradient_norms/layer_000/mlp/up_proj/l2"
    ].item() == pytest.approx(576.0 / (4.0 * 2.0 ** 2))
    assert collector.matrix_squared_l2_records[
        "gradient_norms/layer_000/attention/k_proj/l2"
    ].item() == 0.0
    collector.end_forward()


def test_fsdp2_matrix_norms_use_per_parameter_dtensor_shards():
    collector = health_metrics.HealthMetricCollector(
        enabled=True,
        freq=1,
        num_layers=1,
        enable_weight_norms=True,
        enable_gradient_norms=True,
    )
    collector.begin_step(1)
    weight_values = [
        [3.0, 4.0], [-5.0], [6.0], [-7.0], [8.0], [-9.0], [10.0]
    ]
    gradient_values = [
        [6.0, 8.0], [-10.0], [12.0], [-14.0], [16.0], [-18.0], [20.0]
    ]

    with patch.object(health_metrics, "DTensor", FakeDTensor):
        collector.record_matrix_norms(
            make_fsdp2_model(
                weight_values,
                gradient_values,
                placements=(Shard(0),),
                mesh_shape=(2,),
            ),
            gradient_scale=2.0,
        )

    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/attention/q_proj/l2"
    ].item() == 25.0
    assert collector.matrix_squared_l2_records[
        "gradient_norms/layer_000/attention/q_proj/l2"
    ].item() == 25.0
    assert collector.matrix_squared_l2_records[
        "gradient_norms/layer_000/mlp/down_proj/l2"
    ].item() == 100.0
    collector.end_forward()


def test_fsdp2_matrix_norms_correct_hybrid_and_context_replicas():
    collector = health_metrics.HealthMetricCollector(
        enabled=True,
        freq=1,
        num_layers=1,
        enable_weight_norms=True,
        enable_gradient_norms=True,
    )
    collector.begin_step(1)
    weight_values = [[3.0, 4.0]] + [[1.0]] * 6
    gradient_values = [[6.0, 8.0]] + [[2.0]] * 6

    with patch.object(health_metrics, "DTensor", FakeDTensor):
        collector.record_matrix_norms(
            make_fsdp2_model(
                weight_values,
                gradient_values,
                placements=(Replicate(), Shard(0)),
                mesh_shape=(4, 2),
            ),
            gradient_scale=2.0,
        )

    assert collector.matrix_squared_l2_records[
        "weight_norms/layer_000/attention/q_proj/l2"
    ].item() == pytest.approx(25.0 / 4.0)
    assert collector.matrix_squared_l2_records[
        "gradient_norms/layer_000/attention/q_proj/l2"
    ].item() == pytest.approx(25.0 / 4.0)
    collector.end_forward()


def test_weight_norms_share_existing_sum_reduction():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_weight_norms=True
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0]))
    collector.record_matrix_norms(make_fsdp1_model([1.0] * 7))
    collector.end_forward()
    reductions = []

    def reduce(tensor, dst, op):
        assert dst == 0
        reductions.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            assert tensor.numel() == 9  # Activation sum/count and seven squared norms.

    with patch.object(health_metrics.dist, "is_initialized", return_value=True), \
         patch.object(health_metrics.dist, "all_reduce"), \
         patch.object(health_metrics.dist, "get_rank", return_value=0), \
         patch.object(health_metrics.dist, "reduce", side_effect=reduce):
        collector.flush(logger, 1)

    assert reductions == [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ]
    assert len([
        key for key in logger.calls[0][0] if key.startswith("weight_norms/")
    ]) == 7


def test_logit_and_matrix_norms_share_existing_sum_reduction():
    collector = health_metrics.HealthMetricCollector(
        enabled=True,
        freq=1,
        num_layers=1,
        enable_weight_norms=True,
        enable_gradient_norms=True,
        enable_logits_stats=True,
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0]))
    collector.record_matrix_norms(make_fsdp1_model(
        [1.0] * 7, gradient_values=[2.0] * 7
    ), gradient_scale=2.0)
    collector.record_logits_stats(
        torch.tensor(6.0), torch.tensor(2), torch.tensor(-1.0), torch.tensor(4.0)
    )
    collector.end_forward()
    reductions = []

    def reduce(tensor, dst, op):
        assert dst == 0
        reductions.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            assert tensor.numel() == 18  # Two sums/counts and fourteen squared norms.

    with patch.object(health_metrics.dist, "is_initialized", return_value=True), \
         patch.object(health_metrics.dist, "all_reduce"), \
         patch.object(health_metrics.dist, "get_rank", return_value=0), \
         patch.object(health_metrics.dist, "reduce", side_effect=reduce):
        collector.flush(logger, 1)

    assert reductions == [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ]
    assert len([
        key for key in logger.calls[0][0] if key.startswith("weight_norms/")
    ]) == 7
    assert len([
        key for key in logger.calls[0][0] if key.startswith("gradient_norms/")
    ]) == 7
    assert logger.calls[0][0]["logits_stats/mean"] == 3.0
    assert logger.calls[0][0]["logits_stats/min"] == -1.0
    assert logger.calls[0][0]["logits_stats/max"] == 4.0
    assert logger.calls[0][0]["weight_norms/layer_000/attention/q_proj/l2"] == 1.0
    assert logger.calls[0][0]["gradient_norms/layer_000/attention/q_proj/l2"] == 1.0


def test_health_stats_use_three_packed_reductions():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_query_key_stats=True
    )
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0, 3.0]))
    collector.record_query_key_stats(
        0, torch.tensor([[[[1.0, 2.0]]]]), torch.tensor([[[[3.0, 4.0]]]])
    )
    collector.end_forward()
    reductions = []

    def reduce(tensor, dst, op):
        assert dst == 0
        reductions.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            tensor[:5] += 8.0
            tensor[5:] += 2.0
        elif op == torch.distributed.ReduceOp.MIN:
            tensor.copy_(torch.minimum(tensor, torch.tensor([-4.0])))
        else:
            tensor.copy_(torch.maximum(tensor, torch.tensor([9.0])))

    with patch.object(health_metrics.dist, "is_initialized", return_value=True), \
         patch.object(health_metrics.dist, "all_reduce"), \
         patch.object(health_metrics.dist, "get_rank", return_value=0), \
         patch.object(health_metrics.dist, "reduce", side_effect=reduce):
        collector.flush(logger, 1)

    metrics, _ = logger.calls[0]
    assert reductions == [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ]
    assert metrics["activation_stats/layer_000/mean"] == 3.0
    assert metrics["activation_stats/layer_000/min"] == -4.0
    assert metrics["activation_stats/layer_000/max"] == 9.0


def test_checkpoint_recomputation_does_not_duplicate_stats():
    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=1, num_layers=1, enable_query_key_stats=True
    )
    logger = RecordingLogger()
    calls = 0

    def layer(x):
        nonlocal calls
        calls += 1
        active = health_metrics.get_active_health_collector()
        if active is not None:
            active.record_query_key_stats(0, x, x)
        return torch.sin(x)

    x = torch.tensor([1.0], requires_grad=True)
    collector.begin_step(1)
    active = health_metrics.get_active_health_collector()
    y = checkpoint(layer, x, use_reentrant=False)
    active.record_activation(0, y)
    collector.end_forward()
    y.sum().backward()
    collector.flush(logger, 1)

    assert calls == 2
    assert len(logger.calls) == 1


@pytest.mark.parametrize("enable_prefix_keys,prefix", [(False, ""), (True, "health/")])
def test_metric_logger_batches_tensorboard_and_wandb(tmp_path, enable_prefix_keys, prefix):
    metrics = {
        "activation_stats/layer_000/mean": 1.0,
        "activation_stats/layer_000/min": -2.0,
        "activation_stats/layer_000/max": 3.0,
    }
    expected = {f"{prefix}{key}": value for key, value in metrics.items()}
    logger = metric_logging.MetricLogger(
        outdir=tmp_path, tag="health", enable=True,
        enable_prefix_keys=enable_prefix_keys
    )
    logger.tb_writer = MagicMock()
    logger.tb_writer.file_writer = MagicMock()
    logger.log_wandb = True

    with patch.object(metric_logging, "wandb", create=True) as wandb:
        logger.log(metrics, step=4)
        wandb.log.assert_called_once_with(expected, step=4)

    add_summary = logger.tb_writer.file_writer.add_summary
    add_summary.assert_called_once()
    summary = add_summary.call_args.args[0]
    assert {value.tag for value in summary.value} == set(expected)
    logger.close()

    records = (tmp_path / "metrics.health.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["activation_stats/layer_000/mean"] == 1.0
