import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import xllm.health_metrics as health_metrics
import xllm.metrics as metric_logging


class RecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))


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


def test_health_stats_use_three_packed_reductions():
    collector = health_metrics.HealthMetricCollector(enabled=True, freq=1, num_layers=1)
    logger = RecordingLogger()
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([1.0, 3.0]))
    collector.end_forward()
    reductions = []

    def reduce(tensor, dst, op):
        assert dst == 0
        reductions.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            tensor[0] += 8.0
            tensor[1] += 2.0
        elif op == torch.distributed.ReduceOp.MIN:
            tensor.copy_(torch.minimum(tensor, torch.tensor([-4.0])))
        else:
            tensor.copy_(torch.maximum(tensor, torch.tensor([9.0])))

    with patch.object(health_metrics.dist, "is_initialized", return_value=True), \
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
    collector = health_metrics.HealthMetricCollector(enabled=True, freq=1, num_layers=1)
    logger = RecordingLogger()
    calls = 0

    def layer(x):
        nonlocal calls
        calls += 1
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
