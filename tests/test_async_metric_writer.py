import json
import time
from unittest.mock import MagicMock, patch

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from xllm.metrics import metric_logger
from xllm.metrics.metric_logger import AsyncMetricWriter, MetricLogger

pytestmark = pytest.mark.filterwarnings("ignore:datetime.datetime.utcnow:DeprecationWarning")


def make_logger(writer, outdir, tag="train", **kwargs):
    return MetricLogger(outdir, tag=tag, enable=True, async_writer=writer, **kwargs)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def scalars(outdir):
    events = EventAccumulator(str(outdir / "tb")).Reload()
    return {tag: [(e.step, e.value) for e in events.Scalars(tag)] for tag in events.Tags()["scalars"]}


def stalled_writer(*args):
    time.sleep(60)


def run_writer(records, outdir, tags, log_tb=False, wandb_init_kwargs=None):
    queue = MagicMock()
    queue.get.side_effect = [records, None]
    metric_logger._run_metric_writer(queue, MagicMock(), MagicMock(), outdir, tags, log_tb, wandb_init_kwargs, 16)


def test_loggers_are_batched_by_step(tmp_path):
    writer = AsyncMetricWriter()
    health, train = make_logger(writer, tmp_path, "health"), make_logger(writer, tmp_path, "train")
    health.log({"activation": 1.0}, step=1)
    train.log({"loss": 2.0}, step=1)
    health.log({"activation": 3.0}, step=2)
    writer.close()

    assert [r["global_step"] for r in jsonl(tmp_path / "metrics.health.jsonl")] == [1, 2]
    assert [r["global_step"] for r in jsonl(tmp_path / "metrics.train.jsonl")] == [1]
    assert writer.get_stats()["async_writer/written_batches"] == 2


def test_async_matches_sync_output(tmp_path):
    for mode, writer in (("sync", None), ("async", AsyncMetricWriter())):
        outdir = tmp_path / mode
        loggers = {
            "train": make_logger(writer, outdir, "train", log_tb=True),
            "health": make_logger(writer, outdir, "health", log_tb=True, enable_prefix_keys=False),
        }
        for step in (1, 2):
            loggers["train"].log({"loss": 1.0 / step}, step)
            loggers["health"].log(
                {"weight_norms/expert_000/l2": 3.0, "weight_norms/expert_001/l2": 5.0},
                step, dashboard_metrics={"weight_norms/experts/l2/mean": 4.0},
            )
        if writer is not None:
            writer.close()
        for logger in loggers.values():
            logger.close()

    for tag in ("train", "health"):
        sync, asynchronous = (jsonl(tmp_path / mode / f"metrics.{tag}.jsonl") for mode in ("sync", "async"))
        for row in sync + asynchronous:
            del row["created_at"]
        assert sync == asynchronous
    assert scalars(tmp_path / "sync") == scalars(tmp_path / "async") == {
        "train/loss": [(1, 1.0), (2, 0.5)], "weight_norms/experts/l2/mean": [(1, 4.0), (2, 4.0)],
    }


def test_writer_process_owns_wandb_run_and_gets_dashboard_metrics(tmp_path):
    record = metric_logger._MetricRecord(
        tag="health", metrics={"norm/a": 1.0, "norm/b": 3.0}, enable_prefix_keys=False, step=1,
        created_at="now", dashboard_metrics={"norm/mean": 2.0}, log_wandb=True,
    )
    with (
        patch.object(metric_logger, "_pin_writer_process"),
        patch.object(metric_logger, "wandb", create=True) as wandb,
    ):
        run_writer([record], tmp_path, ("health",), wandb_init_kwargs={"project": "test"})

    wandb.init.assert_called_once_with(project="test")
    wandb.init.return_value.log.assert_called_once_with({"norm/mean": 2.0}, step=1)
    wandb.init.return_value.finish.assert_called_once_with()
    assert jsonl(tmp_path / "metrics.health.jsonl")[0] == {
        "global_step": 1, "created_at": "now", "norm/a": 1.0, "norm/b": 3.0,
    }


def test_each_sink_gets_one_filtered_batch(tmp_path):
    writer = AsyncMetricWriter()
    preferences = {
        "shared": (True, True),
        "tensorboard_only": (True, False),
        "wandb_only": (False, True),
        "json_only": (False, False),
    }
    with patch.object(metric_logger, "has_wandb", True):
        for tag, (log_tb, log_wandb) in preferences.items():
            make_logger(writer, tmp_path, tag, log_tb=log_tb, log_wandb=log_wandb).log({"value": 2.0}, 3)

    with (
        patch.object(metric_logger, "_pin_writer_process"),
        patch.object(metric_logger, "SummaryWriter") as tensorboard,
        patch.object(metric_logger, "wandb", create=True) as wandb,
    ):
        run_writer(writer.pending_records, tmp_path, tuple(preferences), True, {"project": "test"})

    wandb.init.return_value.log.assert_called_once_with({"shared/value": 2.0, "wandb_only/value": 2.0}, step=3)
    add_summary = tensorboard.return_value.file_writer.add_summary
    add_summary.assert_called_once()
    assert {value.tag for value in add_summary.call_args.args[0].value} == {"shared/value", "tensorboard_only/value"}
    for tag in preferences:
        assert jsonl(tmp_path / f"metrics.{tag}.jsonl")[0]["value"] == 2.0


def test_tensorboard_opt_out_is_kept(tmp_path):
    writer = AsyncMetricWriter()
    make_logger(writer, tmp_path, "train", log_tb=True).log({"loss": 1.0}, 1)
    make_logger(writer, tmp_path, "json_only", log_tb=False).log({"value": 2.0}, 1)
    writer.close()

    assert list(scalars(tmp_path)) == ["train/loss"]
    assert jsonl(tmp_path / "metrics.json_only.jsonl")[0]["value"] == 2.0


def test_close_reports_dead_worker_with_full_queue(tmp_path):
    writer = AsyncMetricWriter(max_queue_size=1, timeout_seconds=0.5)
    make_logger(writer, tmp_path).log({"loss": 1.0}, 1)
    writer._start()
    writer.process.terminate()
    writer.process.join(timeout=5)
    writer.queue.put(["full"])

    with pytest.raises(RuntimeError, match="writer exited"):
        writer.close()
    assert not writer.process.is_alive()
    writer.close()


@pytest.mark.parametrize("operation", ["submit", "close"])
def test_stalled_worker_has_bounded_wait(tmp_path, operation):
    writer = AsyncMetricWriter(max_queue_size=1, timeout_seconds=0.2)
    logger = make_logger(writer, tmp_path)
    with patch.object(metric_logger, "_run_metric_writer", stalled_writer):
        logger.log({"loss": 1.0}, 1)
        writer.flush()
        if operation == "submit":
            logger.log({"loss": 2.0}, 2)
            with pytest.raises(TimeoutError, match="submitting metrics"):
                writer.flush()
        with pytest.raises(TimeoutError):
            writer.close()
    assert not writer.process.is_alive()


def test_close_drains_unpolled_telemetry(tmp_path):
    writer = AsyncMetricWriter()
    logger = make_logger(writer, tmp_path)
    with writer:
        for step in range(2000):
            logger.log({"loss": 1.0}, step)
            writer.flush()
    assert not writer.process.is_alive()
    stats = writer.get_stats()
    assert stats["async_writer/written_batches"] == stats["async_writer/submitted_batches"] == 2000
    assert len(jsonl(tmp_path / "metrics.train.jsonl")) == 2000


def test_context_closes_writer_on_training_error(tmp_path):
    writer = AsyncMetricWriter()
    logger = make_logger(writer, tmp_path)
    with pytest.raises(ValueError, match="training failed"):
        with writer:
            logger.log({"loss": 1.0}, 1)
            writer.flush()
            raise ValueError("training failed")
    assert not writer.process.is_alive()
    assert writer.get_stats()["async_writer/written_batches"] == 1


def test_cleanup_error_preserves_training_error(caplog):
    writer = AsyncMetricWriter()
    with patch.object(writer, "close", side_effect=RuntimeError("sink failed")):
        with pytest.raises(ValueError, match="training failed"):
            with writer:
                raise ValueError("training failed")
    assert "Metric writer cleanup failed" in caplog.text


def test_worker_error_is_reported(tmp_path):
    writer = AsyncMetricWriter()
    make_logger(writer, tmp_path).log({"not_json_serializable": object()}, 1)
    with pytest.raises(RuntimeError, match="Asynchronous metric writ"):
        writer.close()
    assert not writer.process.is_alive()


@pytest.mark.parametrize("kwargs", [
    {"max_queue_size": 0},
    {"jsonl_flush_interval": 0},
    {"timeout_seconds": 0},
    {"timeout_seconds": float("inf")},
])
def test_writer_rejects_unbounded_settings(kwargs):
    with pytest.raises(ValueError):
        AsyncMetricWriter(**kwargs)
