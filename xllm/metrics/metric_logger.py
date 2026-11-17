from typing import Optional, Dict, Any, TextIO
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from multiprocessing import get_context
from pathlib import Path
from queue import Empty, Full
from time import perf_counter
import json
import math
import os
import traceback
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

logger = getLogger()


@dataclass(frozen=True)
class _MetricRecord:
    tag: Optional[str]
    metrics: Dict[str, Any]
    enable_prefix_keys: bool
    step: int
    created_at: str
    dashboard_metrics: Optional[Dict[str, Any]] = None
    log_tb: bool = False
    log_wandb: bool = False


def _prefixed_metrics(record: _MetricRecord) -> Dict[str, Any]:
    metrics = record.metrics if record.dashboard_metrics is None else record.dashboard_metrics
    if record.tag is None or not record.enable_prefix_keys:
        return metrics
    return {f"{record.tag}/{key}": value for key, value in metrics.items()}


def _merge_prefixed_metrics(records: list[_MetricRecord]) -> Dict[str, Any]:
    metrics = {}
    for record in records:
        prefixed_metrics = _prefixed_metrics(record)
        duplicate_keys = metrics.keys() & prefixed_metrics.keys()
        if duplicate_keys:
            raise ValueError(f"Duplicate metrics at step {record.step}: {duplicate_keys}")
        metrics.update(prefixed_metrics)
    return metrics


def _pin_writer_process() -> None:
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available_cpus = os.sched_getaffinity(0)
        if available_cpus:
            os.sched_setaffinity(0, {max(available_cpus)})


def _run_metric_writer(
    queue,
    error_queue,
    stats_queue,
    outdir: Path,
    tags: tuple[Optional[str], ...],
    log_tb: bool,
    wandb_init_kwargs: Optional[Dict[str, Any]],
    jsonl_flush_interval: int,
) -> None:
    _pin_writer_process()
    tb_writer = None
    jsonl_writers = {}
    wandb_run = None
    batches_since_flush = 0
    try:
        if log_tb:
            tb_writer = SummaryWriter(log_dir=str(outdir / "tb"), max_queue=1000)
        jsonl_writers = {
            tag: open(outdir / (f"metrics.{tag}.jsonl" if tag else "metrics.jsonl"), "a")
            for tag in tags
        }
        if wandb_init_kwargs is not None:
            wandb_run = wandb.init(**wandb_init_kwargs)

        while True:
            records = queue.get()
            if records is None:
                break
            start = perf_counter()
            step = records[0].step
            if any(record.step != step for record in records):
                raise ValueError("Metric batch contains multiple steps")
            tb_records = [record for record in records if record.log_tb]
            wandb_records = [record for record in records if record.log_wandb]
            tb_metrics = _merge_prefixed_metrics(tb_records)
            wandb_metrics = tb_metrics if wandb_records == tb_records else _merge_prefixed_metrics(wandb_records)

            if tb_writer is not None and tb_metrics:
                summary = Summary(value=[
                    Summary.Value(tag=k, simple_value=float(v)) for k, v in tb_metrics.items()
                ])
                tb_writer.file_writer.add_summary(summary, global_step=step)

            if wandb_run is not None and wandb_metrics:
                wandb_run.log(wandb_metrics, step=step)

            for record in records:
                json_metrics = dict(global_step=step, created_at=record.created_at)
                json_metrics.update(record.metrics)
                print(json.dumps(json_metrics), file=jsonl_writers[record.tag])

            batches_since_flush += 1
            if batches_since_flush == jsonl_flush_interval:
                for writer in jsonl_writers.values():
                    writer.flush()
                batches_since_flush = 0
            stats_queue.put(perf_counter() - start)
    except BaseException:
        error_queue.put(traceback.format_exc())
        raise
    finally:
        if tb_writer is not None:
            tb_writer.close()
        for writer in jsonl_writers.values():
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()


class AsyncMetricWriter:
    """Write one metric batch per step in a separate process."""

    def __init__(
        self,
        max_queue_size: int = 32,
        jsonl_flush_interval: int = 16,
        wandb_init_kwargs: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 60.0,
    ):
        if max_queue_size < 1:
            raise ValueError("Metric queue size must be positive")
        if jsonl_flush_interval < 1:
            raise ValueError("JSONL flush interval must be positive")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("Metric writer timeout must be finite and positive")
        self.context = get_context("spawn")
        self.max_queue_size = max_queue_size
        self.jsonl_flush_interval = jsonl_flush_interval
        self.wandb_init_kwargs = wandb_init_kwargs
        self.timeout_seconds = timeout_seconds
        self.closed = False
        self.started = False
        self.pending_step: Optional[int] = None
        self.pending_records: list[_MetricRecord] = []
        self.outdir: Optional[Path] = None
        self.tags = set()
        self.log_tb = False
        self.log_wandb = False
        self.queue = None
        self.error_queue = None
        self.stats_queue = None
        self.process = None
        self.max_queue_depth = 0
        self.submit_block_seconds = 0.0
        self.submitted_batches = 0
        self.write_seconds = 0.0
        self.written_batches = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                logger.exception("Metric writer cleanup failed during a training error")

    def register(self, outdir: Path, tag: Optional[str], log_tb: bool, log_wandb: bool) -> None:
        if self.outdir is None:
            self.outdir = outdir
            outdir.mkdir(exist_ok=True, parents=True)
        elif self.outdir != outdir:
            raise ValueError("Async metric loggers must share one output directory")
        if self.started:
            raise RuntimeError("Cannot register a metric logger after writing starts")
        if tag in self.tags:
            raise ValueError(f"Duplicate asynchronous metric logger tag: {tag}")

        if log_tb:
            (outdir / "tb").mkdir(exist_ok=True, parents=False)
            self.log_tb = True
        self.log_wandb |= log_wandb and has_wandb
        self.tags.add(tag)
        (outdir / (f"metrics.{tag}.jsonl" if tag else "metrics.jsonl")).touch(exist_ok=True)

    def _start(self) -> None:
        if self.started:
            return
        if self.outdir is None:
            raise RuntimeError("No asynchronous metric loggers are registered")
        if self.log_wandb and self.wandb_init_kwargs is None:
            raise RuntimeError("W&B initialization settings are missing")

        self.queue = self.context.Queue(maxsize=self.max_queue_size)
        self.error_queue = self.context.Queue(maxsize=1)
        self.stats_queue = self.context.Queue()
        self.process = self.context.Process(
            target=_run_metric_writer,
            name="metric-writer",
            args=(
                self.queue,
                self.error_queue,
                self.stats_queue,
                self.outdir,
                tuple(sorted(self.tags, key=lambda tag: tag or "")),
                self.log_tb,
                self.wandb_init_kwargs if self.log_wandb else None,
                self.jsonl_flush_interval,
            ),
        )
        self.process.start()
        self.started = True

    def submit(self, record: _MetricRecord) -> None:
        self._raise_if_failed()
        if self.closed:
            raise RuntimeError("Metric writer is closed")
        if self.pending_step is None:
            self.pending_step = record.step
        elif record.step < self.pending_step:
            raise ValueError(f"Metric steps must be monotonic: {record.step} < {self.pending_step}")
        elif record.step > self.pending_step:
            self._submit_pending()
            self.pending_step = record.step
        self.pending_records.append(record)

    def _enqueue_batch(self, metric_batch: Optional[list[_MetricRecord]], deadline: Optional[float] = None) -> None:
        if deadline is None:
            deadline = perf_counter() + self.timeout_seconds
        while True:
            self._raise_if_failed()
            if self.process.exitcode == 0:
                raise RuntimeError("Metric writer stopped before accepting all records")
            self._drain_stats()
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("Timed out submitting metrics to the writer")
            try:
                self.queue.put(metric_batch, timeout=min(0.1, remaining))
                return
            except Full:
                pass

    def _submit_pending(self, deadline: Optional[float] = None) -> None:
        if self.pending_records:
            self._start()
            depth = min(self._queue_depth() + 1, self.max_queue_size)
            start = perf_counter()
            self._enqueue_batch(self.pending_records, deadline)
            self.submit_block_seconds += perf_counter() - start
            self.submitted_batches += 1
            self.max_queue_depth = max(self.max_queue_depth, depth)
            self.pending_records = []

    def flush(self) -> None:
        self._raise_if_failed()
        if self.closed:
            raise RuntimeError("Metric writer is closed")
        self._submit_pending()
        self.pending_step = None

    def get_stats(self, reset: bool = False) -> Dict[str, float]:
        self._drain_stats()
        depth = self._queue_depth()
        stats = {
            "async_writer/queue_depth": depth,
            "async_writer/max_queue_depth": self.max_queue_depth,
            "async_writer/submit_block_ms": self.submit_block_seconds * 1e3,
            "async_writer/write_ms": self.write_seconds * 1e3,
            "async_writer/submitted_batches": self.submitted_batches,
            "async_writer/written_batches": self.written_batches,
        }
        if reset:
            self.max_queue_depth = depth
            self.submit_block_seconds = 0.0
            self.submitted_batches = 0
            self.write_seconds = 0.0
            self.written_batches = 0
        return stats

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        deadline = perf_counter() + self.timeout_seconds
        try:
            self._raise_if_failed()
            self._submit_pending(deadline)
            if self.started:
                self._enqueue_batch(None, deadline)
                while self.process.is_alive():
                    # A worker cannot exit until its telemetry feeder is drained.
                    self._drain_stats()
                    self._raise_if_failed()
                    remaining = deadline - perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("Timed out closing the metric writer")
                    self.process.join(timeout=min(0.1, remaining))
                self._drain_stats()
                self._raise_if_failed()
        finally:
            if self.started:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.0)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(timeout=1.0)
                for queue in (self.queue, self.error_queue, self.stats_queue):
                    queue.cancel_join_thread()
                    queue.close()
                self.queue = self.error_queue = self.stats_queue = None

    def _raise_if_failed(self) -> None:
        if not self.started:
            return
        error = None
        if self.error_queue is not None:
            try:
                error = self.error_queue.get_nowait()
            except Empty:
                pass
        if error is not None:
            raise RuntimeError(f"Asynchronous metric writing failed:\n{error}")
        if self.process.exitcode not in (None, 0):
            raise RuntimeError(f"Asynchronous metric writer exited with code {self.process.exitcode}")

    def _drain_stats(self) -> None:
        if self.stats_queue is None:
            return
        while True:
            try:
                elapsed = self.stats_queue.get_nowait()
            except Empty:
                return
            self.write_seconds += elapsed
            self.written_batches += 1

    def _queue_depth(self) -> int:
        if self.queue is None:
            return 0
        try:
            return self.queue.qsize()
        except NotImplementedError:
            return 0


class MetricLogger:
    def __init__(
        self,
        outdir: Path,
        tag: Optional[str] = None,
        enable: bool = False,
        log_tb: bool = False,
        log_wandb: bool = False,
        enable_prefix_keys: bool = True,
        async_writer: Optional[AsyncMetricWriter] = None,
    ):
        self.outdir = outdir
        self.tag = tag
        self.enable = enable
        self.log_tb = log_tb
        self.log_wandb = log_wandb and has_wandb
        self.enable_prefix_keys = enable_prefix_keys
        self.async_writer = async_writer

        self.tb_writer: Optional[SummaryWriter] = None
        self.jsonl_writer: Optional[TextIO] = None

        if not enable:
            assert not log_tb
            return

        if async_writer is not None:
            async_writer.register(outdir=outdir, tag=tag, log_tb=log_tb, log_wandb=self.log_wandb)
            return

        outdir.mkdir(exist_ok=True, parents=True)
        if log_tb:
            tb_dir = outdir / "tb"
            tb_dir.mkdir(exist_ok=True, parents=False)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir), max_queue=1000)

        fname = f"metrics.{tag}.jsonl" if tag else "metrics.jsonl"
        fpath = outdir / fname
        self.jsonl_writer = open(fpath, "a")

    def log(self, metrics: Dict[str, Any], step: int, dashboard_metrics: Optional[Dict[str, Any]] = None):
        """Write ``metrics`` to the jsonl, and ``dashboard_metrics`` (default: ``metrics``) to TensorBoard / W&B."""
        if not self.enable:
            return

        # check / sanitize inputs
        assert all(type(k) is str for k in metrics.keys())
        for k, v in (metrics if dashboard_metrics is None else dashboard_metrics).items():
            assert not isinstance(v, dict), f"{k} {v}"

        record = _MetricRecord(
            tag=self.tag,
            metrics=dict(metrics),
            dashboard_metrics=None if dashboard_metrics is None else dict(dashboard_metrics),
            enable_prefix_keys=self.enable_prefix_keys,
            step=step,
            created_at=datetime.utcnow().isoformat(),
            log_tb=self.log_tb,
            log_wandb=self.log_wandb,
        )
        if self.async_writer is not None:
            self.async_writer.submit(record)
            return

        prefixed_metrics = _prefixed_metrics(record)

        # log in tensorboard
        if self.tb_writer is not None and prefixed_metrics:
            summary = Summary(value=[
                Summary.Value(tag=k, simple_value=float(v))
                for k, v in prefixed_metrics.items()
            ])
            self.tb_writer.file_writer.add_summary(summary, global_step=step)

        # log in wandb
        if self.log_wandb and prefixed_metrics:
            wandb.log(prefixed_metrics, step=step)

        json_metrics = dict(global_step=step, created_at=record.created_at)
        json_metrics.update(metrics)
        print(json.dumps(json_metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None


def build_mlogger(
    outdir: str,
    tag: str,
    enable: bool,
    enable_wandb: bool = False,
    enable_prefix_keys: bool = True,
    async_writer: Optional[AsyncMetricWriter] = None,
) -> MetricLogger:
    return MetricLogger(
        outdir=Path(outdir), tag=tag, enable=enable, log_tb=enable,
        log_wandb=enable_wandb, enable_prefix_keys=enable_prefix_keys,
        async_writer=async_writer,
    )
