import json
import math
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tests.health.helpers as helpers
import xllm.distributed.initialize as parallel_state
import xllm.health_metrics as health_metrics
from tests.health.helpers import MASK, MATRICES, TARGETS, TOKENS, logged_metrics, shard, tiny_transformer
from xllm.health_metrics import HealthMetricCollector, get_active_health_collector
from xllm.metrics import MetricLogger
from xllm.modules.rms_norm import GroupRMSNorm

STATS = ("mean", "min", "max")
ALL_FLAGS = dict(enable_query_key_stats=True, enable_logits_stats=True, enable_weight_norms=True, enable_gradient_norms=True)


def _step(collector, step, layers=None):
    """One step: begin, record an activation per layer, end forward, flush. Returns (was active, logged)."""
    logged = []
    collector.begin_step(step)
    active = get_active_health_collector() is collector
    for layer_idx in range(collector.num_layers) if layers is None else layers:
        collector.record_activation(layer_idx, torch.tensor([float(step)]))
    collector.end_forward()
    assert get_active_health_collector() is None
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.append((step, metrics))), step)
    return active, logged


# Sampling and lifecycle

@pytest.mark.parametrize("enabled, freq, sampled", [(True, 1, {1, 2, 3, 4, 5, 6}), (True, 3, {3, 6}), (False, 1, set())])
def test_only_every_freq_th_step_is_recorded_and_logged(enabled, freq, sampled):
    collector = HealthMetricCollector(enabled=enabled, freq=freq, num_layers=1)
    for step in range(1, 7):
        active, logged = _step(collector, step)
        assert active == (step in sampled)
        assert logged == ([(step, {f"activation_stats/layer_000/{s}": float(step) for s in STATS})] if step in sampled else [])


def test_flush_logs_once_per_sampled_step():
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1)
    logged = []
    logger = SimpleNamespace(log=lambda metrics, step: logged.append(step))
    collector.flush(logger, 0)  # nothing begun
    _step(collector, 1)
    collector.flush(logger, 1)  # already flushed
    assert logged == []


def test_begin_step_discards_an_abandoned_step():
    # A loss-scale overflow repeats the step without flushing; the retry must start clean.
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1)
    collector.begin_step(1)
    collector.record_activation(0, torch.tensor([100.0]))
    collector.end_forward()

    _, logged = _step(collector, 1)

    assert logged[0][1]["activation_stats/layer_000/mean"] == 1.0


# Record validation

ONES = torch.ones(2, 2)
HEADS = torch.ones(1, 1, 1, 2)
SCALAR = torch.tensor(1.0)


@pytest.mark.parametrize("flags, record, error", [
    ({}, lambda c: [c.record_activation(0, ONES) for _ in range(2)], "Duplicate activation"),
    ({}, lambda c: c.record_activation(0, torch.empty(0)), "Empty activation"),
    ({"enable_query_key_stats": True}, lambda c: [c.record_query_key_stats(0, ONES, ONES) for _ in range(2)], "Duplicate query/key"),
    ({"enable_query_key_stats": True}, lambda c: c.record_query_key_stats(0, torch.empty(0), ONES), "Empty query/key"),
    ({"enable_query_key_stats": True}, lambda c: c.record_query_key_stats(0, SCALAR, SCALAR), "head dimension"),
    ({"enable_logits_stats": True}, lambda c: [c.record_logits_stats(SCALAR, SCALAR, SCALAR, SCALAR) for _ in range(2)], "Duplicate logit"),
    ({"enable_logits_stats": True}, lambda c: c.record_logits_stats(ONES, SCALAR, SCALAR, SCALAR), "must be scalars"),
])
def test_malformed_records_are_rejected(flags, record, error):
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, **flags)
    collector.begin_step(1)
    with pytest.raises(RuntimeError, match=error):
        record(collector)


@pytest.mark.parametrize("gradient_scale", [0.0, -1.0, math.inf, math.nan])
def test_gradient_scale_must_be_finite_and_positive(gradient_scale):
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_gradient_norms=True)
    collector.begin_step(1)
    with pytest.raises(ValueError, match="gradient_scale"):
        collector.record_matrix_norms(None, gradient_scale=gradient_scale)  # rejected before the model is read


@pytest.mark.usefixtures("single_process_group")
def test_matrix_norms_are_recorded_once_per_step():
    model = shard(tiny_transformer(1), "fsdp2")
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_weight_norms=True)
    collector.begin_step(1)
    collector.record_matrix_norms(model)
    with pytest.raises(RuntimeError, match="Duplicate matrix norm records"):
        collector.record_matrix_norms(model)


# Steps and families that are off

def test_unsampled_steps_ignore_every_record_call():
    collector = HealthMetricCollector(enabled=True, freq=2, num_layers=1, enable_weight_norms=True, **{k: v for k, v in ALL_FLAGS.items() if k != "enable_weight_norms"})
    collector.begin_step(1)  # not sampled: these would all be errors on a sampled step
    collector.record_activation(0, torch.empty(0))
    collector.record_query_key_stats(0, SCALAR, SCALAR)
    collector.record_logits_stats(None, None, None, None)
    collector.record_matrix_norms(object(), gradient_scale=0.0)
    logged = []
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.append(metrics)), 1)
    assert logged == [] and collector.decoder_layer_activation_records == {}


def test_families_that_are_off_are_ignored():
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1)  # activations only
    collector.begin_step(1)
    collector.record_activation(0, torch.ones(1))
    collector.record_query_key_stats(0, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))
    collector.record_logits_stats(SCALAR, SCALAR, SCALAR, SCALAR)
    collector.record_matrix_norms(object())
    collector.end_forward()
    logged = {}
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.update(metrics)), 1)
    assert set(logged) == {f"activation_stats/layer_000/{s}" for s in STATS}


# Incomplete steps are skipped, together, with a warning

@pytest.mark.parametrize("flags, extra_records, flush_step, message", [
    ({}, lambda c: None, 1, "Expected 2 activation records"),  # layer 1 never recorded
    ({"enable_query_key_stats": True}, lambda c: c.record_activation(1, ONES), 1, "Expected 2 query/key records"),
    ({"enable_query_key_stats": True}, lambda c: (c.record_activation(1, ONES), c.record_query_key_stats(0, HEADS, HEADS, kq=HEADS),
                                                  c.record_query_key_stats(1, HEADS, HEADS)), 1, "cover layers [0] only"),  # kq on some layers
    ({"enable_logits_stats": True}, lambda c: c.record_activation(1, ONES), 1, "Missing logit statistics"),
    ({"enable_weight_norms": True}, lambda c: c.record_activation(1, ONES), 1, "matrix norm records"),  # norms never recorded
    ({}, lambda c: c.record_activation(1, ONES), 2, "at step 2"),  # flushed with the wrong step
])
def test_an_incomplete_step_is_skipped_with_a_warning(caplog, flags, extra_records, flush_step, message):
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=2, **flags)
    collector.begin_step(1)
    collector.record_activation(0, ONES)
    extra_records(collector)
    collector.end_forward()
    logged = []

    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.append(step)), flush_step)

    assert logged == [] and not collector.recording and collector.decoder_layer_activation_records == {}
    assert "Skipping health metrics" in caplog.text and message in caplog.text


def test_a_skipped_step_does_not_affect_the_next_one():
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=2)
    _, skipped = _step(collector, 1, layers=[0])
    _, logged = _step(collector, 2)
    assert skipped == [] and [step for step, _ in logged] == [2]


# Everything on: one agreement collective, three packed reductions, the documented names, plain floats

def _recording(name, calls):
    original = getattr(dist, name)

    def wrapper(tensor, *args, **kwargs):
        calls.append((name, kwargs.get("op"), tensor.numel()))
        return original(tensor, *args, **kwargs)
    return wrapper


@pytest.mark.usefixtures("single_process_group")
def test_flush_uses_one_agreement_and_three_packed_reductions(monkeypatch):
    calls = []
    for name in ("all_reduce", "reduce"):
        monkeypatch.setattr(health_metrics.dist, name, _recording(name, calls))
    model = shard(tiny_transformer(2), "fsdp2")
    calls.clear()

    metrics = logged_metrics(model, TOKENS, TARGETS, MASK, loss_scale=1.0, **ALL_FLAGS)

    entries = 2 * 5  # per layer: activations, q, q head-L2, k, k head-L2
    norms = 2 * 2 * len(MATRICES)  # weight and gradient norms
    assert [(name, op) for name, op, _ in calls] == [
        ("all_reduce", dist.ReduceOp.MAX), ("reduce", dist.ReduceOp.SUM), ("reduce", dist.ReduceOp.MIN), ("reduce", dist.ReduceOp.MAX),
    ]
    assert [size for _, _, size in calls] == [1, 2 * entries + 2 + norms, entries + 1, entries + 1]  # sums+counts (+logits), extrema
    assert len(metrics) == 3 * entries + 3 + norms


@pytest.mark.usefixtures("single_process_group")
def test_everything_enabled_logs_the_documented_metric_names_as_floats():
    model = shard(tiny_transformer(2), "fsdp2")

    metrics = logged_metrics(model, TOKENS, TARGETS, MASK, loss_scale=1.0, **ALL_FLAGS)

    expected = {f"logits_stats/{s}" for s in STATS}
    for layer in ("layer_000", "layer_001"):
        expected |= {f"activation_stats/{layer}/{s}" for s in STATS}
        expected |= {f"{qk}_stats/{layer}/{s}" for qk in ("q", "k") for s in (*STATS, "vector_l2_mean", "vector_l2_min", "vector_l2_max")}
        expected |= {f"{family}/{layer}/{path}/l2" for family in ("weight_norms", "gradient_norms") for path in MATRICES}
    assert set(metrics) == expected
    assert all(type(value) is float for value in metrics.values())


def test_checkpoint_recompute_does_not_record_twice():
    model = tiny_transformer(2, layerwise_ckpt=True)
    forwards = []
    for layer in model.layers:
        layer.attention.causal_attention.register_forward_hook(lambda module, args, output: forwards.append(module))

    metrics = logged_metrics(model, loss_scale=1.0, enable_query_key_stats=True)  # backward recomputes every layer

    assert len(forwards) == 2 * len(model.layers)
    assert {name for name in metrics if name.startswith("q_stats/")} == {f"q_stats/layer_{i:03d}/{s}" for i in range(2) for s in (*STATS, "vector_l2_mean", "vector_l2_min", "vector_l2_max")}


def test_flush_writes_one_jsonl_row_through_a_real_metric_logger(tmp_path):
    mlogger = MetricLogger(outdir=tmp_path, tag="health", enable=True, log_tb=False, enable_prefix_keys=False)
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1)
    collector.begin_step(7)
    collector.record_activation(0, torch.tensor([1.0, 3.0]))
    collector.end_forward()

    collector.flush(mlogger, 7)
    mlogger.close()

    rows = [json.loads(line) for line in (tmp_path / "metrics.health.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["global_step"] == 7 and rows[0]["activation_stats/layer_000/mean"] == 2.0


# Two real ranks: statistics reduce globally, only rank 0 logs, and a bad rank makes every rank skip

def _rank(rank, world_size, rendezvous, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    parallel_state._DATA_PARALLEL_GROUP = dist.group.WORLD
    GroupRMSNorm.forward = helpers.cpu_rms_norm_forward
    result = {"logged": []}
    logger = SimpleNamespace(log=lambda metrics, step: result["logged"].append((step, metrics)))
    try:
        collector = HealthMetricCollector(
            enabled=True, freq=1, num_layers=1, enable_logits_stats=True, enable_weight_norms=True,
            router_stats_layers={"moe": [0]},
        )
        model = tiny_transformer(1)
        result["full_norms"] = {f"weight_norms/layer_000/{p}/l2": model.layers[0].get_parameter(f).norm().item() for p, f in MATRICES.items()}
        model = shard(model, "fsdp2")
        activations = {1: ([1.0, 3.0], [-5.0, 9.0, 10.0]), 2: ([1.0], None), 3: ([1.0], [1.0])}  # step -> per-rank
        logits = {1: ((6.0, 3, -1.0, 4.0), (2.0, 1, 0.0, 9.0))}  # step -> per-rank (sum, count, min, max)
        routing = (  # per-rank (loads, top-k scores, aux loss)
            ([3, 1, 0, 4], [[0.75, 0.25], [0.5, 0.5]], 0.3),
            ([1, 1, 0, 0], [[1.0, 0.0]], 0.5),
        )
        for step in (1, 2, 3):
            collector.begin_step(step)
            if activations[step][rank] is not None:  # rank 1 drops its record at step 2
                collector.record_activation(0, torch.tensor(activations[step][rank]))
            collector.record_logits_stats(*(torch.tensor(v) for v in logits.get(step, ((1.0, 1, 1.0, 1.0),) * 2)[rank]))
            loads, scores, aux_loss = routing[rank]
            collector.record_router_stats("moe", 0, torch.tensor(loads), torch.tensor(scores), aux_loss=torch.tensor(aux_loss))
            collector.end_forward()
            collector.record_matrix_norms(model)
            collector.flush(logger, step)
    finally:
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
        dist.destroy_process_group()


def test_two_ranks_reduce_globally_log_on_rank_zero_and_skip_together(tmp_path):
    ranks = mp.spawn(_rank, args=(2, str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=False)
    deadline = time.monotonic() + 120  # join returns as each rank exits; a lone rank in a reduction never does
    while not ranks.join(timeout=max(0.0, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in ranks.processes:
                process.kill()
            pytest.fail("the ranks did not finish: a collective is probably hanging")
    rank0, rank1 = (json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1))

    assert rank1["logged"] == []
    assert [step for step, _ in rank0["logged"]] == [1, 3]  # step 2 skipped on both ranks, no hang
    step1 = rank0["logged"][0][1]
    assert [step1[f"activation_stats/layer_000/{s}"] for s in STATS] == pytest.approx([18 / 5, -5.0, 10.0])
    assert [step1[f"logits_stats/{s}"] for s in STATS] == pytest.approx([8 / 4, -1.0, 9.0])
    assert {k: v for k, v in step1.items() if k.startswith("weight_norms/")} == pytest.approx(rank0["full_norms"], rel=1e-5)
    # Router loads sum to [4, 2, 0, 4]; entropy averages over all 3 tokens; aux loss averages over ranks.
    entropy = lambda p: -sum(x * math.log(x) for x in p if x > 0)  # noqa: E731
    assert {k: v for k, v in step1.items() if k.startswith("moe_")} == pytest.approx({
        "moe_utilization_stats/layer_000/load_max_over_mean": 4 / 2.5,
        "moe_utilization_stats/layer_000/load_entropy_normalized": entropy([0.4, 0.2, 0.4]) / math.log(4),
        "moe_utilization_stats/layer_000/zero_experts": 1.0,
        "moe_router_confidence_stats/layer_000/topk_entropy_mean": (entropy([0.75, 0.25]) + math.log(2)) / 3,
        "moe_auxiliary_loss/layer_000/value": 0.4,
    })
