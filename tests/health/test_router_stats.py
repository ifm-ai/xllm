"""Router statistics shared by MoE and MoVA, checked by hand on a real collector without a model."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import xllm.distributed.initialize as parallel_state
import xllm.health_metrics as health_metrics
from xllm.health_metrics import HealthMetricCollector

pytestmark = pytest.mark.usefixtures("single_process_group")

LOADS = torch.tensor([3, 1, 0, 4])
SCORES = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
BIAS = torch.tensor([-1.0, 0.0, 1.0, 2.0])
BASE = dict(tokens_per_route=LOADS, topk_scores=SCORES)


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


TOPK_ENTROPY = (entropy([0.75, 0.25]) + math.log(2)) / 2


def flush(records, layers=None, num_layers=2, after_recording=None):
    """One sampled step with an activation per layer; records = [(family, layer, kwargs)]. Returns the logged dict or None."""
    collector = HealthMetricCollector(True, 1, num_layers, router_stats_layers={"moe": [1]} if layers is None else layers)
    collector.begin_step(0)
    for layer_idx in range(num_layers):
        collector.record_activation(layer_idx, torch.ones(2))
    for family, layer_idx, kwargs in records:
        collector.record_router_stats(family, layer_idx, **kwargs)
    collector.end_forward()
    if after_recording is not None:
        after_recording()
    logged = []
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.append(metrics)), 0)
    return logged[0] if logged else None


def test_hand_computed_metrics_for_one_layer():
    metrics = flush([("moe", 1, BASE | dict(router_bias=BIAS, aux_loss=torch.tensor(0.3)))])

    assert {k: v for k, v in metrics.items() if k.startswith("moe")} == pytest.approx({
        "moe_utilization_stats/layer_001/load_max_over_mean": 2.0,
        "moe_utilization_stats/layer_001/load_entropy_normalized": entropy([3 / 8, 1 / 8, 4 / 8]) / math.log(4),
        "moe_utilization_stats/layer_001/zero_experts": 1.0,
        "moe_router_confidence_stats/layer_001/topk_entropy_mean": TOPK_ENTROPY,
        "moe_router_bias_stats/layer_001/mean": 0.5,
        "moe_router_bias_stats/layer_001/rms": math.sqrt(1.5),
        "moe_router_bias_stats/layer_001/min": -1.0,
        "moe_router_bias_stats/layer_001/max": 2.0,
        "moe_router_bias_stats/layer_001/max_abs": 2.0,
        "moe_auxiliary_loss/layer_001/value": 0.3,
    })
    assert all(type(value) is float for value in metrics.values())


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_scaling_factor_cancels_and_top1_has_zero_entropy(scale):
    name = "moe_router_confidence_stats/layer_001/topk_entropy_mean"
    assert flush([("moe", 1, BASE | dict(topk_scores=SCORES * scale))])[name] == pytest.approx(TOPK_ENTROPY)
    assert flush([("moe", 1, BASE | dict(topk_scores=SCORES[:, :1] * scale))])[name] == 0.0


def test_scores_with_leading_batch_dimensions_count_every_token():
    batched = torch.stack((SCORES, SCORES[[1, 0]]))  # B x L x K, as the router returns them
    metrics = flush([("moe", 1, BASE | dict(topk_scores=batched))])
    assert metrics["moe_router_confidence_stats/layer_001/topk_entropy_mean"] == pytest.approx(TOPK_ENTROPY)


def test_balanced_loads_have_ratio_one_and_full_entropy():
    metrics = flush([("moe", 1, BASE | dict(tokens_per_route=torch.tensor([2, 2, 2, 2])))])
    assert metrics["moe_utilization_stats/layer_001/load_max_over_mean"] == pytest.approx(1.0)
    assert metrics["moe_utilization_stats/layer_001/load_entropy_normalized"] == pytest.approx(1.0)
    assert metrics["moe_utilization_stats/layer_001/zero_experts"] == 0.0


def test_optional_parts_are_omitted_when_absent():
    metrics = flush([("moe", 1, BASE)])
    assert not any("bias" in name or "auxiliary" in name for name in metrics)


def test_moe_and_mova_log_under_their_own_names_in_one_step():
    metrics = flush(
        [("moe", 1, BASE), ("mova", 1, BASE | dict(tokens_per_route=torch.tensor([0, 0, 5, 1])))],
        layers={"moe": [1], "mova": [1]},
    )
    assert metrics["moe_utilization_stats/layer_001/zero_experts"] == 1.0
    assert metrics["mova_value_utilization/layer_001/dead_values"] == 2.0
    assert metrics["mova_value_utilization/layer_001/load_max_over_mean"] == pytest.approx(5 / 1.5)
    assert metrics["mova_router_confidence_stats/layer_001/topk_entropy_mean"] == pytest.approx(TOPK_ENTROPY)


def test_families_that_are_off_ignore_records():
    metrics = flush([("moe", 1, BASE)], layers={})
    assert not any(name.startswith("moe") for name in metrics)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_in_place_changes_after_recording_do_not_leak_into_the_log(dtype):
    loads, bias = LOADS.to(dtype), BIAS.clone()

    def backward():  # what the router's backward does: all-reduce the loads, then update the bias
        loads.mul_(8)
        bias.add_(1)

    metrics = flush([("moe", 1, BASE | dict(tokens_per_route=loads, router_bias=bias))], after_recording=backward)
    assert metrics["moe_utilization_stats/layer_001/zero_experts"] == 1.0
    assert metrics["moe_utilization_stats/layer_001/load_max_over_mean"] == 2.0
    assert metrics["moe_router_bias_stats/layer_001/mean"] == 0.5


def test_other_model_parallel_ranks_contribute_nothing(monkeypatch):
    # Every model-parallel rank sees the same routing, so only rank 0 counts it. Alone here, the
    # loads reduce to zero: logged as NaN, never as invented numbers.
    monkeypatch.setattr(parallel_state, "_MODEL_PARALLEL_RANK", 1)
    metrics = flush([("moe", 1, BASE | dict(aux_loss=torch.tensor(0.3)))])
    for name in ("moe_utilization_stats/layer_001/load_max_over_mean",
                 "moe_router_confidence_stats/layer_001/topk_entropy_mean",
                 "moe_auxiliary_loss/layer_001/value"):
        assert math.isnan(metrics[name])
    assert metrics["moe_utilization_stats/layer_001/zero_experts"] == 4.0


@pytest.mark.parametrize("bad, error", [
    (dict(tokens_per_route=LOADS.view(2, 2)), "1-D"),
    (dict(tokens_per_route=LOADS[:0]), "1-D"),
    (dict(topk_scores=SCORES[:0]), "Empty"),
    (dict(aux_loss=torch.tensor([0.1, 0.2])), "scalar"),
])
def test_malformed_records_raise(bad, error):
    with pytest.raises(RuntimeError, match=error):
        flush([("moe", 1, BASE | bad)])


def test_duplicate_layer_raises():
    with pytest.raises(RuntimeError, match="Duplicate"):
        flush([("moe", 1, BASE), ("moe", 1, BASE)])


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="Unknown router statistics families"):
        HealthMetricCollector(True, 1, 1, router_stats_layers={"moa": [0]})


@pytest.mark.parametrize("records, message", [
    ([], "Expected moe router statistics"),  # layer 2 never recorded
    ([("moe", 2, dict(aux_loss=torch.tensor(0.1)))], "auxiliary loss recorded for only some layers"),
])
def test_incomplete_or_inconsistent_steps_are_skipped_with_a_warning(caplog, records, message):
    records = [("moe", 1, BASE)] + [(family, layer, BASE | kwargs) for family, layer, kwargs in records]
    assert flush(records, layers={"moe": [1, 2]}, num_layers=3) is None
    assert message in caplog.text


def test_router_stats_ride_in_the_existing_sum_packet(monkeypatch):
    calls = []
    for name in ("all_reduce", "reduce"):
        original = getattr(dist, name)
        monkeypatch.setattr(health_metrics.dist, name, lambda tensor, *a, _f=original, _n=name, **kw: (
            calls.append((_n, kw.get("op"), tensor.numel())), _f(tensor, *a, **kw))[1])

    flush([("moe", 1, BASE | dict(aux_loss=torch.tensor(0.3)))])

    router_packet = len(LOADS) + 2 + 2  # loads, entropy sum and token count, aux sum and count
    assert calls == [
        ("all_reduce", dist.ReduceOp.MAX, 1),
        ("reduce", dist.ReduceOp.SUM, 2 * 2 + router_packet),  # two activation entries: sums and counts
        ("reduce", dist.ReduceOp.MIN, 2),
        ("reduce", dist.ReduceOp.MAX, 2),
    ]
