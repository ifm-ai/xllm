"""RMSNorm gain statistics: every gain vector in the model, summarised as RMS and its extreme channels."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.health.helpers import logged_metrics, shard, stats, tiny_transformer
from xllm.metrics.health import HealthMetricCollector, _norm_gain_path
from xllm.modules.rms_norm import GroupRMSNorm

MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=8, num_dense_layers=1)
MOVA = dict(num_values=4, num_activated_values=2)  # MoVA layers carry an MoE too
STATS = ("rms", "min", "max")


def gain_metrics(model, **kwargs):
    return {
        name: value for name, value in logged_metrics(model, enable_norm_gains=True, **kwargs).items()
        if name.startswith("norm_gains/")
    }


def expected(module):
    """The gains the model applies are the stored offsets plus one."""
    gains = module.weight.detach() + 1.0
    return [gains.square().mean().sqrt().item(), gains.min().item(), gains.max().item()]


def norm_modules(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, GroupRMSNorm) and m.weight is not None}


# Which gains get logged

@pytest.mark.parametrize("overrides, paths", [
    ({}, ["layer_000/attention/norm", "layer_000/nffn/norm", "layer_001/attention/norm",
          "layer_001/nffn/norm", "final_norm"]),
    ({"qknorm": True}, ["layer_000/attention/norm", "layer_000/attention/query_norm",
                        "layer_000/attention/key_norm", "layer_000/nffn/norm",
                        "layer_001/attention/norm", "layer_001/attention/query_norm",
                        "layer_001/attention/key_norm", "layer_001/nffn/norm", "final_norm"]),
    (MOE, ["layer_000/attention/norm", "layer_000/nffn/norm", "layer_001/attention/norm",
           "layer_001/moe/norm", "final_norm"]),
    (MOE | MOVA, ["layer_000/attention/norm", "layer_000/nffn/norm", "layer_001/mova/norm",
                  "layer_001/moe/norm", "final_norm"]),
], ids=["dense", "qknorm", "moe", "mova"])
def test_every_gain_in_the_model_is_logged(overrides, paths):
    model = tiny_transformer(2, **overrides)

    metrics = gain_metrics(model)

    assert set(metrics) == {f"norm_gains/{path}/{s}" for path in paths for s in STATS}
    assert set(paths) == {_norm_gain_path(name) for name in norm_modules(model)}


def test_paths_name_the_module_that_owns_the_gain():
    assert _norm_gain_path("layers.7.attention.norm") == "layer_007/attention/norm"
    assert _norm_gain_path("layers.12.moe.norm") == "layer_012/moe/norm"
    assert _norm_gain_path("output.final_norm") == "final_norm"


# What the values are

def test_gains_are_the_stored_offset_plus_one():
    model = tiny_transformer(1)
    with torch.no_grad():  # offsets of -0.5 and 2.0 are gains of 0.5 and 3.0
        model.layers[0].attention.norm.weight.copy_(torch.tensor([-0.5] * 4 + [2.0] * 4))

    metrics = gain_metrics(model)

    assert stats(metrics, "norm_gains/layer_000/attention/norm", STATS) == pytest.approx(
        [math.sqrt((4 * 0.25 + 4 * 9.0) / 8), 0.5, 3.0]
    )


def test_untrained_gains_are_one():
    model = tiny_transformer(2)  # every offset starts at zero

    metrics = gain_metrics(model)

    assert all(value == pytest.approx(1.0) for value in metrics.values())


@pytest.mark.parametrize("overrides", [{}, {"qknorm": True}, MOE | MOVA])
def test_every_gain_matches_its_own_module(overrides):
    model = tiny_transformer(2, **overrides)
    with torch.no_grad():  # a distinct spread per norm, so no two gains agree by accident
        for i, module in enumerate(norm_modules(model).values()):
            module.weight.copy_(torch.linspace(-0.4 * (i + 1), 0.5 * (i + 1), module.weight.numel()))

    metrics = gain_metrics(model)

    for name, module in norm_modules(model).items():
        assert stats(metrics, f"norm_gains/{_norm_gain_path(name)}", STATS) == pytest.approx(
            expected(module), rel=1e-6
        )


def test_a_single_runaway_channel_shows_in_the_maximum():
    model = tiny_transformer(1)
    with torch.no_grad():
        model.layers[0].nffn.norm.weight[3] = 9.0  # one channel of eight

    metrics = gain_metrics(model)

    rms, minimum, maximum = stats(metrics, "norm_gains/layer_000/nffn/norm", STATS)
    assert maximum == pytest.approx(10.0) and minimum == pytest.approx(1.0)
    assert rms == pytest.approx(math.sqrt((7 * 1.0 + 100.0) / 8))


# Sharding and lifecycle

@pytest.mark.usefixtures("single_process_group")
def test_sharded_gains_log_the_same_values():
    model = tiny_transformer(2)
    with torch.no_grad():
        for i, module in enumerate(norm_modules(model).values()):
            module.weight.copy_(torch.linspace(-0.3, 0.7 * (i + 1), module.weight.numel()))
    unsharded = gain_metrics(model)

    metrics = gain_metrics(shard(model, "fsdp2"))

    assert metrics == pytest.approx(unsharded, rel=1e-6)


def test_a_rank_with_no_elements_of_a_gain_contributes_nothing():
    model = tiny_transformer(1)
    with torch.no_grad():  # a gain sharded over more ranks than it has channels
        next(iter(norm_modules(model).values())).weight = torch.nn.Parameter(torch.zeros(0))
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_norm_gains=True)
    collector.begin_step(1)
    collector.record_activation(0, torch.ones(2))
    collector.record_norm_gains(model)  # no forward: a zero-length gain cannot be applied
    collector.end_forward()
    logged = {}

    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.update(metrics)), 1)

    # Neutral extrema and an empty count, so the other gains are untouched and the step still logs.
    assert math.isnan(logged["norm_gains/layer_000/attention/norm/rms"])
    assert logged["norm_gains/layer_000/attention/norm/min"] == float("inf")
    assert logged["norm_gains/layer_000/nffn/norm/rms"] == pytest.approx(1.0)


def test_gains_are_not_collected_when_disabled():
    metrics = logged_metrics(tiny_transformer(1))

    assert not any(name.startswith("norm_gains/") for name in metrics)


def test_a_second_record_in_one_step_is_an_error():
    model = tiny_transformer(1)
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_norm_gains=True)
    collector.begin_step(1)
    collector.record_norm_gains(model)

    with pytest.raises(RuntimeError, match="Duplicate norm gain records"):
        collector.record_norm_gains(model)


def test_a_step_that_never_records_gains_is_skipped(caplog):
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_norm_gains=True)
    collector.begin_step(1)
    collector.record_activation(0, torch.ones(2))
    collector.end_forward()
    logged = []

    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.append(metrics)), 1)

    assert logged == [] and "Missing norm gain records" in caplog.text
