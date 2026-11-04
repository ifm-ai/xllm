import math
from types import SimpleNamespace

import pytest
import torch

from tests.health.helpers import stats
from xllm.config import ModelConf
from xllm.distributed import FullyShardedDataParallel
from xllm.health_metrics import HealthMetricCollector
from xllm.models.gekko import GekkoBlock

# The Gekko hooks run inside fused CUDA kernels, so the statistics tests feed the collector exactly the
# tensors the hooks pass, shaped (batch, seq, heads, head_dim); the norm tests use a real GekkoBlock.

PREFIX = "gekko_stats/layer_000"
STATS = ("mean", "min", "max")


def _collector(num_layers=1, **flags):
    flags.setdefault("enable_gekko_stats", True)
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=num_layers, **flags)
    collector.begin_step(1)
    return collector


def _flush(collector):
    for layer_idx in range(collector.num_layers):
        collector.record_activation(layer_idx, torch.ones(1))
    collector.end_forward()
    logged = []
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.append(metrics)), 1)
    return logged[0] if logged else None


# One layer's hook inputs, chosen so every derived metric is hand-computable.
Q = torch.tensor([[[[3.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]])  # raw queries: head L2 5 and 1
K = torch.tensor([[[[0.0, 0.0, 0.0, 2.0], [-1.0, 0.0, 0.0, 0.0]]]])  # raw keys
SQ = torch.tensor([[[[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]]])  # normed queries: head L2 1 and 2
SK_L2 = torch.tensor([[[1.0, 3.0]]])  # pre-RoPE normed key L2 per head
RATIOS = (torch.tensor([0.25, 0.75]), torch.tensor([0.5]))  # one call per AWM memory update
SCA = torch.full((1, 2, 8), 3.0)  # (batch, seq, heads * v): RMS 3
AWK = torch.full((1, 2, 8), 4.0)  # RMS 4
GATE = torch.tensor([[[0.0, 0.0005, 0.5, -2.0]]])  # 2 of 4 within 1e-3 of zero
UNAVAILABLE = torch.tensor([[True, False, False, False]])  # history missing at 1 of 4 positions


def _record_layer(collector, layer_idx=0, ratios=RATIOS, sca=SCA, awk=AWK):
    collector.record_gekko_query_key_stats(layer_idx, Q, K, SQ, sca_key_l2=SK_L2)
    for ratio in ratios:
        collector.record_gekko_update_ratio(layer_idx, ratio)
    collector.record_gekko_branches(layer_idx, sca, awk, GATE, UNAVAILABLE)


# Statistics

def test_gekko_stats_turn_query_key_stats_on():
    assert HealthMetricCollector(True, 1, 1, enable_gekko_stats=True).enable_query_key_stats
    assert not HealthMetricCollector(True, 1, 1).enable_query_key_stats


def test_raw_and_normed_query_key_norms():
    collector = _collector()
    _record_layer(collector)

    metrics = _flush(collector)

    assert stats(metrics, "q_stats/layer_000") == pytest.approx([1.0, 0.0, 4.0])
    assert stats(metrics, "q_stats/layer_000", ("vector_l2_mean", "vector_l2_min", "vector_l2_max")) == [3.0, 1.0, 5.0]
    assert stats(metrics, "k_stats/layer_000") == pytest.approx([1 / 8, -1.0, 2.0])
    assert stats(metrics, f"{PREFIX}/sca_query_l2") == pytest.approx([1.5, 1.0, 2.0])
    assert stats(metrics, f"{PREFIX}/sca_key_l2") == pytest.approx([2.0, 1.0, 3.0])


def test_branch_energy_gate_and_history_metrics():
    collector = _collector()
    _record_layer(collector)

    metrics = _flush(collector)

    assert metrics[f"{PREFIX}/sca_rms"] == pytest.approx(3.0)
    assert metrics[f"{PREFIX}/awk_rms"] == pytest.approx(4.0)
    assert metrics[f"{PREFIX}/awk_energy_fraction"] == pytest.approx(16 / 25)
    assert stats(metrics, f"{PREFIX}/output_gate") == pytest.approx([-1.4995 / 4, -2.0, 0.5])
    assert metrics[f"{PREFIX}/output_gate/near_zero_fraction"] == 0.5
    assert metrics[f"{PREFIX}/awk_history_available_fraction"] == 0.75


@pytest.mark.parametrize("ratios, expected", [
    (RATIOS, [0.5, 0.25, 0.75, 3]),
    ((), [math.nan, math.nan, math.nan, 0]),  # cold start: no memory update on this rank
])
def test_awk_update_ratio_accumulates_across_updates(ratios, expected):
    collector = _collector()
    _record_layer(collector, ratios=ratios)

    metrics = _flush(collector)

    assert stats(metrics, f"{PREFIX}/awk_update_ratio", (*STATS, "count")) == pytest.approx(expected, nan_ok=True)


def test_gekko_metric_names():
    collector = _collector()
    _record_layer(collector)

    metrics = _flush(collector)

    assert {name[len(PREFIX) + 1:] for name in metrics if name.startswith(PREFIX)} == {
        *(f"sca_query_l2/{s}" for s in STATS), *(f"sca_key_l2/{s}" for s in STATS),
        *(f"output_gate/{s}" for s in STATS), "output_gate/near_zero_fraction",
        "sca_rms", "awk_rms", "awk_energy_fraction", "awk_history_available_fraction",
        *(f"awk_update_ratio/{s}" for s in (*STATS, "count")),
    }


def test_each_layer_keeps_its_own_gekko_stats():
    collector = _collector(num_layers=2)
    _record_layer(collector, 0)
    _record_layer(collector, 1, sca=SCA * 2, awk=AWK * 4)

    metrics = _flush(collector)

    assert (metrics["gekko_stats/layer_000/sca_rms"], metrics["gekko_stats/layer_001/sca_rms"]) == pytest.approx((3.0, 6.0))
    assert metrics["gekko_stats/layer_001/awk_energy_fraction"] == pytest.approx(256 / (36 + 256))


def test_gekko_hooks_only_record_query_key_when_gekko_stats_are_off():
    collector = _collector(enable_gekko_stats=False, enable_query_key_stats=True)
    _record_layer(collector)

    metrics = _flush(collector)

    assert "q_stats/layer_000/mean" in metrics
    assert not any(name.startswith("gekko_stats/") for name in metrics)


def test_gekko_records_must_follow_the_hook_order():
    collector = _collector()
    with pytest.raises((KeyError, RuntimeError)):
        collector.record_gekko_update_ratio(0, RATIOS[0])
    with pytest.raises((KeyError, RuntimeError)):
        collector.record_gekko_branches(0, SCA, AWK, GATE, UNAVAILABLE)
    _record_layer(collector)
    with pytest.raises(RuntimeError, match="Duplicate"):
        collector.record_gekko_query_key_stats(0, Q, K, SQ, sca_key_l2=SK_L2)
    with pytest.raises(RuntimeError, match="Duplicate Gekko branches"):
        collector.record_gekko_branches(0, SCA, AWK, GATE, UNAVAILABLE)


@pytest.mark.parametrize("partial, message", [
    (lambda collector: None, "Missing Gekko statistics"),
    (lambda collector: collector.record_gekko_query_key_stats(0, Q, K, SQ, sca_key_l2=SK_L2), "Missing Gekko branch statistics"),
])
def test_an_incomplete_gekko_layer_skips_the_step(caplog, partial, message):
    collector = _collector()
    partial(collector)

    assert _flush(collector) is None
    assert message in caplog.text


# Parameter norms, on a real GekkoBlock

GEKKO_MATRICES = {
    "attention/q_proj": "gda.wq.weight",
    "attention/k_proj": "gda.wk.weight",
    "attention/v_proj": "gda.wv.weight",
    "attention/o_proj": "gda.wo.weight",
    "attention/output_gate": "gda.wr.weight",
    "attention/q_conv": "gda.q_conv.weight",
    "attention/k_conv": "gda.k_conv.weight",
    "attention/v_conv": "gda.v_conv.weight",
    "attention/query_norm": "gda.query_norm.weight",
    "attention/time_norm/weight": "gda.timenorm.weight",
    "attention/time_norm/bias": "gda.timenorm.bias",
    "mlp/gate_proj": "nffn.fc1.weight",
    "mlp/up_proj": "nffn.fc3.weight",
    "mlp/down_proj": "nffn.fc2.weight",
}
GEKKO_OPTIONAL = {  # logged only when the layer has them
    "attention/v_proj/bias": "gda.wv.bias",
    "attention/output_gate/bias": "gda.wr.bias",
    "mlp/norm/weight": "nffn.norm.weight",
    "mlp/norm/bias": "nffn.norm.bias",
}


def _gekko_block(**overrides):
    cfg = dict(
        arch="gekko", num_layers=1, model_dim=8, num_heads=2, ffn_hidden_dim=16, apply_rmsnorm=True,
        swiglu=True, vocab_size=11, chunk_size=4, timenorm_num_groups=2,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return GekkoBlock(ModelConf(**cfg), layer_id=0)


def _shard_block(block, backend, monkeypatch):
    """Shard one decoder layer. FSDP1 builds its handles on the first forward, so run an identity one."""
    if backend == "fsdp1":
        wrapped = FullyShardedDataParallel(block, device_id=torch.device("cpu"))
        monkeypatch.setattr(GekkoBlock, "forward", lambda self, x, *args, **kwargs: (x, None, None))
        wrapped(torch.zeros(1, 4, 8))
        return wrapped
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    fully_shard(block, mesh=init_device_mesh("cpu", (1,)))
    return SimpleNamespace(layers=[block])  # the FSDP2 path reads decoder layers off the model


def _weight_norms(model):
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_weight_norms=True)
    collector.begin_step(1)
    collector.record_activation(0, torch.ones(1))
    collector.end_forward()
    collector.record_matrix_norms(model)
    logged = {}
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.update(metrics)), 1)
    return {name: value for name, value in logged.items() if name.startswith("weight_norms/")}


@pytest.mark.usefixtures("single_process_group")
@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("overrides, optional", [
    ({}, {"attention/v_proj/bias", "mlp/norm/weight"}),
    ({"apply_bias_term": True}, {"attention/v_proj/bias", "attention/output_gate/bias", "mlp/norm/weight"}),
    ({"apply_bias_term": True, "apply_rmsnorm": False}, set(GEKKO_OPTIONAL)),
])
def test_gekko_weight_norms_cover_every_present_parameter(monkeypatch, backend, overrides, optional):
    block = _gekko_block(**overrides)
    paths = {**GEKKO_MATRICES, **{path: GEKKO_OPTIONAL[path] for path in optional}}
    expected = {}
    with torch.no_grad():
        for i, (path, fqn) in enumerate(paths.items()):
            parameter = block.get_parameter(fqn)
            parameter.fill_(0.5 * (i + 1))  # a constant c over n entries has l2 = |c| * sqrt(n)
            expected[f"weight_norms/layer_000/{path}/l2"] = 0.5 * (i + 1) * math.sqrt(parameter.numel())

    assert _weight_norms(_shard_block(block, backend, monkeypatch)) == pytest.approx(expected, rel=1e-5)


@pytest.mark.usefixtures("single_process_group")
def test_a_missing_required_gekko_parameter_is_an_error(monkeypatch):
    block = _gekko_block()
    block.gda.q_conv.register_parameter("weight", None)

    with pytest.raises(RuntimeError, match="Expected 16 matrix parameters for layer 0, found 15"):
        _weight_norms(_shard_block(block, "fsdp2", monkeypatch))
