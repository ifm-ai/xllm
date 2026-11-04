"""MoVA value-router statistics from real MoVA layers: what each value router chose is what gets logged, fused or not."""
import math

import pytest
import torch
import torch.distributed as dist

import xllm.distributed.initialize as parallel_state
import xllm.models.fused_blocks.transformer.moe as moe_fused
import xllm.models.fused_blocks.transformer.mova as mova_fused
from tests.health.helpers import MASK, TARGETS, TOKENS, logged_metrics, shard, tiny_transformer
from tests.health.test_fused_dense_block import _layer_or_rmsnorm_fwd, fused_attention_calls  # noqa: F401
from xllm.modules.moe.moe import moe_layer_ids
from xllm.modules.moe.mova import mova_layer_ids

MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=8, num_dense_layers=1)
MOVA = MOE | dict(num_values=4, num_activated_values=2)  # MoVA requires MoE; both routers share the moe_router_* settings
BIAS = dict(moe_router_bias=True, moe_router_bias_update_rate=0.5)
SCALED = dict(moe_router_scaling_factor=2.5)
PRESET_BIAS = torch.tensor([0.3, -0.2, 0.1, 0.0])


@pytest.fixture(autouse=True)
def bias_update_group(single_process_group, monkeypatch):
    # The router's backward all-reduces its loads over this group before updating the bias.
    monkeypatch.setattr(parallel_state, "_HYBRID_SHARD_DATA_PARALLEL_GROUP", dist.group.WORLD)
    monkeypatch.setattr(parallel_state, "_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE", 1)


def routed_metrics(model, families=("mova",), **kwargs):
    """One logged step expecting each family on exactly the layers the model built with it."""
    helpers = {"moe": moe_layer_ids, "mova": mova_layer_ids}
    return logged_metrics(
        model, TOKENS, TARGETS, MASK, router_stats_layers={f: helpers[f](model) for f in families},
        load_balancing_type="dot", **kwargs,
    )


def entropy(p, dim=-1):
    return -(p * p.clamp_min(1e-300).log()).sum(dim)


def expected_value_router_metrics(layer_idx, scores, tokens_per_value, aux_loss, bias):
    layer = f"layer_{layer_idx:03d}"
    loads = tokens_per_value.double()
    topk = scores.double() / scores.double().sum(-1, keepdim=True)
    metrics = {
        f"mova_value_utilization/{layer}/load_max_over_mean": loads.max() / loads.mean(),
        f"mova_value_utilization/{layer}/load_entropy_normalized": entropy(loads / loads.sum()) / math.log(len(loads)),
        f"mova_value_utilization/{layer}/dead_values": (loads == 0).sum(),
        f"mova_router_confidence_stats/{layer}/topk_entropy_mean": entropy(topk).mean(),
        f"mova_auxiliary_loss/{layer}/value": aux_loss,
    }
    if bias is not None:
        metrics.update({
            f"mova_router_bias_stats/{layer}/mean": bias.mean(),
            f"mova_router_bias_stats/{layer}/rms": bias.square().mean().sqrt(),
            f"mova_router_bias_stats/{layer}/min": bias.min(),
            f"mova_router_bias_stats/{layer}/max": bias.max(),
            f"mova_router_bias_stats/{layer}/max_abs": bias.abs().max(),
        })
    return {name: float(value) for name, value in metrics.items()}


def mova_only(metrics):
    return {name: value for name, value in metrics.items() if name.startswith("mova_")}


@pytest.mark.parametrize("overrides", [MOVA, MOVA | BIAS, MOVA | BIAS | SCALED], ids=["mova", "bias", "scaled"])
def test_each_mova_layer_logs_what_its_value_router_chose(overrides):
    model = tiny_transformer(3, **overrides)
    expected = {}
    for layer_idx in (1, 2):
        router = model.layers[layer_idx].mova.router
        if router.bias is not None:  # non-zero, so a logged zero could not pass by accident
            router.bias.copy_(PRESET_BIAS * layer_idx)
        bias = None if router.bias is None else router.bias.clone()

        def capture(module, args, output, layer_idx=layer_idx, bias=bias):
            scores, _, tokens_per_value, aux_loss = (t.detach().clone() for t in output)
            expected.update(expected_value_router_metrics(layer_idx, scores, tokens_per_value, aux_loss, bias))
        router.register_forward_hook(capture)

    metrics = routed_metrics(model, loss_scale=1.0)

    assert mova_only(metrics) == pytest.approx(expected)
    if "moe_router_bias" in overrides:  # backward moved the bias; the log kept the forward's value
        assert not torch.equal(model.layers[1].mova.router.bias, PRESET_BIAS)


@pytest.mark.parametrize("overrides, expected", [
    ({}, []),
    (MOE, []),
    (MOVA, [1, 2]),
    (MOVA | {"num_dense_layers": 0}, [0, 1, 2]),
], ids=["dense", "moe-only", "mova", "no-dense-prefix"])
def test_mova_layers_are_read_from_the_built_model(overrides, expected):
    assert mova_layer_ids(tiny_transformer(3, **overrides)) == expected
    for backend in ("fsdp1", "fsdp2"):
        assert mova_layer_ids(shard(tiny_transformer(3, **overrides), backend)) == expected


def test_a_layout_with_dense_layers_between_mova_layers_logs_only_the_mova_layers(caplog):
    # The Transformer only builds a dense prefix today; swap a dense block in to stand for any other layout.
    model = tiny_transformer(4, **MOVA)
    model.layers[2] = tiny_transformer(4).layers[2]

    metrics = routed_metrics(model)

    assert mova_layer_ids(model) == [1, 3]
    assert {name.split("/")[1] for name in mova_only(metrics)} == {"layer_001", "layer_003"}
    assert "Skipping health metrics" not in caplog.text


def test_moe_and_mova_stats_together_log_what_each_logs_alone(caplog):
    # Each MoVA layer has two routers; both families log for the same layers without touching each other.
    both = routed_metrics(tiny_transformer(3, **MOVA | BIAS), families=("moe", "mova"))
    moe = routed_metrics(tiny_transformer(3, **MOVA | BIAS), families=("moe",))
    mova = routed_metrics(tiny_transformer(3, **MOVA | BIAS), families=("mova",))

    assert {name.split("/")[1] for name in both if name.startswith(("moe_", "mova_"))} == {"layer_001", "layer_002"}
    assert both == pytest.approx(moe | mova)
    assert "Skipping health metrics" not in caplog.text


@pytest.fixture
def fused_mova_calls(fused_attention_calls, monkeypatch):
    for module in (moe_fused, mova_fused):
        monkeypatch.setattr(module, "layer_or_rmsnorm_fwd", _layer_or_rmsnorm_fwd)
    return fused_attention_calls


@pytest.mark.parametrize("overrides", [
    MOVA,
    MOVA | BIAS,
    MOVA | SCALED,  # the fused hook sees unscaled scores, the unfused one scaled
    MOVA | {"moe_router_score_func": "softmax"},
], ids=["mova", "bias", "scaled", "softmax"])
def test_fused_mova_blocks_log_what_the_unfused_blocks_log(fused_mova_calls, overrides):
    unfused = tiny_transformer(3, **overrides)
    for layer_idx in (1, 2):  # distinct non-zero biases, so logging one router's bias for the other cannot pass
        layer = unfused.layers[layer_idx]
        for router, sign in ((layer.mova.router, 1.0), (layer.moe.router, -2.0)):
            if router.bias is not None:
                router.bias.copy_(PRESET_BIAS * sign * layer_idx)
    fused = tiny_transformer(3, fused_block=True, causal_attn_backend="flash", **overrides)
    fused.load_state_dict(unfused.state_dict())  # the bias buffers travel with the state dict
    expected = routed_metrics(unfused, families=("moe", "mova"))

    metrics = routed_metrics(fused, families=("moe", "mova"), train=True)  # training mode is the fused path

    assert len(fused_mova_calls) == 3  # every layer went through a fused block
    assert any(name.startswith("mova_auxiliary_loss/") for name in metrics)
    if "moe_router_bias" in overrides:
        assert metrics["mova_router_bias_stats/layer_002/max"] == pytest.approx(0.6)
        assert metrics["moe_router_bias_stats/layer_002/max"] == pytest.approx(0.8)
    assert metrics == pytest.approx(expected, rel=1e-4, abs=1e-5)
