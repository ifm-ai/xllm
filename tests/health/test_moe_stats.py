"""MoE router statistics from real MoE layers: what each router chose is what gets logged, fused or not."""
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

MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=8, num_dense_layers=1)
BIAS = dict(moe_router_bias=True, moe_router_bias_update_rate=0.5)
MOVA = dict(num_values=4, num_activated_values=2)  # MoVA layers carry an MoE too


@pytest.fixture(autouse=True)
def bias_update_group(single_process_group, monkeypatch):
    # The router's backward all-reduces its loads over this group before updating the bias.
    monkeypatch.setattr(parallel_state, "_HYBRID_SHARD_DATA_PARALLEL_GROUP", dist.group.WORLD)
    monkeypatch.setattr(parallel_state, "_HYBRID_SHARD_DATA_PARALLEL_WORLD_SIZE", 1)


def moe_metrics(model, **kwargs):
    """One logged step with MoE statistics expected on exactly the layers the model built with an MoE."""
    return logged_metrics(
        model, TOKENS, TARGETS, MASK, router_stats_layers={"moe": moe_layer_ids(model)},
        load_balancing_type="dot", **kwargs,
    )


def entropy(p, dim=-1):
    return -(p * p.clamp_min(1e-300).log()).sum(dim)


def expected_router_metrics(layer_idx, scores, tokens_per_expert, aux_loss, bias):
    layer = f"layer_{layer_idx:03d}"
    loads = tokens_per_expert.double()
    topk = scores.double() / scores.double().sum(-1, keepdim=True)
    metrics = {
        f"moe_utilization_stats/{layer}/load_max_over_mean": loads.max() / loads.mean(),
        f"moe_utilization_stats/{layer}/load_entropy_normalized": entropy(loads / loads.sum()) / math.log(len(loads)),
        f"moe_utilization_stats/{layer}/zero_experts": (loads == 0).sum(),
        f"moe_router_confidence_stats/{layer}/topk_entropy_mean": entropy(topk).mean(),
        f"moe_auxiliary_loss/{layer}/value": aux_loss,
    }
    if bias is not None:
        metrics.update({
            f"moe_router_bias_stats/{layer}/mean": bias.mean(),
            f"moe_router_bias_stats/{layer}/rms": bias.square().mean().sqrt(),
            f"moe_router_bias_stats/{layer}/min": bias.min(),
            f"moe_router_bias_stats/{layer}/max": bias.max(),
            f"moe_router_bias_stats/{layer}/max_abs": bias.abs().max(),
        })
    return {name: float(value) for name, value in metrics.items()}


@pytest.mark.parametrize("overrides", [MOE, MOE | BIAS, MOE | BIAS | MOVA], ids=["moe", "bias", "mova"])
def test_each_moe_layer_logs_what_its_router_chose(overrides):
    model = tiny_transformer(3, **overrides)
    expected = {}
    for layer_idx in (1, 2):
        router = model.layers[layer_idx].moe.router
        if router.bias is not None:  # non-zero, so a logged zero could not pass by accident
            router.bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.0]) * layer_idx)
        bias = None if router.bias is None else router.bias.clone()

        def capture(module, args, output, layer_idx=layer_idx, bias=bias):
            scores, _, tokens_per_expert, aux_loss = (t.detach().clone() for t in output)
            expected.update(expected_router_metrics(layer_idx, scores, tokens_per_expert, aux_loss, bias))
        router.register_forward_hook(capture)

    metrics = moe_metrics(model, loss_scale=1.0)

    assert {name: value for name, value in metrics.items() if name.startswith("moe_")} == pytest.approx(expected)
    if "moe_router_bias" in overrides:  # backward moved the bias; the log kept the forward's value
        assert not torch.equal(model.layers[1].moe.router.bias, torch.tensor([0.3, -0.2, 0.1, 0.0]))


@pytest.mark.parametrize("overrides, expected", [
    ({}, []),
    (MOE, [1, 2]),
    (MOE | {"num_dense_layers": 0}, [0, 1, 2]),
    (MOE | MOVA, [1, 2]),
], ids=["dense", "moe", "no-dense-prefix", "mova"])
def test_moe_layers_are_read_from_the_built_model(overrides, expected):
    model = tiny_transformer(3, **overrides)
    assert moe_layer_ids(model) == expected
    for backend in ("fsdp1", "fsdp2"):
        assert moe_layer_ids(shard(tiny_transformer(3, **overrides), backend)) == expected


def test_a_layout_with_dense_layers_between_moe_layers_logs_only_the_moe_layers(caplog):
    # The Transformer only builds a dense prefix today; swap a dense block in to stand for any other layout.
    model = tiny_transformer(4, **MOE)
    model.layers[2] = tiny_transformer(4).layers[2]

    metrics = moe_metrics(model)

    assert moe_layer_ids(model) == [1, 3]
    assert {name.split("/")[1] for name in metrics if name.startswith("moe_")} == {"layer_001", "layer_003"}
    assert "Skipping health metrics" not in caplog.text


@pytest.fixture
def fused_moe_calls(fused_attention_calls, monkeypatch):
    for module in (moe_fused, mova_fused):
        monkeypatch.setattr(module, "layer_or_rmsnorm_fwd", _layer_or_rmsnorm_fwd)
    return fused_attention_calls


@pytest.mark.parametrize("overrides", [
    MOE,
    MOE | BIAS,
    MOE | {"moe_router_scaling_factor": 2.5},  # the fused hook sees unscaled scores, the unfused one scaled
    MOE | {"moe_router_score_func": "softmax"},
    MOE | BIAS | MOVA,
], ids=["moe", "bias", "scaled", "softmax", "mova"])
def test_fused_moe_blocks_log_what_the_unfused_blocks_log(fused_moe_calls, overrides):
    unfused = tiny_transformer(3, **overrides)
    fused = tiny_transformer(3, fused_block=True, causal_attn_backend="flash", **overrides)
    fused.load_state_dict(unfused.state_dict())
    expected = moe_metrics(unfused)

    metrics = moe_metrics(fused, train=True)  # training mode is the fused path

    assert len(fused_moe_calls) == 3  # every layer went through a fused block
    assert any(name.startswith("moe_auxiliary_loss/") for name in metrics)
    assert metrics == pytest.approx(expected, rel=1e-4, abs=1e-5)
