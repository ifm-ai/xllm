import pytest
import torch

from tests.health.helpers import (
    TOKENS, logged_metrics, make_layers_identity, set_embeddings, stats, tiny_transformer,
)
from xllm.modules.moe.moe import moe_layer_ids

MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=8, num_dense_layers=1)
MOVA = dict(num_values=4, num_activated_values=2)  # MoVA layers carry an MoE too


def _activations(metrics, layer_idx, point="ffn"):
    return stats(metrics, f"activation_stats/layer_{layer_idx:03d}/{point}")


def _layer_outputs(model, tokens=TOKENS):
    """Reference: the same real decoder layers run one at a time, without a collector."""
    with torch.no_grad():
        x = model.embed(tokens)
        freqs_cis = model.rope.get_freqs_cis(0, tokens.shape[1], x.device)
        outputs = []
        for layer in model.layers:
            x, _, _ = layer(x, freqs_cis)
            outputs.append([x.mean().item(), x.min().item(), x.max().item()])
    return outputs


def _attention_outputs(model, tokens=TOKENS):
    """Reference: what each layer's attention sublayer returns, before its FFN or MoE runs."""
    with torch.no_grad():
        x = model.embed(tokens)
        freqs_cis = model.rope.get_freqs_cis(0, tokens.shape[1], x.device)
        outputs = []
        for layer in model.layers:
            y = (layer.mova(x, freqs_cis) if hasattr(layer, "mova") else layer.attention(x, freqs_cis))[0]
            outputs.append([y.mean().item(), y.min().item(), y.max().item()])
            x, _, _ = layer(x, freqs_cis)
    return outputs


# Each case: embedding rows to set, input tokens, expected (mean, min, max) of the layer output.
HAND_COMPUTED_CASES = [
    ({3: [1.0] * 8}, [[3]], (1.0, 1.0, 1.0)),
    ({3: [-2.0] * 4 + [2.0] * 4}, [[3]], (0.0, -2.0, 2.0)),
    ({3: [1.0] * 8, 4: [3.0] * 8}, [[3, 4]], (2.0, 1.0, 3.0)),
    ({3: [0.5] * 8, 5: [-1.5] * 8}, [[3, 3], [5, 3]], (0.0, -1.5, 0.5)),
]
REAL_WEIGHT_CONFIGS = [{}, {"qknorm": True}, {"apply_attn_gate": True}]


# Single decoder layer

@pytest.mark.parametrize("rows, tokens, expected", HAND_COMPUTED_CASES)
def test_single_decoder_layer_logs_hand_computed_stats(rows, tokens, expected):
    model = tiny_transformer(num_layers=1)
    make_layers_identity(model)
    set_embeddings(model, rows)

    metrics = logged_metrics(model, torch.tensor(tokens))

    assert _activations(metrics, 0) == pytest.approx(expected)


@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS)
def test_single_decoder_layer_logs_its_output(overrides):
    model = tiny_transformer(num_layers=1, **overrides)

    metrics = logged_metrics(model)

    assert _activations(metrics, 0) == pytest.approx(_layer_outputs(model)[0], rel=1e-5, abs=1e-6)


# Entire tiny model

@pytest.mark.parametrize("rows, tokens, expected", HAND_COMPUTED_CASES)
def test_tiny_model_logs_hand_computed_stats_for_every_layer(rows, tokens, expected):
    model = tiny_transformer(num_layers=3)
    make_layers_identity(model)
    set_embeddings(model, rows)

    metrics = logged_metrics(model, torch.tensor(tokens))

    assert [_activations(metrics, i) for i in range(3)] == [pytest.approx(expected)] * 3


@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS + [{"layerwise_ckpt": True}])
def test_tiny_model_logs_each_layers_own_output(num_layers, overrides):
    model = tiny_transformer(num_layers, **overrides)

    metrics = logged_metrics(model)

    assert [_activations(metrics, i) for i in range(num_layers)] == [
        pytest.approx(layer, rel=1e-5, abs=1e-6) for layer in _layer_outputs(model)
    ]


# The attention sublayer, logged separately from the layer output

@pytest.mark.parametrize("rows, tokens, expected", HAND_COMPUTED_CASES)
def test_identity_layers_log_hand_computed_stats_at_both_points(rows, tokens, expected):
    model = tiny_transformer(num_layers=2)
    make_layers_identity(model)  # both sublayers pass their input through
    set_embeddings(model, rows)

    metrics = logged_metrics(model, torch.tensor(tokens))

    for layer_idx in range(2):
        assert _activations(metrics, layer_idx, "attention") == pytest.approx(expected)
        assert _activations(metrics, layer_idx) == pytest.approx(expected)


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS + [{"layerwise_ckpt": True}])
def test_each_layer_logs_its_own_attention_output(num_layers, overrides):
    model = tiny_transformer(num_layers, **overrides)

    metrics = logged_metrics(model, loss_scale=1.0)  # a backward recomputes checkpointed blocks

    assert [_activations(metrics, i, "attention") for i in range(num_layers)] == [
        pytest.approx(attention, rel=1e-5, abs=1e-6) for attention in _attention_outputs(model)
    ]


@pytest.mark.parametrize("overrides", [MOE, MOE | MOVA], ids=["moe", "mova"])
def test_moe_and_mova_layers_log_both_points(overrides):
    model = tiny_transformer(3, **overrides)

    metrics = logged_metrics(model, router_stats_layers={"moe": moe_layer_ids(model)})

    assert [_activations(metrics, i, "attention") for i in range(3)] == [
        pytest.approx(attention, rel=1e-5, abs=1e-6) for attention in _attention_outputs(model)
    ]
    assert [_activations(metrics, i) for i in range(3)] == [
        pytest.approx(layer, rel=1e-5, abs=1e-6) for layer in _layer_outputs(model)
    ]
    # The routed sublayer of layers 1 and 2 moved the residual stream; layer 0 is dense.
    assert all(_activations(metrics, i) != _activations(metrics, i, "attention") for i in range(3))
