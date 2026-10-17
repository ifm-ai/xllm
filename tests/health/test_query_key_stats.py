import pytest
import torch

from tests.health.helpers import (
    TOKENS, logged_metrics, make_layers_identity, set_embeddings, stats, tiny_transformer,
)
from xllm.modules.rotary_positional_embedding import apply_rotary_embeddings

STATS = ("mean", "min", "max", "vector_l2_mean", "vector_l2_min", "vector_l2_max")


def _query_key(metrics, name, layer_idx):
    return stats(metrics, f"{name}_stats/layer_{layer_idx:03d}", STATS)


def _summary(x):
    head_l2 = x.norm(dim=-1)
    return [x.mean().item(), x.min().item(), x.max().item(),
            head_l2.mean().item(), head_l2.min().item(), head_l2.max().item()]


def _query_key_reference(model, tokens=TOKENS):
    """Reference: capture each layer's Q/K projections, then apply the head split and RoPE by hand."""
    captured = []
    hooks = [
        layer.attention.causal_attention.register_forward_pre_hook(
            lambda module, args: captured.append((module, *args[:2], args[4]))
        )
        for layer in model.layers
    ]
    try:
        with torch.no_grad():
            model(tokens, multi_segments=False)
    finally:
        for hook in hooks:
            hook.remove()

    references = []
    for attention, xq, xk, freqs_cis in captured:
        q = xq.view(*xq.shape[:2], attention.n_heads, attention.head_dim)
        k = xk.view(*xk.shape[:2], attention.n_kv_heads, attention.head_dim)
        rope = attention.rope_head_dim
        if rope > 0:
            q_rope, k_rope = apply_rotary_embeddings(q[..., :rope], k[..., :rope], freqs_cis=freqs_cis, backward=False)
            q, k = torch.cat([q_rope, q[..., rope:]], -1), torch.cat([k_rope, k[..., rope:]], -1)
        references.append({"q": _summary(q), "k": _summary(k)})
    return references


def _set_query_key_projections(model, q_scale, k_scale):
    # Q = q_scale * normed input and K = k_scale * normed input, split into 2 heads of 4.
    with torch.no_grad():
        for layer in model.layers:
            layer.attention.wq.weight.copy_(q_scale * torch.eye(8))
            layer.attention.wk.weight.copy_(k_scale * torch.eye(8))


def _hand_computed_model(num_layers, rows):
    model = tiny_transformer(num_layers, rope_head_dim=0)
    make_layers_identity(model)
    _set_query_key_projections(model, q_scale=1.0, k_scale=2.0)
    set_embeddings(model, rows)
    return model


# No RoPE, Q = 1x and K = 2x the RMS-normed embedding (norm weight 1, so each row has RMS 1).
# Each case: embedding rows, tokens, expected Q and K (mean, min, max, head-L2 mean, min, max).
HAND_COMPUTED_CASES = [
    ({3: [1.0] * 8}, [[3]],
     [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]),
    ({3: [-2.0] * 4 + [2.0] * 4}, [[3]],
     [0.0, -1.0, 1.0, 2.0, 2.0, 2.0], [0.0, -2.0, 2.0, 4.0, 4.0, 4.0]),
    ({3: [5.0] * 8, 4: [-3.0] * 8}, [[3, 4]],
     [0.0, -1.0, 1.0, 2.0, 2.0, 2.0], [0.0, -2.0, 2.0, 4.0, 4.0, 4.0]),
    ({3: [2.0, 0.0] * 4}, [[3, 3]],
     [0.7071068, 0.0, 1.4142135, 2.0, 2.0, 2.0], [1.4142135, 0.0, 2.828427, 4.0, 4.0, 4.0]),
]
REAL_WEIGHT_CONFIGS = [
    {},                                   # full RoPE
    {"rope_head_dim": 2},                 # partial RoPE
    {"rope_head_dim": 0},                 # no RoPE
    {"num_heads": 4, "num_kv_heads": 2},  # GQA
    {"qknorm": True},
]


# Single decoder layer

@pytest.mark.parametrize("rows, tokens, expected_q, expected_k", HAND_COMPUTED_CASES)
def test_single_decoder_layer_logs_hand_computed_query_key_stats(rows, tokens, expected_q, expected_k):
    model = _hand_computed_model(1, rows)

    metrics = logged_metrics(model, torch.tensor(tokens), enable_query_key_stats=True)

    assert _query_key(metrics, "q", 0) == pytest.approx(expected_q, rel=1e-5, abs=1e-6)
    assert _query_key(metrics, "k", 0) == pytest.approx(expected_k, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS)
def test_single_decoder_layer_logs_its_post_rope_query_key(overrides):
    model = tiny_transformer(num_layers=1, **overrides)

    metrics = logged_metrics(model, enable_query_key_stats=True)

    reference = _query_key_reference(model)[0]
    for name in ("q", "k"):
        assert _query_key(metrics, name, 0) == pytest.approx(reference[name], rel=1e-5, abs=1e-6)


# Entire tiny model

@pytest.mark.parametrize("rows, tokens, expected_q, expected_k", HAND_COMPUTED_CASES)
def test_tiny_model_logs_hand_computed_query_key_stats_for_every_layer(rows, tokens, expected_q, expected_k):
    model = _hand_computed_model(3, rows)

    metrics = logged_metrics(model, torch.tensor(tokens), enable_query_key_stats=True)

    for layer_idx in range(3):
        assert _query_key(metrics, "q", layer_idx) == pytest.approx(expected_q, rel=1e-5, abs=1e-6)
        assert _query_key(metrics, "k", layer_idx) == pytest.approx(expected_k, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS + [{"layerwise_ckpt": True}])
def test_tiny_model_logs_each_layers_post_rope_query_key(num_layers, overrides):
    model = tiny_transformer(num_layers, **overrides)

    metrics = logged_metrics(model, enable_query_key_stats=True)

    for layer_idx, reference in enumerate(_query_key_reference(model)):
        for name in ("q", "k"):
            assert _query_key(metrics, name, layer_idx) == pytest.approx(reference[name], rel=1e-5, abs=1e-6)
