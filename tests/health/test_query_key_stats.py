import pytest
import torch

from tests.health.helpers import (
    TOKENS, logged_metrics, make_layers_identity, set_embeddings, stats, tiny_transformer,
)
from xllm.modules.rotary_positional_embedding import apply_rotary_embedding, apply_rotary_embeddings

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
        heads, head_dim, rope = attention.n_heads, attention.head_dim, attention.rope_head_dim
        num_query_sets = xq.shape[-1] // (heads * head_dim)  # 2 under softdelta: wq packs (head, set, dim)
        q, *kq = xq.view(*xq.shape[:2], heads, num_query_sets, head_dim).unbind(dim=3)
        k = xk.view(*xk.shape[:2], attention.n_kv_heads, head_dim)
        if rope > 0:
            q_rope, k_rope = apply_rotary_embeddings(q[..., :rope], k[..., :rope], freqs_cis=freqs_cis, backward=False)
            q, k = torch.cat([q_rope, q[..., rope:]], -1), torch.cat([k_rope, k[..., rope:]], -1)
            kq = [torch.cat([apply_rotary_embedding(x[..., :rope], freqs_cis=freqs_cis, backward=False), x[..., rope:]], -1)
                  for x in kq]
        reference = {"q": _summary(q), "k": _summary(k)}
        if kq:
            reference["kq"] = _summary(kq[0])
        references.append(reference)
    return references


# With identity layers, no RoPE and scaled-identity projections, every family is a known multiple of
# the RMS-normed embedding (norm weight 1, so each row has RMS 1): Q = 1x, K = 2x and, under softdelta,
# the second query set KQ = 3x.
SCALES = {"q": 1.0, "k": 2.0, "kq": 3.0}
ATTENTION = ["softmax", "softdelta"]


def _set_scaled_projections(model):
    # One head per 4 features. Under softdelta wq packs (head, query set, dim): 4 q rows then 4 kq rows per head.
    with torch.no_grad():
        for layer in model.layers:
            layer.attention.wk.weight.copy_(SCALES["k"] * torch.eye(8))
            wq = layer.attention.wq.weight
            query_sets = ("q", "kq")[:wq.shape[0] // 8]
            for head in range(2):
                rows = torch.eye(8)[head * 4:(head + 1) * 4]
                for index, name in enumerate(query_sets):
                    start = (head * len(query_sets) + index) * 4
                    wq[start:start + 4] = SCALES[name] * rows


def _hand_computed_model(num_layers, rows, attn_act_func):
    model = tiny_transformer(num_layers, rope_head_dim=0, attn_act_func=attn_act_func)
    make_layers_identity(model)
    _set_scaled_projections(model)
    set_embeddings(model, rows)
    return model


def _expected_families(normed, attn_act_func):
    names = ("q", "k", "kq") if attn_act_func == "softdelta" else ("q", "k")
    return {name: [SCALES[name] * value for value in normed] for name in names}


# Each case: embedding rows, tokens, and (mean, min, max, head-L2 mean, min, max) of the normed embedding.
HAND_COMPUTED_CASES = [
    ({3: [1.0] * 8}, [[3]], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
    ({3: [-2.0] * 4 + [2.0] * 4}, [[3]], [0.0, -1.0, 1.0, 2.0, 2.0, 2.0]),
    ({3: [5.0] * 8, 4: [-3.0] * 8}, [[3, 4]], [0.0, -1.0, 1.0, 2.0, 2.0, 2.0]),
    ({3: [2.0, 0.0] * 4}, [[3, 3]], [0.7071068, 0.0, 1.4142135, 2.0, 2.0, 2.0]),
]
REAL_WEIGHT_CONFIGS = [
    {},                                   # full RoPE
    {"rope_head_dim": 2},                 # partial RoPE
    {"rope_head_dim": 0},                 # no RoPE
    {"num_heads": 4, "num_kv_heads": 2},  # GQA
    {"qknorm": True},
    {"attn_act_func": "softdelta"},  # two query sets: q and kq
]


# Single decoder layer

@pytest.mark.parametrize("attn_act_func", ATTENTION)
@pytest.mark.parametrize("rows, tokens, normed", HAND_COMPUTED_CASES)
def test_single_decoder_layer_logs_hand_computed_query_key_stats(attn_act_func, rows, tokens, normed):
    model = _hand_computed_model(1, rows, attn_act_func)

    metrics = logged_metrics(model, torch.tensor(tokens), enable_query_key_stats=True)

    for name, expected in _expected_families(normed, attn_act_func).items():
        assert _query_key(metrics, name, 0) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert ("kq_stats/layer_000/mean" in metrics) == (attn_act_func == "softdelta")


@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS)
def test_single_decoder_layer_logs_its_post_rope_query_key(overrides):
    model = tiny_transformer(num_layers=1, **overrides)

    metrics = logged_metrics(model, enable_query_key_stats=True)

    reference = _query_key_reference(model)[0]
    for name in reference:
        assert _query_key(metrics, name, 0) == pytest.approx(reference[name], rel=1e-5, abs=1e-6)
    assert ("kq_stats/layer_000/mean" in metrics) == (overrides.get("attn_act_func") == "softdelta")


# Entire tiny model

@pytest.mark.parametrize("attn_act_func", ATTENTION)
@pytest.mark.parametrize("rows, tokens, normed", HAND_COMPUTED_CASES)
def test_tiny_model_logs_hand_computed_query_key_stats_for_every_layer(attn_act_func, rows, tokens, normed):
    model = _hand_computed_model(3, rows, attn_act_func)

    metrics = logged_metrics(model, torch.tensor(tokens), enable_query_key_stats=True)

    for layer_idx in range(3):
        for name, expected in _expected_families(normed, attn_act_func).items():
            assert _query_key(metrics, name, layer_idx) == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("overrides", REAL_WEIGHT_CONFIGS + [{"layerwise_ckpt": True}])
def test_tiny_model_logs_each_layers_post_rope_query_key(num_layers, overrides):
    model = tiny_transformer(num_layers, **overrides)

    metrics = logged_metrics(model, enable_query_key_stats=True)

    for layer_idx, reference in enumerate(_query_key_reference(model)):
        for name in reference:
            assert _query_key(metrics, name, layer_idx) == pytest.approx(reference[name], rel=1e-5, abs=1e-6)
