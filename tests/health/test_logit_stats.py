import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import xllm.models.fused_blocks.cross_entropy as cross_entropy
import xllm.models.fused_blocks.gekko.gekko as gekko_fused
import xllm.models.fused_blocks.transformer.transformer as transformer_fused
from tests.health.helpers import (
    MASK, TARGETS, TOKENS, logged_metrics, make_layers_identity, set_embeddings, stats, tiny_transformer,
)
from xllm.health_metrics import HealthMetricCollector


def _logits_stats(metrics):
    return stats(metrics, "logits_stats")


def _reference(logits, targets, mask):
    """Mean/min/max over the logits of tokens that are neither ignored nor masked out."""
    valid = targets.ne(-100) if mask is None else targets.ne(-100) & mask
    chosen = logits.float()[valid]
    return [chosen.mean().item(), chosen.min().item(), chosen.max().item()]


def _collector_run(fn, enable_logits_stats=True):
    """Call fn under an active collector; return what flush logs."""
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=1, enable_logits_stats=enable_logits_stats)
    collector.begin_step(1)
    try:
        with torch.no_grad():
            fn()
    finally:
        collector.end_forward()
    collector.record_activation(0, torch.ones(1))
    logged = {}
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.update(metrics)), 1)
    return logged


# Unfused output layer (eval path), through the real tiny model

# Identity layers and output.weight = eye(11, 8): a token's logits are its RMS-normed embedding plus three
# zeros. Each case: embedding rows, tokens, targets, token mask, expected (mean, min, max) over valid tokens.
ONES, SIGNS = [1.0] * 8, [-2.0] * 4 + [2.0] * 4  # normed: [1] * 8 and [-1] * 4 + [1] * 4
HAND_COMPUTED_CASES = [
    ({3: ONES}, [[3]], [[0]], None, (8 / 11, 0.0, 1.0)),
    ({3: SIGNS}, [[3]], [[0]], None, (0.0, -1.0, 1.0)),
    ({3: ONES, 4: SIGNS}, [[3, 4]], [[0, 0]], None, (4 / 11, -1.0, 1.0)),
    ({3: ONES, 4: SIGNS}, [[3, 4]], [[0, 0]], [[1, 0]], (8 / 11, 0.0, 1.0)),  # token 4 masked out
    ({3: ONES, 4: SIGNS}, [[3, 4]], [[-100, 0]], None, (0.0, -1.0, 1.0)),  # token 3 ignored
]


@pytest.mark.parametrize("rows, tokens, targets, mask, expected", HAND_COMPUTED_CASES)
def test_unfused_output_layer_logs_hand_computed_logit_stats(rows, tokens, targets, mask, expected):
    model = tiny_transformer(num_layers=1)
    make_layers_identity(model)
    set_embeddings(model, rows)
    with torch.no_grad():
        model.output.output.weight.copy_(torch.eye(11, 8))
    mask = None if mask is None else torch.tensor(mask, dtype=torch.bool)

    metrics = logged_metrics(model, torch.tensor(tokens), torch.tensor(targets), mask, enable_logits_stats=True)

    assert _logits_stats(metrics) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("mask", [None, MASK])
def test_unfused_output_layer_logs_stats_over_valid_tokens_only(num_layers, mask):
    model = tiny_transformer(num_layers)
    with torch.no_grad():
        logits, _, _ = model.eval()(TOKENS, multi_segments=False)  # no targets: the model returns logits

    metrics = logged_metrics(model, TOKENS, TARGETS, mask, enable_logits_stats=True)

    assert _logits_stats(metrics) == pytest.approx(_reference(logits, TARGETS, mask), rel=1e-5)


def test_all_tokens_masked_logs_nan_logit_stats_and_keeps_the_rest(caplog):
    model = tiny_transformer(num_layers=1)

    metrics = logged_metrics(model, TOKENS, TARGETS, torch.zeros_like(MASK), enable_logits_stats=True)

    assert all(math.isnan(value) for value in _logits_stats(metrics))
    assert "No valid logits" in caplog.text
    assert "activation_stats/layer_000/mean" in metrics


def test_logit_stats_are_not_collected_when_disabled():
    metrics = logged_metrics(tiny_transformer(num_layers=1), TOKENS, TARGETS, MASK)

    assert not any(name.startswith("logits_stats/") for name in metrics)


# Fused output layer (training path): the real TransformerOutputLayerFunction with a CPU final norm

def _torch_layer_or_rmsnorm_fwd(x, weight, bias, groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm):
    # Stand-in for the CUDA final-norm kernel, matching the CPU GroupRMSNorm used on the unfused path.
    return F.rms_norm(x, x.shape[-1:], weight + 1.0, rmsnorm_eps), None, None


@pytest.mark.parametrize("recompute_logits", [False, True])
def test_fused_output_layer_matches_unfused_logit_stats(monkeypatch, recompute_logits):
    monkeypatch.setattr(transformer_fused, "layer_or_rmsnorm_fwd", _torch_layer_or_rmsnorm_fwd)
    model = tiny_transformer(num_layers=1)
    model.output.recompute_logits = recompute_logits
    unfused = logged_metrics(model, TOKENS, TARGETS, MASK, enable_logits_stats=True)

    fused = logged_metrics(model, TOKENS, TARGETS, MASK, train=True, enable_logits_stats=True)

    assert _logits_stats(fused) == pytest.approx(_logits_stats(unfused), rel=1e-5)


# Fused cross-entropy kernel entry point

@pytest.mark.parametrize("compute_grad", [False, True])
@pytest.mark.parametrize("mask", [None, MASK, torch.zeros_like(MASK)])
def test_fused_cross_entropy_counts_only_valid_tokens(compute_grad, mask):
    torch.manual_seed(0)
    x, weight = torch.randn(2, 4, 8), torch.randn(11, 8)
    valid = TARGETS.ne(-100) if mask is None else TARGETS.ne(-100) & mask
    logits = F.linear(x, weight)[valid]

    output = cross_entropy.fused_linear_cross_entropy_fwd(
        x, weight, TARGETS, mask, compute_grad=compute_grad, gather_input=False, collect_logits_stats=True,
    )

    assert output.logits_stats.count.item() == valid.sum().item() * 11
    if valid.any():
        torch.testing.assert_close(output.logits_stats.sum, logits.sum())
        torch.testing.assert_close(output.logits_stats.minimum, logits.min())
        torch.testing.assert_close(output.logits_stats.maximum, logits.max())
    else:
        assert output.logits_stats.sum.item() == 0.0
        assert (output.logits_stats.minimum.item(), output.logits_stats.maximum.item()) == (math.inf, -math.inf)
    assert cross_entropy.fused_linear_cross_entropy_fwd(
        x, weight, TARGETS, mask, compute_grad=compute_grad, gather_input=False,
    ).logits_stats is None


def test_fused_logit_stats_add_no_tensor_parallel_collective(monkeypatch):
    # Pretend to be rank 0 of a 2-way tensor-parallel group holding an 11-row vocabulary shard.
    monkeypatch.setattr(cross_entropy, "get_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cross_entropy, "get_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(cross_entropy, "get_model_parallel_group", lambda: None)
    monkeypatch.setattr(cross_entropy, "all_reduce", lambda tensor, parallel_region, async_op=False: (tensor, None))
    collectives = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: collectives.append(tensor.numel()))
    x, weight = torch.randn(1, 4, 8), torch.randn(11, 8)

    def collectives_with(collect_logits_stats):
        collectives.clear()
        cross_entropy.fused_linear_cross_entropy_fwd(
            x, weight, TARGETS[:1], MASK[:1], gather_input=False, collect_logits_stats=collect_logits_stats,
        )
        return list(collectives)

    assert collectives_with(True) == collectives_with(False) != []


# Gekko fused output layer: the real GekkoOutputLayerFunction with no context-parallel neighbours

def _identity_timenorm_fwd(x, bos_mask, prev_count, prev_mean, prev_var, *args):
    # Stand-in for the CUDA timestep norm; the logits are then just x @ wo.T.
    return x, prev_count, prev_mean, prev_var, None, None


@pytest.mark.parametrize("enable_logits_stats", [True, False])
@pytest.mark.parametrize("recompute_logits", [False, True])
def test_gekko_fused_output_layer_records_logit_stats(monkeypatch, recompute_logits, enable_logits_stats):
    monkeypatch.setattr(gekko_fused, "should_recv_from_prev", lambda: False)
    monkeypatch.setattr(gekko_fused, "should_send_to_next", lambda: False)
    monkeypatch.setattr(gekko_fused, "timenorm_fwd", _identity_timenorm_fwd)
    torch.manual_seed(0)
    x, wo = torch.randn(2, 4, 8), torch.randn(11, 8)

    def output_layer():
        gekko_fused.GekkoOutputLayerFunction.apply(
            x, TARGETS, MASK, None,  # x, y, mask, bos_mask
            torch.zeros(8), torch.zeros(8), torch.zeros(()), torch.zeros(8), torch.ones(8),  # timenorm params
            1, None, None, "cub", wo, 1e-5, recompute_logits,
        )

    metrics = _collector_run(output_layer, enable_logits_stats)

    if enable_logits_stats:
        assert _logits_stats(metrics) == pytest.approx(_reference(F.linear(x, wo), TARGETS, MASK), rel=1e-5)
    else:
        assert not any(name.startswith("logits_stats/") for name in metrics)
