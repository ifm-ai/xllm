import pytest
import torch.nn.functional as F

import xllm.models.fused_blocks.mha as mha
import xllm.models.fused_blocks.transformer.transformer as transformer_fused
import xllm.models.fused_blocks.utils as fused_utils
from tests.health.helpers import MASK, TARGETS, TOKENS, grouped_rms_norm, logged_metrics, tiny_transformer

# Large runs use the fused dense block. Its forward (TransformerBlockFunction, multihead_attention_fwd and
# the flash_attention_fwd wrapper) runs here as-is; only the three CUDA kernels it calls are replaced by
# plain torch, so what its hooks record is checked against the unfused path on the same weights.

FLAGS = dict(enable_query_key_stats=True, enable_logits_stats=True)


def _layer_or_rmsnorm_fwd(x, weight, bias, groups, layernorm_eps, rmsnorm_eps, apply_rmsnorm):
    assert apply_rmsnorm
    return grouped_rms_norm(x, weight, groups, rmsnorm_eps), None, None


def _rmsnorm_fwd(x, weight, num_groups, eps):  # qk-norm: one group per head
    return grouped_rms_norm(x, weight, num_groups, eps), None


def _flash_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout, causal):
    # The flash kernel: (batch, seq, heads, dim) in, (out, lse, rng_state) back.
    assert cu_seqlens_q is None and dropout == 0.0 and causal
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=scale, enable_gqa=True,
    )
    return out.transpose(1, 2), None, None


@pytest.fixture
def fused_attention_calls(monkeypatch):
    calls = []

    def flash_fwd(*args):
        calls.append(args[0].shape)
        return _flash_fwd(*args)

    monkeypatch.setattr(transformer_fused, "layer_or_rmsnorm_fwd", _layer_or_rmsnorm_fwd)
    monkeypatch.setattr(mha, "rmsnorm_fwd", _rmsnorm_fwd)
    monkeypatch.setattr(fused_utils, "flash_fwd", flash_fwd)
    return calls


CONFIGS = [
    {},
    {"qknorm": True, "recompute_q": True, "recompute_v": True},  # config.py requires both with a fused block
    {"num_heads": 4, "num_kv_heads": 2},  # GQA
    {"rope_head_dim": 2},  # partial RoPE
    {"rope_head_dim": 0},  # no RoPE
    {"apply_attn_gate": True},
]


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("overrides", CONFIGS)
def test_fused_dense_block_logs_what_the_unfused_block_logs(fused_attention_calls, num_layers, overrides):
    unfused = tiny_transformer(num_layers, **overrides)
    fused = tiny_transformer(num_layers, fused_block=True, causal_attn_backend="flash", **overrides)
    fused.load_state_dict(unfused.state_dict())
    expected = logged_metrics(unfused, TOKENS, TARGETS, MASK, **FLAGS)

    metrics = logged_metrics(fused, TOKENS, TARGETS, MASK, train=True, **FLAGS)  # training mode is the fused path

    assert len(fused_attention_calls) == num_layers  # every layer went through the fused attention
    assert metrics == pytest.approx(expected, rel=1e-4, abs=1e-5)
