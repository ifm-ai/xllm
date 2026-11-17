"""model.embed_multiplier: a float multiplier on the embedding output, on top of scale_emb's sqrt(model_dim). CPU.

    PYTHONPATH=. pytest -q tests/test_embed_multiplier.py
"""
import math

import pytest
import torch

from tests.health.helpers import tiny_transformer
from xllm.config import ModelConf
from xllm.modules.model_parallel.layers import ParallelEmbedding

TOKENS = torch.tensor([[1, 5, 7, 2], [3, 3, 0, 9]])


def embedding(**kw):
    torch.manual_seed(0)
    return ParallelEmbedding(16, 8, **kw)


@pytest.mark.parametrize("scale_emb, multiplier, scale", [
    (False, 1.0, None),  # the default: no multiply, as before
    (True, 1.0, math.sqrt(8)),  # scale_emb alone, as before
    (False, 3.5, 3.5),
    (True, 0.25, 0.25 * math.sqrt(8)),
    (True, 1 / math.sqrt(8), None),  # the two cancel: no multiply
])
def test_output_and_gradient_scale(scale_emb, multiplier, scale):
    plain, scaled = embedding(), embedding(scale_emb=scale_emb, embed_multiplier=multiplier)
    assert torch.equal(plain.weight, scaled.weight)  # the init is unchanged
    assert scaled.embedding_scale == scale
    s = 1.0 if scale is None else scale
    out_plain, out = plain(TOKENS), scaled(TOKENS)
    assert torch.allclose(out, out_plain * s, rtol=1e-6, atol=0)
    out_plain.square().sum().backward()  # d/dW sum((s W)^2) = s^2 x the plain gradient
    out.square().sum().backward()
    assert torch.allclose(scaled.weight.grad, plain.weight.grad * s * s, rtol=1e-6, atol=0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
def test_model_conf_rejects_bad_multiplier(value):
    with pytest.raises(AssertionError, match="model.embed_multiplier must be finite and > 0"):
        ModelConf(embed_multiplier=value)
    assert ModelConf().embed_multiplier == 1.0


def test_transformer_passes_it_to_the_embedding():
    """model.embed_multiplier reaches the model's embedding, and its output is the plain model's x the multiplier."""
    plain, scaled = tiny_transformer(1), tiny_transformer(1, embed_multiplier=2.5)
    assert plain.embed.embedding_scale is None and scaled.embed.embedding_scale == 2.5
    assert torch.equal(scaled.embed.weight, plain.embed.weight)
    assert torch.allclose(scaled.embed(TOKENS), 2.5 * plain.embed(TOKENS), rtol=1e-6, atol=0)
