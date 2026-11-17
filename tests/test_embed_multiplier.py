"""model.embed_multiplier: a float multiplier on the embedding output, on top of scale_emb's sqrt(model_dim). CPU."""
import math

import torch

from tests.health.helpers import tiny_transformer


def test_embed_multiplier():
    tokens = torch.tensor([[1, 5, 7, 2]])
    plain = tiny_transformer(1)
    assert plain.embed.embedding_scale is None  # the default: no multiply, as before
    for kw, scale in (({"scale_emb": True}, math.sqrt(8)), ({"embed_multiplier": 2.5}, 2.5),
                      ({"scale_emb": True, "embed_multiplier": 0.5}, 0.5 * math.sqrt(8))):
        model = tiny_transformer(1, **kw)
        assert torch.equal(model.embed.weight, plain.embed.weight)  # init unchanged
        assert torch.allclose(model.embed(tokens), scale * plain.embed(tokens), rtol=1e-6, atol=0)
