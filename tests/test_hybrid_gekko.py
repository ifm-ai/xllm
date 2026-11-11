"""Focused hybrid configuration, training and cache smoke tests."""

from types import SimpleNamespace

import pytest
import torch

from xllm.config import ModelConf
from xllm.models import init_cache
from xllm.models.gekko import Gekko, GekkoBlock, GekkoOutputLayer
from xllm.models.transformer import TransformerBlock, TransformerOutputLayer


def test_hybrid_is_opt_in():
    assert ModelConf(arch='gekko').full_attention_interval == 0
    assert ModelConf(arch='gekko', full_attention_interval=4).full_attention_interval == 4


def test_invalid_hybrid_configuration():
    for interval in (-1, 1, True, 4.0):
        with pytest.raises(ValueError):
            ModelConf(arch='gekko', full_attention_interval=interval)
    for options in (
        {'arch': 'transformer'}, {'num_experts': 1}, {'num_values': 1},
        {'head_dim': 128, 'v_head_dim': 256},
        {'fused_block': True}, {'fused_block': True, 'causal_attn_backend': 'swift'},
    ):
        with pytest.raises(ValueError):
            ModelConf(**{'arch': 'gekko', 'full_attention_interval': 4, **options})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA kernels')
@pytest.mark.parametrize('fused', [False, True])
def test_hybrid_train_and_cache(fused):
    torch.manual_seed(17)
    cfg = ModelConf(
        arch='gekko', full_attention_interval=4, num_layers=5, model_dim=256,
        ffn_hidden_dim=512, vocab_size=256, num_heads=2, num_kv_heads=1,
        head_dim=128, v_head_dim=128, rope_head_dim=64, chunk_size=128,
        swiglu=True, qknorm=True, apply_rmsnorm=True, timenorm_num_groups=8,
        causal_attn_backend='xattn', sca_backend='xattn', fused_block=fused,
        recompute_q=fused, recompute_v=fused, recompute_attention=fused,
        recompute_awk=fused, recompute_fc1_out=fused, recompute_fc3_out=fused,
    )
    tokenizer = SimpleNamespace(vocab_size=256, bos_id=1, eos_id=2, pad_id=0)
    model = Gekko(cfg, tokenizer).cuda().bfloat16()
    assert model.full_attention_layers == (0, 4)
    assert all(isinstance(layer, TransformerBlock if i in (0, 4) else GekkoBlock)
               for i, layer in enumerate(model.layers))
    assert isinstance(model.output, TransformerOutputLayer)
    assert model.num_parameters()[0] == sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(3, 256, (1, 256), device='cuda')
    tokens[:, [73, 129]] = 2
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, -17:] = False
    loss, _, _ = model(tokens, True, targets=tokens.roll(-1, 1), token_mask=mask)
    loss.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval()
    with torch.no_grad():
        _, _, cache = model(tokens[:, :128], False, cache=init_cache(model))
        logits, _, cache = model(tokens[:, 128:129], False, cache=cache)
    assert logits.shape == (1, 1, 256) and torch.isfinite(logits).all()
    assert all(cache[0][i][-1] == 129 for i in (0, 4))
    with pytest.raises(ValueError, match='packed inference'):
        model(tokens[:, :128], True, cache=cache)

    cfg.full_attention_interval = 0
    pure = Gekko(cfg, tokenizer)
    assert pure.full_attention_layers == ()
    assert all(isinstance(layer, GekkoBlock) for layer in pure.layers)
    assert isinstance(pure.output, GekkoOutputLayer)
