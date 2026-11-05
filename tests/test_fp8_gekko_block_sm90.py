"""tests/test_fp8_gekko_block.py on Hopper: the fp8 GEMMs there need every dimension a multiple of 512, so the small
model is 512 wide (fp8_sm90 backend; the bf16 reference and the tolerances are the Blackwell test's)."""
import copy

import pytest
import torch

from tests import test_fp8_gekko_block as base

pytestmark = pytest.mark.skipif(not base._HAVE_EXT or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                                reason="needs a Hopper GPU and xllm_extension")


def _build(fp8: bool, **overrides):
    from xllm.configuration import ConfStore
    from xllm.models import build_model
    cfg = copy.deepcopy(ConfStore["gekko-0.9B"])
    cfg.num_layers, cfg.model_dim, cfg.num_heads, cfg.num_kv_heads = 2, 512, 4, 4
    cfg.head_dim, cfg.v_head_dim, cfg.rope_head_dim, cfg.chunk_size, cfg.ffn_hidden_dim = 128, 128, 64, 256, 1024
    cfg.vocab_size = base._Tok.vocab_size
    cfg.timenorm_num_groups = 8
    cfg.fused_block, cfg.awm_backend, cfg.timenorm_backend, cfg.sca_backend = True, "cute", "cub", "cute"
    cfg.fp8_gemm = fp8
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True, reshard_after_forward=True,
                       forward_prefetch=False, tokenizer=base._Tok())


@pytest.mark.parametrize("overrides", base.test_fp8_block_matches_bf16.pytestmark[0].args[1])
def test_fp8_block_matches_bf16_sm90(monkeypatch, overrides):
    monkeypatch.setattr(base, "_build", _build)
    base.test_fp8_block_matches_bf16(monkeypatch, overrides)
