"""The fused Gekko block with ModelConf.fp8_gemm against bf16 on a small model (1 GPU, the whole training step's
forward + backward through the model): losses and parameter gradients must agree to fp8 precision.
Run with the extension on the path, e.g. PYTHONPATH=<xext> pytest tests/test_fp8_gekko_block.py."""
import copy
import os
import sys
import types

import pytest
import torch

try:
    import xllm_extension  # noqa: F401
    _HAVE_EXT = True
except ImportError:
    _HAVE_EXT = False

pytestmark = pytest.mark.skipif(not _HAVE_EXT or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="needs a Blackwell GPU and xllm_extension")


class _Tok:
    bos_id, eos_id, pad_id, vocab_size, n_words = 1, 2, 0, 1024, 1024


def _build(fp8: bool, **overrides):
    from xllm.configuration import ConfStore
    from xllm.models import build_model
    cfg = copy.deepcopy(ConfStore["gekko-0.9B"])
    cfg.num_layers, cfg.model_dim, cfg.num_heads, cfg.num_kv_heads = 2, 256, 2, 1
    cfg.head_dim, cfg.v_head_dim, cfg.rope_head_dim, cfg.chunk_size, cfg.ffn_hidden_dim = 128, 128, 64, 256, 512
    cfg.vocab_size = _Tok.vocab_size
    cfg.timenorm_num_groups = 8
    cfg.fused_block, cfg.awm_backend, cfg.timenorm_backend, cfg.sca_backend = True, "cute", "cub", "cute"
    cfg.fp8_gemm = fp8
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return build_model(cfg, dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True, reshard_after_forward=True,
                       forward_prefetch=False, tokenizer=_Tok())


def _step(model, tokens, targets):
    tok_loss, _, _ = model(tokens=tokens, targets=targets, token_mask=None, multi_segments=True, deterministic=False)
    loss = tok_loss.sum() / tok_loss.numel()
    loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.item(), grads


@pytest.mark.parametrize("overrides", [
    {},
    {"recompute_conv_qk": False},  # the fp8 operands of mx / xf are kept from the forward
    {"recompute_fc1_out": True, "recompute_fc3_out": True},  # h1 / h3 recomputed from the kept operands
    {"recompute_conv_qk": False, "recompute_fc1_out": True, "recompute_fc3_out": True},
    {"recompute_q": True, "recompute_kv": True},  # q / k / v projections recomputed from the rowwise mx operand
    {"recompute_conv_qk": False, "recompute_q": True, "recompute_kv": True},
])
def test_fp8_block_matches_bf16(monkeypatch, overrides):
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        from xllm.distributed import init_torch_distributed, initialize_model_parallel
        init_torch_distributed(600)
        initialize_model_parallel(1, 1, 600)
    import xllm.modules.fused_ops.mxfp8 as mx
    calls = []
    real_mm = mx.mx_mm
    monkeypatch.setattr(mx, "mx_mm", lambda *a, **k: (calls.append(1), real_mm(*a, **k))[1])

    g = torch.Generator(device="cuda").manual_seed(1)
    tokens = torch.randint(3, _Tok.vocab_size, (1, 513), device="cuda", generator=g)
    tokens[0, 200] = _Tok.eos_id  # a segment boundary
    x, y = tokens[:, :-1], tokens[:, 1:]

    loss_bf16, grads_bf16 = _step(_build(False, **overrides), x, y)
    assert not calls
    loss_fp8, grads_fp8 = _step(_build(True, **overrides), x, y)
    assert calls, "the fp8 GEMM path did not run"
    assert abs(loss_fp8 - loss_bf16) / loss_bf16 < 2e-2, (loss_bf16, loss_fp8)
    assert grads_fp8.keys() == grads_bf16.keys()
    for name in grads_bf16:
        a, b = grads_bf16[name].flatten(), grads_fp8[name].flatten()
        if a.norm() == 0:
            assert b.norm() == 0, name
            continue
        cos = torch.dot(a, b) / (a.norm() * b.norm())
        rel = (a - b).norm() / a.norm()
        assert cos > 0.97 and rel < 0.25, (name, cos.item(), rel.item())


def test_fp8_attn_dp_block_matches_bf16(monkeypatch):
    """ModelConf.fp8_attn_dp (the attention backward's dP GEMMs in MXFP8, v_head_dim > 256) against the bf16 dP on the
    same small model: losses equal to run-to-run noise (the forward is untouched), gradients within fp8 precision."""
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        from xllm.distributed import init_torch_distributed, initialize_model_parallel
        init_torch_distributed(600)
        initialize_model_parallel(1, 1, 600)
    import xllm.modules.fused_ops.cute.sliding_chunk_attention as sca
    calls = []
    real = sca._mx_dp
    monkeypatch.setattr(sca, "_mx_dp", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    g = torch.Generator(device="cuda").manual_seed(1)
    tokens = torch.randint(3, _Tok.vocab_size, (1, 1025), device="cuda", generator=g)
    tokens[0, 300] = _Tok.eos_id
    x, y = tokens[:, :-1], tokens[:, 1:]
    loss_bf16, grads_bf16 = _step(_build(False, v_head_dim=512), x, y)
    assert not calls
    loss_fp8, grads_fp8 = _step(_build(False, v_head_dim=512, fp8_attn_dp=True), x, y)
    assert calls, "the fp8 dP path did not run"
    assert abs(loss_fp8 - loss_bf16) < 1e-3, (loss_bf16, loss_fp8)  # the forward is untouched (run-to-run noise only)
    for name in grads_bf16:
        a, b = grads_bf16[name].flatten(), grads_fp8[name].flatten()
        if a.norm() == 0:
            assert b.norm() == 0, name
            continue
        cos = torch.dot(a, b) / (a.norm() * b.norm())
        rel = (a - b).norm() / a.norm()
        assert cos > 0.97 and rel < 0.25, (name, cos.item(), rel.item())
