"""Single-GPU smoke of hyper-connections inside the Transformer and Gekko models.

Builds tiny models for every residual_func, runs a training forward / backward and an eval forward, and checks finite
losses, gradients on every controller parameter and that num_parameters() matches the real parameter count.

    python -m pytest -q tests/residual/test_hyper_connection_model.py
"""

import datetime
import os
import types

import pytest
import torch

from xllm.config import ModelConf
from xllm.distributed.initialize import initialize_model_parallel, model_parallel_is_initialized
from xllm.models.build import init_cache
from xllm.models.gekko import Gekko
from xllm.models.transformer import Transformer
from xllm.utils import create_on_gpu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.fixture(scope="module", autouse=True)
def single_process():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29523")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0,
                                             timeout=datetime.timedelta(seconds=600))
    if not model_parallel_is_initialized():
        initialize_model_parallel(1, 1, 600)


def make_cfg(arch: str, residual_func: str, **kwargs) -> ModelConf:
    return ModelConf(
        arch=arch, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, chunk_size=32,
        ffn_hidden_dim=128, swiglu=True, apply_rmsnorm=True, vocab_size=256,
        causal_attn_backend='flash' if arch == 'transformer' else None,
        residual_func=residual_func, **kwargs,
    )


MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=64, num_dense_layers=1)


# Gekko has no MoE block in this tree (GekkoBlock is dense; num_experts is ignored), so no gekko MoE case
@pytest.mark.parametrize("arch,residual_func,kwargs", [
    *[(arch, rf, {}) for arch in ('transformer', 'gekko') for rf in ('base', 'ihc', 'mhc', 'ohc')],
    *[('transformer', rf, MOE) for rf in ('ihc', 'mhc', 'ohc')],
    *[('transformer', rf, dict(num_values=4, num_activated_values=2, **MOE)) for rf in ('ihc', 'mhc', 'ohc')],
])
def test_train_and_eval_step(arch, residual_func, kwargs):
    torch.manual_seed(0)
    cfg = make_cfg(arch, residual_func, **kwargs)
    tokenizer = types.SimpleNamespace(bos_id=1, eos_id=2, pad_id=0)
    model_cls = Transformer if arch == 'transformer' else Gekko
    with create_on_gpu():
        model = model_cls(cfg, tokenizer).cuda().to(torch.bfloat16)
    model.train()

    total, activated, _ = model.num_parameters()
    actual = sum(p.numel() for p in model.parameters())
    assert actual == total and activated <= total, (arch, residual_func, actual, total, activated)
    assert model.tflops_per_token(64) > 0

    tokens = torch.randint(3, cfg.vocab_size, (2, 64), device='cuda')
    targets = torch.roll(tokens, -1, dims=1)
    loss, _, _ = model(tokens, multi_segments=False, targets=targets)
    loss = loss.float().mean()
    assert torch.isfinite(loss), loss
    loss.backward()
    hc_params = [(n, p) for n, p in model.named_parameters() if '.residual.' in n]
    if residual_func in ('ihc', 'mhc', 'ohc'):
        assert len(hc_params) == 2 * cfg.num_layers * 3, [n for n, _ in hc_params]
    for name, p in hc_params:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name

    model.eval()
    with torch.no_grad():
        eval_loss, _, _ = model(tokens, multi_segments=False, targets=targets, cache=init_cache(model))
    assert torch.isfinite(eval_loss.float().mean())
