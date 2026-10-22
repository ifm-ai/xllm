"""Single-GPU smoke of hyper-connections inside the Transformer and Gekko models.

Builds tiny models for every residual_func, runs a training forward / backward and an
eval forward, and checks finite losses, gradients on every controller parameter and
that num_parameters() matches the real parameter count.

    python tests/residual/test_hyper_connection_model.py
"""

import datetime
import os
import types

import torch

from xllm.config import ModelConf
from xllm.distributed import initialize_model_parallel
from xllm.models.build import init_cache
from xllm.models.gekko import Gekko
from xllm.models.transformer import Transformer
from xllm.utils import create_on_gpu


def init_single_process():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=1, rank=0, timeout=datetime.timedelta(seconds=600)
    )
    initialize_model_parallel(1, 1, 600)


def make_cfg(arch: str, residual_func: str, **kwargs) -> ModelConf:
    return ModelConf(
        arch=arch, num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, chunk_size=32,
        ffn_hidden_dim=128, swiglu=True, apply_rmsnorm=True, vocab_size=256,
        causal_attn_backend='flash' if arch == 'transformer' else None,
        residual_func=residual_func, **kwargs,
    )


def run(arch: str, residual_func: str, **kwargs):
    torch.manual_seed(0)
    cfg = make_cfg(arch, residual_func, **kwargs)
    tokenizer = types.SimpleNamespace(bos_id=1, eos_id=2, pad_id=0)
    model_cls = Transformer if arch == 'transformer' else Gekko
    with create_on_gpu():
        model = model_cls(cfg, tokenizer).cuda().to(torch.bfloat16)
    model.train()

    total, activated, embed = model.num_parameters()
    actual = sum(p.numel() for p in model.parameters())
    assert actual == total and activated <= total, (arch, residual_func, actual, total, activated)

    tokens = torch.randint(3, cfg.vocab_size, (2, 64), device='cuda')
    targets = torch.roll(tokens, -1, dims=1)
    loss, aux, _ = model(tokens, multi_segments=False, targets=targets)
    loss = loss.float().mean()
    assert torch.isfinite(loss), loss
    loss.backward()
    hc_params = [(n, p) for n, p in model.named_parameters() if '_hc.' in n]
    if residual_func in ('ihc', 'mhc', 'ohc'):
        assert len(hc_params) > 0
    for name, p in hc_params:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name

    model.eval()
    with torch.no_grad():
        eval_loss, _, _ = model(tokens, multi_segments=False, targets=targets, cache=init_cache(model))
    assert torch.isfinite(eval_loss.float().mean())
    print(f"{arch:12s} {residual_func:5s} {kwargs} params {actual} train loss {loss.item():.3f} "
          f"eval loss {eval_loss.float().mean().item():.3f} tflops/token {model.tflops_per_token(64):.3e}")


def main():
    init_single_process()
    for arch in ('transformer', 'gekko'):
        run(arch, 'base')
        run(arch, 'ihc')
        run(arch, 'mhc')
        run(arch, 'ohc')
        run(arch, 'ohc', residual_groups=8, residual_freq_base=2.0)
    moe = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=64, num_dense_layers=1)
    for residual_func in ('ihc', 'mhc', 'ohc'):
        run('transformer', residual_func, **moe)
        run('transformer', residual_func, num_values=4, num_activated_values=2, **moe)
    torch.distributed.destroy_process_group()
    print("pass")


if __name__ == '__main__':
    main()
