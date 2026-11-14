"""What the GPU tests of the hyper-connections share: tiny models, their loss and gradients."""

import types

import torch

from xllm.config import ModelConf
from xllm.models.gekko import Gekko
from xllm.models.transformer import Transformer
from xllm.utils import create_on_gpu

MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=64, num_dense_layers=1)
PRODUCTION = dict(recompute_q=True, recompute_fc1_out=True, recompute_fc3_out=True)  # the fused block's recompute switches


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def build(residual_func, fused_block, model_dim, num_layers, arch='transformer', **kwargs):
    heads = model_dim // 32 if arch == 'transformer' else min(model_dim // 32, 32)  # Gekko's fused per-head norm: <= 32 heads
    cfg = ModelConf(arch=arch, num_layers=num_layers, model_dim=model_dim, num_heads=heads,
                    num_kv_heads=max(1, heads // 2), head_dim=32, chunk_size=32, ffn_hidden_dim=2 * model_dim, swiglu=True,
                    apply_rmsnorm=True, vocab_size=256, causal_attn_backend='flash' if arch == 'transformer' else None,
                    residual_func=residual_func, fused_block=fused_block, **kwargs)
    torch.manual_seed(0)
    with create_on_gpu():
        model = (Transformer if arch == 'transformer' else Gekko)(cfg, types.SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    model = model.cuda().to(torch.bfloat16).train()
    with torch.no_grad():  # move the controllers off their near-identity init so every coefficient matters
        for name, p in model.named_parameters():
            if '.residual.' in name and not name.endswith('weight'):
                p.add_(torch.randn_like(p) * 0.5)
    return model


def loss_and_grads(model, B=2, L=64, noise=0.0):
    """`noise` > 0 perturbs the embedding by that relative amount (the module path's BF16 sensitivity, see `bf16_floor`
    in test_hyper_connection_fused_block.py)."""
    g = torch.Generator(device='cuda').manual_seed(1)
    tokens = torch.randint(3, 256, (B, L), device='cuda', generator=g)
    if noise:
        def hook(mod, inp, out):
            return (out.float() * (1 + noise * torch.randn(out.shape, device=out.device, generator=g))).to(out.dtype)
        model.embed.register_forward_hook(hook)
    loss, _, _ = model(tokens, multi_segments=False, targets=torch.roll(tokens, -1, dims=1))
    loss = loss.float().mean()
    loss.backward()
    return loss.item(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
