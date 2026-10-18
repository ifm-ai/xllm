from logging import getLogger

import torch
from torch import nn
from torch.optim import AdamW

from xllm.config import OptimConf
from .lr_scheduler import build_lr_scheduler


logger = getLogger()


@torch.no_grad()
def rescale_grads(model: nn.Module, scale: float):
    if scale == 1.0:
        return
    p_with_grads = (p for p in model.parameters() if p.grad is not None)
    for p in p_with_grads:
        assert p.grad is not None  # mypy
        p.grad /= scale


def embed_readout_param_groups(model: nn.Module, weight_decay: float):
    """Two AdamW groups: the token embedding and readout matrix without weight decay, everything else with it."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        clean = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        if clean.startswith("embed.") or clean == "output.output.weight":
            no_decay.append((clean, p))
        else:
            decay.append(p)
    names = sorted(n for n, _ in no_decay)
    if not any(n.startswith("embed.") for n in names) or "output.output.weight" not in names:
        raise ValueError(f"no_decay_embed_readout: expected embedding and readout parameters, found {names}")
    logger.info(f"No weight decay on: {names}")
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": [p for _, p in no_decay], "weight_decay": 0.0}]


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str):
    logger.info("Starting build of optimizer...")
    params = model.parameters()
    if cfg.no_decay_embed_readout:
        params = embed_readout_param_groups(model, cfg.weight_decay)
    optimizer = AdamW(
        params,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.epsilon,
    )

    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
