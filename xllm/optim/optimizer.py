from logging import getLogger
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW

from xllm.config import MuPConf, OptimConf
from .lr_scheduler import build_lr_scheduler
from .param_roles import param_roles


logger = getLogger()


@torch.no_grad()
def rescale_grads(model: nn.Module, scale: float):
    if scale == 1.0:
        return
    p_with_grads = (p for p in model.parameters() if p.grad is not None)
    for p in p_with_grads:
        assert p.grad is not None  # mypy
        p.grad /= scale


def base_weight_decay(cfg: OptimConf, total_steps: int) -> float:
    """Weight decay before per-role scaling.

    With tau_epoch set, the AdamW timescale 1 / (lr * weight_decay) is held at
    tau_epoch * total_steps (Wang & Aitchison), as in the CompleteP experiments.
    """
    if cfg.tau_epoch is None:
        return cfg.weight_decay
    return 1.0 / (cfg.lr * cfg.tau_epoch * total_steps)


def no_decay_embed_readout_groups(model: nn.Module, weight_decay: float):
    """Two AdamW groups: the token embedding and readout matrix without weight decay, everything else with it."""
    no_decay = [p for p in model.embed.parameters() if p.requires_grad]
    if model.output.output.weight.requires_grad:
        no_decay.append(model.output.output.weight)
    no_decay_ids = {id(p) for p in no_decay}
    if not no_decay or not no_decay_ids <= {id(p) for p in model.parameters()}:
        raise ValueError("no_decay_embed_readout requires unflattened embedding and readout parameters (fsdp2)")
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    return [
        {"name": "decay", "params": decay, "weight_decay": weight_decay},
        {"name": "no_decay", "params": no_decay, "weight_decay": 0.0},
    ]


def mup_param_groups(model: nn.Module, cfg: OptimConf, mup: MuPConf, weight_decay: float):
    """One AdamW group per Complete(d)P role (arXiv:2512.22382, Table 1)."""
    m_n = mup.width_multiplier
    m_l = mup.depth_multiplier
    if m_n is None or m_l is None:
        raise ValueError("mup multipliers are unset; call MuPConf.resolve() first")
    alpha = 1.0 if mup.alpha is None else mup.alpha

    # (lr, eps, weight_decay) multipliers per role
    scales = {
        "embed": (1.0, 1.0 / m_n, 1.0),
        "hidden_matrix": (m_l ** (alpha - 1) / m_n, m_l ** -alpha / m_n, m_n),
        "hidden_vector": (m_l ** (alpha - 1), m_l ** -alpha / m_n, 1.0),
        "final_norm": (1.0, 1.0, 1.0),
        "readout": (1.0 / m_n, 1.0, m_n),
    }
    groups = []
    for role, params in param_roles(model).items():
        if not params:
            continue
        lr_scale, eps_scale, wd_scale = scales[role]
        role_weight_decay = weight_decay * wd_scale
        if cfg.no_decay_embed_readout and role in ("embed", "readout"):
            role_weight_decay = 0.0
        groups.append({
            "name": role,
            "params": params,
            "lr": cfg.lr * lr_scale,
            "eps": cfg.epsilon * eps_scale,
            "weight_decay": role_weight_decay,
        })
    return groups


def optimizer_layout(config: Dict[str, Any]) -> str:
    """Param group layout that build_optimizer uses for a TrainerConf dict (as saved in config.json)."""
    if config.get("mup") is not None:
        return "mup"
    if config["optim"].get("no_decay_embed_readout", False):
        return "no_decay_embed_readout"
    return "default"


def get_group_hparams(optimizer):
    """Per-group hyperparameters that an optimizer state reload would overwrite."""
    names = ("initial_lr", "eps", "weight_decay", "betas")
    return [{k: group[k] for k in names if k in group} for group in optimizer.param_groups]


def set_group_hparams(optimizer, group_hparams) -> None:
    """Restore per-group hyperparameters saved by get_group_hparams."""
    if len(group_hparams) != len(optimizer.param_groups):
        raise RuntimeError(
            f"Optimizer has {len(optimizer.param_groups)} param groups, expected {len(group_hparams)}"
        )
    for group, hparams in zip(optimizer.param_groups, group_hparams):
        changed = {k: (group.get(k), v) for k, v in hparams.items() if group.get(k) != v}
        if changed:
            logger.info(f"Param group {group.get('name')}: using configured values over reloaded ones: {changed}")
        group.update(hparams)


def build_optimizer(
    model: nn.Module,
    cfg: OptimConf,
    total_steps: int,
    param_dtype: str,
    mup: Optional[MuPConf] = None,
):
    logger.info("Starting build of optimizer...")
    weight_decay = base_weight_decay(cfg, total_steps)
    if cfg.tau_epoch is not None:
        logger.info(
            f"Base weight decay from tau_epoch: 1 / ({cfg.lr} * {cfg.tau_epoch} * {total_steps}) = {weight_decay:.6g}"
        )

    if mup is not None:
        params = mup_param_groups(model, cfg, mup, weight_decay)
    elif cfg.no_decay_embed_readout:
        params = no_decay_embed_readout_groups(model, weight_decay)
    else:
        params = model.parameters()

    optimizer = AdamW(
        params,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=weight_decay,
        eps=cfg.epsilon,
    )
    for group in optimizer.param_groups:
        if "name" in group:
            logger.info(
                f"Param group {group['name']}: {len(group['params'])} tensors, "
                f"lr={group['lr']:.6g}, eps={group['eps']:.6g}, weight_decay={group['weight_decay']:.6g}"
            )

    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
