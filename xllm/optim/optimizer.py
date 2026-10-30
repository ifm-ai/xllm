from logging import getLogger
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW

from xllm.config import OptimConf, RolesConf
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


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str,
                    roles: Optional[RolesConf] = None):
    """roles: per-role hyperparameters (xllm/optim/roles.py); at their defaults, the plain param groups."""
    logger.info("Starting build of optimizer...")
    if roles == RolesConf():
        roles = None
    if cfg.optimizer == "muon":
        from .muon import build_muon
        optimizer = build_muon(model, cfg, roles)
    else:
        assert roles is None, "roles require optim.optimizer=muon"
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
            eps=cfg.epsilon,
        )

    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler


def restore_optimizer_config(optimizer, cfg: OptimConf, roles: Optional[RolesConf] = None):
    """Training resume policy: explicitly apply the configured Muon/AdamW hyperparameters.

    Loading the optimizer alone restores checkpoint values. Training calls this
    after loading state; the scheduler separately determines the current LRs.
    The standalone AdamW path retains upstream's existing resume behavior.
    A role group gets its role's scaled values (xllm/optim/roles.py).
    """
    if cfg.optimizer != "muon":
        return
    from .muon import group_hparams
    from .roles import role_hparams
    for group in optimizer.param_groups:
        if "role" not in group:
            lr = cfg.muon_lr if group["algorithm"] == "muon" else cfg.lr
            values = dict(group_hparams(cfg, group["algorithm"]), eps=cfg.epsilon, initial_lr=lr)
        else:
            values = role_hparams(cfg, roles, group["role"])
            values["initial_lr"] = values.pop("lr")
        changed = {key: (group.get(key), value) for key, value in values.items() if group.get(key) != value}
        if changed:
            logger.info(f"Resume {group.get('role', group['algorithm'])}: configured values replace checkpoint values "
                        f"(old, new): {changed}")
        group.update(values)
