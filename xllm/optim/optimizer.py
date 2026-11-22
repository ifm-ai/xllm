from logging import getLogger

import torch
from torch import nn
from torch.optim import AdamW

from typing import Optional

from xllm.config import OptimConf, RoleConf, RolesConf
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
    """roles: per-role hyperparameters (xllm/optim/roles.py), Muon only; unset: the plain groups."""
    logger.info("Starting build of optimizer...")
    if cfg.optimizer == "muon":
        from .muon import build_muon
        optimizer = build_muon(model, cfg, roles)
    else:
        assert roles is None or not roles.is_set(), "roles require optim.optimizer=muon"
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.adamw.lr,
            betas=(cfg.adamw.beta1, cfg.adamw.beta2),
            weight_decay=cfg.adamw.weight_decay,
            eps=cfg.adamw.epsilon,
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
    With roles (xllm/optim/roles.py), every group's values are its role's, the Muon groups included.
    """
    if cfg.optimizer != "muon":
        return
    from .muon import adamw_group, muon_group
    if any("role" in group for group in optimizer.param_groups) and (roles is None or not roles.is_set()):
        raise ValueError("restore_optimizer_config: the optimizer has per-role groups; pass the roles it was built with")
    roles = roles or RolesConf()
    for group in optimizer.param_groups:
        r = getattr(roles, group["role"]) if "role" in group else RoleConf()
        if group["algorithm"] == "muon":
            configured = muon_group(cfg, [], lr=r.lr, weight_decay=r.weight_decay)
            configured["eps"] = cfg.adamw.epsilon  # Muon's Newton-Schulz eps
        else:
            configured = adamw_group(cfg, [], lr=r.lr, eps=r.eps, weight_decay=r.weight_decay)
        values = {key: value for key, value in configured.items() if key not in ("params", "algorithm")}
        values["initial_lr"] = values.pop("lr")
        changed = {key: (group.get(key), value) for key, value in values.items() if group.get(key) != value}
        if changed:
            logger.info(f"Resume {group.get('role', group['algorithm'])}: configured values replace checkpoint values "
                        f"(old, new): {changed}")
        group.update(values)
