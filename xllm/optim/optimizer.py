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
    """roles: per-role hyperparameters (xllm/optim/roles.py); unset: the plain groups."""
    logger.info("Starting build of optimizer...")
    if cfg.optimizer == "muon":
        from .muon import build_muon
        optimizer = build_muon(model, cfg, roles)
    else:
        if roles is not None and roles.is_set():
            from .roles import adamw_param_groups
            params = adamw_param_groups(model, cfg, roles)
        else:
            params = model.parameters()
        optimizer = AdamW(
            params,
            lr=cfg.adamw.lr,
            betas=(cfg.adamw.beta1, cfg.adamw.beta2),
            weight_decay=cfg.adamw.weight_decay,
            eps=cfg.adamw.epsilon,
        )

    # the groups as built: loading a checkpoint copies its groups' keys by position (DCP), which must not change them
    optimizer.built_groups = [(group.get("algorithm"), group.get("role")) for group in optimizer.param_groups]

    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler


def restore_optimizer_config(optimizer, cfg: OptimConf, roles: Optional[RolesConf] = None):
    """Training resume policy: explicitly apply the configured Muon/AdamW hyperparameters.

    Loading the optimizer alone restores checkpoint values. Training calls this
    after loading state; the scheduler separately determines the current LRs.
    The standalone AdamW path without roles retains upstream's existing resume behavior.
    With roles (xllm/optim/roles.py), a group's values are its role's: the Muon group's hidden_matrix's.
    """
    built = getattr(optimizer, "built_groups", None) or [(g.get("algorithm"), g.get("role")) for g in optimizer.param_groups]
    has_roles = any(role is not None for _, role in built)
    if cfg.optimizer != "muon" and not has_roles and (roles is None or not roles.is_set()):
        return
    from .muon import adamw_group, muon_group
    if has_roles and (roles is None or not roles.is_set()):
        raise ValueError("restore_optimizer_config: the optimizer has per-role groups; pass the roles it was built with")
    roles = roles or RolesConf()
    for group, (algorithm, role) in zip(optimizer.param_groups, built):
        if algorithm == "muon":
            r = roles.hidden_matrix
            configured = muon_group(cfg, [], lr=r.lr, weight_decay=r.weight_decay)
            configured["eps"] = cfg.adamw.epsilon  # Muon's Newton-Schulz eps
        else:
            r = getattr(roles, role) if role is not None else RoleConf()
            configured = adamw_group(cfg, [], lr=r.lr, eps=r.eps, weight_decay=r.weight_decay)
        values = {key: value for key, value in configured.items() if key != "params"}  # with the group's algorithm
        if role is not None:
            values["role"] = role
        values["initial_lr"] = values.pop("lr")
        changed = {key: (group.get(key), value) for key, value in values.items() if group.get(key) != value}
        if changed:
            logger.info(f"Resume {role or algorithm}: configured values replace checkpoint values (old, new): {changed}")
        group.update(values)
