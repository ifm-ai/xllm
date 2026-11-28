from logging import getLogger

import torch
from torch import nn
from torch.optim import AdamW

from xllm.config import AdamWRoleConf, OptimConf
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


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str):
    """Per-role AdamW values live in cfg.adamw (xllm/optim/roles.py); unset: the plain groups."""
    logger.info("Starting build of optimizer...")
    if cfg.optimizer == "muon":
        from .muon import build_muon
        optimizer = build_muon(model, cfg)
    else:
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


def restore_optimizer_config(optimizer, cfg: OptimConf):
    """Training resume policy: explicitly apply the configured Muon/AdamW hyperparameters.

    Loading the optimizer alone restores checkpoint values. Training calls this
    after loading state; the scheduler separately determines the current LRs.
    The standalone AdamW path retains upstream's existing resume behavior.
    With per-role AdamW values (optim.adamw.<role>.*), an AdamW group's values are its role's; the Muon group is
    optim.muon's. cfg carries the roles, so a resume cannot lose them.
    """
    if cfg.optimizer != "muon":
        return
    from .muon import adamw_group, muon_group
    for group in optimizer.param_groups:
        if group["algorithm"] == "muon":
            configured = muon_group(cfg, [])
            configured["eps"] = cfg.adamw.epsilon  # Muon's Newton-Schulz eps
        else:
            r = getattr(cfg.adamw, group["role"], AdamWRoleConf()) if "role" in group else AdamWRoleConf()
            configured = adamw_group(cfg, [], lr=r.lr, eps=r.epsilon, weight_decay=r.weight_decay)
        values = {key: value for key, value in configured.items() if key not in ("params", "algorithm")}
        values["initial_lr"] = values.pop("lr")
        changed = {key: (group.get(key), value) for key, value in values.items() if group.get(key) != value}
        if changed:
            logger.info(f"Resume {group.get('role', group['algorithm'])}: configured values replace checkpoint values "
                        f"(old, new): {changed}")
        group.update(values)
