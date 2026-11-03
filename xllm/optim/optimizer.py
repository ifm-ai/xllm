from logging import getLogger
from typing import Dict, List, Optional

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
    """roles: per-role hyperparameters (xllm/optim/roles.py); the plain param groups unless some role scales the lr,
    eps or weight decay."""
    from .roles import grouped
    logger.info("Starting build of optimizer...")
    if roles is not None and not grouped(roles):
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


def param_group_names(optimizer) -> List[str]:
    """Saved with the training state, to check a resume against (check_param_groups)."""
    return [group.get("role", group["algorithm"]) for group in optimizer.param_groups]


def check_param_groups(optimizer, training_state_dict: Dict, source: str):
    """Optimizer state loads only into the param groups it was saved with; check before loading it."""
    current = param_group_names(optimizer)
    saved = training_state_dict.get("param_groups")
    if saved is None:  # older checkpoints tell only the count, from the scheduler
        saved, current = len(training_state_dict["scheduler"]["base_lrs"]), len(current)
    if saved != current:
        raise ValueError(
            f"{source} has optimizer param groups {saved}, this run {current}. Use the --roles it was saved "
            f"with (some role's lr, eps or weight_decay other than 1 makes per-role groups), or load only "
            f"its weights (--base_model_dir without --reload_base_optim_state)."
        )
