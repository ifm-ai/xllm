"""Parameter roles (RolesConf in xllm/config.py): per-role lr / eps / weight decay factors, init std and
clipping-norm factor.

A module declares the role of each parameter in its __init__ (register_role). The tag lives on the module, so it
survives FSDP2 replacing the parameters, as register_muon's specs; a parameter no module tags is in role `other`.
AdamW steps one param group per role; Muon scales each matrix's lr and weight decay by its role's (Muon's
lr_scales / wd_scales). MuPConf is one preset of the roles. Unset values change nothing: no re-init, the optimizer's
two groups (while no role's lr / eps / weight decay factor differs from 1), the plain clipping norm.
"""
from contextlib import contextmanager
from dataclasses import fields
from logging import getLogger
from typing import Dict, List, Optional, Tuple

from torch import nn
from torch.optim import Optimizer

from xllm.config import MUON_ROLES, OptimConf, RolesConf
from xllm.utils import get_init_fn

logger = getLogger()
ROLES = tuple(f.name for f in fields(RolesConf))
_init_stds: Dict[str, Optional[float]] = {}


@contextmanager
def init_roles(roles: RolesConf):
    """Build the model inside: register_role then initializes the parameters of the roles that set an init_std."""
    outer = dict(_init_stds)
    _init_stds.update({role: getattr(roles, role).init_std for role in ROLES})
    try:
        yield
    finally:
        _init_stds.clear()
        _init_stds.update(outer)


def register_role(module: nn.Module, name: str, role: str):
    """Declare module.<name>'s role; call it in the module's __init__, after creating the parameter. A None
    parameter (a norm without affine) is skipped. Inside init_roles, a role's init_std re-initializes it (0: zeros)."""
    assert role in ROLES, f"{type(module).__name__}.{name}: unknown role {role!r}, not in {ROLES}"
    param = getattr(module, name)
    if param is None:
        return
    tags = vars(module).setdefault("param_roles", {})
    assert name not in tags, f"{type(module).__name__}.{name} already has role {tags[name]}"
    std = _init_stds.get(role)
    if std is not None:
        import xllm.distributed as xd
        # a tensor-parallel shard would draw the same values on every rank
        assert xd.get_model_parallel_world_size() == 1, f"{role}.init_std needs model_parallel_size 1"
        if std == 0:
            nn.init.zeros_(param)
        else:
            get_init_fn("gaussian", std=std)(param)
    tags[name] = role


def param_roles(model: nn.Module) -> Dict[int, str]:
    """id(param) -> role of every parameter, `other` where no module tags it. Call it after wrapping: they are
    looked up by name."""
    roles = {}
    for fqn, m in model.named_modules():
        for name, role in vars(m).get("param_roles", {}).items():
            if roles.setdefault(id(getattr(m, name)), role) != role:
                raise ValueError(f"{fqn}.{name} has two roles: {roles[id(getattr(m, name))]} and {role}")
    return {id(p): roles.get(id(p), "other") for p in model.parameters()}


def require_roles(model: nn.Module):
    """mup scales every parameter by its role: none may be left to `other`."""
    roles = param_roles(model)
    untagged = [name for name, p in model.named_parameters() if roles[id(p)] == "other"]
    if untagged:
        raise ValueError(f"mup needs a role on every parameter (register_role), these have none: {untagged}")


def grouped(roles: RolesConf) -> bool:
    """Whether the optimizer steps per-role groups: some role's lr, eps or weight decay is not 1. init_std and
    clip_scale alone keep the two plain groups, so a resume without them finds the groups it saved."""
    return any(getattr(roles, role).factor(name) != 1 for role in ROLES for name in ("lr", "eps", "weight_decay"))


def role_hparams(cfg: OptimConf, roles: RolesConf, role: str) -> Dict:
    """An AdamW role group's hyperparameters: the configured ones, lr / eps / weight_decay scaled."""
    from .muon import group_hparams
    scale = getattr(roles, role)
    values = group_hparams(cfg, "adamw")
    return dict(values, lr=cfg.lr * scale.factor("lr"), eps=cfg.epsilon * scale.factor("eps"),
                weight_decay=values["weight_decay"] * scale.factor("weight_decay"))


def param_groups(
    model: nn.Module, cfg: OptimConf, roles: RolesConf, muon_params: List[nn.Parameter]
) -> Tuple[List[Dict], List[float], List[float]]:
    """The AdamW param groups, one per role, and the Muon matrices' lr and weight decay factors."""
    tags = param_roles(model)
    muon = {id(p) for p in muon_params}
    # the MUON_ROLES tag Muon weights only, and a Muon weight is in one of them or untagged
    mismatched = [name for name, p in model.named_parameters()
                  if (id(p) in muon) != (tags[id(p)] in MUON_ROLES) and tags[id(p)] != "other"]
    if mismatched:
        raise ValueError(f"roles: {MUON_ROLES} must be exactly the tagged Muon weights (register_muon): {mismatched}")
    if any(getattr(roles, role).factor("eps") != 1 for role in MUON_ROLES):
        raise ValueError(f"roles: Muon's eps is not scaled per role, {MUON_ROLES} must keep eps 1")
    adamw = {role: [p for p in model.parameters() if id(p) not in muon and tags[id(p)] == role] for role in ROLES}
    groups = [dict(params=ps, role=role, **role_hparams(cfg, roles, role)) for role, ps in adamw.items() if ps]
    scales = [getattr(roles, tags[id(p)]) for p in muon_params]
    return groups, [s.factor("lr") for s in scales], [s.factor("weight_decay") for s in scales]


def log_param_groups(model: nn.Module, roles: RolesConf, optimizer: Optimizer):
    """One line per role: the Muon roles' factors, then the AdamW role groups' values."""
    tags = param_roles(model)
    muon_roles = [tags[id(p)] for p in optimizer.param_groups[0]["params"]]
    for role in dict.fromkeys(muon_roles):
        scale = getattr(roles, role)
        logger.info(f"role {role} (Muon): {muon_roles.count(role)} matrices, lr x {scale.factor('lr'):.6g}, "
                    f"weight_decay x {scale.factor('weight_decay'):.6g}")
    for group in optimizer.param_groups[1:]:
        logger.info(f"role {group['role']} (AdamW): {len(group['params'])} tensors, lr {group['lr']:.6g}, "
                    f"eps {group['eps']:.6g}, weight_decay {group['weight_decay']:.6g}")


def group_lrs(optimizer: Optimizer, roles: RolesConf) -> Dict[str, float]:
    """Muon's current lrs by name: muon and adamw, or with role groups muon and one per role (the Muon roles' as
    muon x their factor)."""
    lrs = {group.get("role") or group["algorithm"]: float(group["lr"]) for group in optimizer.param_groups}
    if "adamw" not in lrs:
        lrs.update({role: lrs["muon"] * getattr(roles, role).factor("lr") for role in MUON_ROLES})
    return lrs


def clip_grad_scales(model: nn.Module, roles: RolesConf) -> Optional[Dict[int, float]]:
    """id(param) -> its role's clip_scale for the roles that set one other than 1, or None if none does (the model
    unread)."""
    scales = {role: getattr(roles, role).factor("clip_scale") for role in ROLES}
    scales = {role: scale for role, scale in scales.items() if scale != 1}
    return {i: scales[role] for i, role in param_roles(model).items() if role in scales} if scales else None
