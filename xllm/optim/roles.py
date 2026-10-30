"""Parameter roles (RolesConf in xllm/config.py): per-role lr / eps / weight decay factors, init std and
clipping-norm factor.

A module declares the role of each parameter in its __init__ (register_role). The tag lives on the module, so it
survives FSDP2 replacing the parameters, as register_muon's specs. AdamW steps one param group per role; Muon
scales each matrix's lr and weight decay by its role's (Muon's lr_scales / wd_scales). MuPConf is one preset of
the roles. At their defaults nothing changes: no re-init, the optimizer's two groups, the plain clipping norm.
"""
from contextlib import contextmanager
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

from torch import nn

from xllm.config import OptimConf, RolesConf
from xllm.utils import get_init_fn

ROLES = tuple(f.name for f in fields(RolesConf))
MUON_ROLES = ("hidden_matrix", "residual_out")  # exactly the Muon weights; the other roles are AdamW's
_init_stds: Dict[str, Optional[float]] = {}


@contextmanager
def init_roles(roles: RolesConf):
    """Build the model inside: register_role then initializes the parameters of the roles that set an init_std."""
    _init_stds.update({role: getattr(roles, role).init_std for role in ROLES})
    try:
        yield
    finally:
        _init_stds.clear()


def register_role(module: nn.Module, name: str, role: str):
    """Declare module.<name>'s role; call it in the module's __init__, after creating the parameter. A None
    parameter (a norm without affine) is skipped. Inside init_roles, a role's init_std re-initializes it."""
    assert role in ROLES, f"{type(module).__name__}.{name}: unknown role {role!r}, not in {ROLES}"
    param = getattr(module, name)
    if param is None:
        return
    std = _init_stds.get(role)
    if std is not None:
        import xllm.distributed as xd
        # a tensor-parallel shard would draw the same values on every rank
        assert xd.get_model_parallel_world_size() == 1, f"{role}.init_std needs model_parallel_size 1"
        get_init_fn("gaussian", std=std)(param)
    vars(module).setdefault("param_roles", {})[name] = role


def param_roles(model: nn.Module) -> Dict[int, str]:
    """id(param) -> role of the registered parameters. Call it after wrapping: they are looked up by name."""
    roles = {}
    for fqn, m in model.named_modules():
        for name, role in vars(m).get("param_roles", {}).items():
            if roles.setdefault(id(getattr(m, name)), role) != role:
                raise ValueError(f"{fqn}.{name} has two roles: {roles[id(getattr(m, name))]} and {role}")
    return roles


def role_hparams(cfg: OptimConf, roles: RolesConf, role: str) -> Dict:
    """An AdamW role group's hyperparameters: the configured ones, lr / eps / weight_decay scaled."""
    from .muon import group_hparams
    scale = getattr(roles, role)
    values = group_hparams(cfg, "adamw")
    return dict(values, lr=cfg.lr * scale.lr, eps=cfg.epsilon * scale.eps,
                weight_decay=values["weight_decay"] * scale.weight_decay)


def param_groups(
    model: nn.Module, cfg: OptimConf, roles: RolesConf, muon_params: List[nn.Parameter]
) -> Tuple[List[Dict], List[float], List[float]]:
    """The AdamW param groups, one per role, and the Muon matrices' lr and weight decay factors."""
    tags = param_roles(model)
    muon = {id(p) for p in muon_params}
    missing = [name for name, p in model.named_parameters() if id(p) not in tags]
    if missing:
        raise ValueError(f"roles: parameters without a role (register_role): {missing}")
    mismatched = [name for name, p in model.named_parameters() if (tags[id(p)] in MUON_ROLES) != (id(p) in muon)]
    if mismatched:
        raise ValueError(f"roles: {MUON_ROLES} must be exactly the Muon weights (register_muon): {mismatched}")
    if any(getattr(roles, role).eps != 1 for role in MUON_ROLES):
        raise ValueError(f"roles: Muon's eps is not scaled per role, {MUON_ROLES} must keep eps 1")
    adamw = {role: [p for p in model.parameters() if id(p) not in muon and tags[id(p)] == role] for role in ROLES}
    groups = [dict(params=ps, role=role, **role_hparams(cfg, roles, role)) for role, ps in adamw.items() if ps]
    scales = [getattr(roles, tags[id(p)]) for p in muon_params]
    return groups, [s.lr for s in scales], [s.weight_decay for s in scales]


def clip_grad_scales(model: nn.Module, roles: RolesConf) -> Optional[Dict[int, float]]:
    """id(param) -> its role's clip_scale for the roles that set one, or None if none does (the model unread)."""
    scales = {role: getattr(roles, role).clip_scale for role in ROLES if getattr(roles, role).clip_scale != 1}
    return {i: scales[role] for i, role in param_roles(model).items() if role in scales} if scales else None
