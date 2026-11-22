"""Parameter roles (RolesConf in xllm/config.py): per-role lr / eps / weight decay with Muon.

A module declares the role of each parameter in its __init__ (register_role). The tag lives on the module, so it
survives FSDP2 replacing the parameters, as register_muon's specs; a parameter no module tags is in role `other`.
param_groups() builds the optimizer's groups: the Muon matrices in one group at hidden_matrix's values, and one
AdamW group per role. A value a role leaves unset is the configured optim.* one. No --roles.*: the optimizer's two
plain groups, as before.
"""
from dataclasses import fields
from logging import getLogger
from typing import Dict, List

from torch import nn

from xllm.config import OptimConf, RolesConf

logger = getLogger()
ROLES = tuple(f.name for f in fields(RolesConf))


def register_role(module: nn.Module, name: str, role: str):
    """Declare module.<name>'s role; call it in the module's __init__. A None parameter (a norm without affine) is
    skipped."""
    assert role in ROLES, f"{type(module).__name__}.{name}: unknown role {role!r}, not in {ROLES}"
    if getattr(module, name) is None:
        return
    tags = vars(module).setdefault("param_roles", {})
    assert name not in tags, f"{type(module).__name__}.{name} already has role {tags[name]}"
    tags[name] = role


def param_roles(model: nn.Module) -> Dict[int, str]:
    """id(param) -> role of every parameter, `other` where no module tags it."""
    tagged = {}
    for fqn, m in model.named_modules():
        for name, role in vars(m).get("param_roles", {}).items():
            if tagged.setdefault(id(getattr(m, name)), role) != role:
                raise ValueError(f"{fqn}.{name} has two roles: {tagged[id(getattr(m, name))]} and {role}")
    return {id(p): tagged.get(id(p), "other") for p in model.parameters()}


def param_groups(model: nn.Module, cfg: OptimConf, roles: RolesConf, muon_params: List[nn.Parameter],
                 adamw_params: List[nn.Parameter]) -> List[Dict]:
    """The optimizer's param groups with roles: the Muon matrices in one group at hidden_matrix's values, then one
    AdamW group per role that has parameters. A value the role leaves unset is the configured optim.* one."""
    from .muon import adamw_group, muon_group
    role_of = param_roles(model)
    untagged = [n for n, p in model.named_parameters() if role_of[id(p)] == "other"]
    if untagged:
        logger.warning(f"roles: {len(untagged)} parameters have no role, so step as `other`: {untagged[:5]} ...")

    muon_roles = {role_of[id(p)] for p in muon_params}
    if not muon_roles <= {"hidden_matrix", "other"}:
        raise ValueError(f"roles: Muon steps only hidden_matrix (and untagged) weights, got {sorted(muon_roles)}")
    if "other" in muon_roles and (roles.other.lr is not None or roles.other.weight_decay is not None):
        raise ValueError("roles.other: untagged Muon weights (MoE) step in the Muon group, at hidden_matrix's values")
    if roles.hidden_matrix.eps is not None:
        raise ValueError("roles.hidden_matrix.eps: Muon has no eps to set")

    matrix = roles.hidden_matrix
    groups = [muon_group(cfg, muon_params, lr=matrix.lr, weight_decay=matrix.weight_decay)]
    for role in ROLES:
        params = [p for p in adamw_params if role_of[id(p)] == role]
        if params:
            r = getattr(roles, role)
            group = adamw_group(cfg, params, lr=r.lr, eps=r.eps, weight_decay=r.weight_decay)
            group["role"] = role
            groups.append(group)

    logger.info("roles: " + "; ".join(
        f"{g.get('role', 'Muon')} ({len(g['params'])} tensors): lr {g['lr']:.6g}, eps {g.get('eps', cfg.adamw.epsilon):.6g}, "
        f"weight_decay {g['weight_decay']:.6g}" for g in groups))
    return groups
