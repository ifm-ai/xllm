"""Parameter roles (AdamWConf's role fields, xllm/config.py): per-role AdamW lr / epsilon / weight decay with Muon.

A module declares the role of each parameter in its __init__ (register_role). The tag lives on the module, so it
survives FSDP2 replacing the parameters, as register_muon's specs; a parameter no module tags is in role `other`.
param_groups() builds the optimizer's groups: the Muon matrices in one group at optim.muon's values, and one AdamW
group per role, at optim.adamw's values unless --optim.adamw.<role>.* replaces them. No role set: the optimizer's
two plain groups, as before.
"""
from logging import getLogger
from typing import Dict, List

from torch import nn

from xllm.config import AdamWRoleConf, OptimConf

logger = getLogger()
# every role a module may tag. optim.adamw holds the values of the roles AdamW steps; hidden_matrix is Muon's, and
# its values are optim.muon's, so it is not a field of AdamWConf.
ROLES = ("embed", "hidden_matrix", "hidden_vector", "final_norm", "readout", "other")


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


def param_groups(model: nn.Module, cfg: OptimConf, muon_params: List[nn.Parameter],
                 adamw_params: List[nn.Parameter]) -> List[Dict]:
    """The optimizer's param groups with roles: the Muon matrices in one group at optim.muon's values, then one
    AdamW group per role that has parameters, at optim.adamw's values unless the role replaces them."""
    from .muon import adamw_group, muon_group
    role_of = param_roles(model)
    untagged = [n for n, p in model.named_parameters() if role_of[id(p)] == "other"]
    if untagged:
        logger.warning(f"roles: {len(untagged)} parameters have no role, so step as `other`: {untagged[:5]} ...")

    muon_roles = {role_of[id(p)] for p in muon_params}
    if not muon_roles <= {"hidden_matrix", "other"}:
        raise ValueError(f"roles: Muon steps only hidden_matrix (and untagged) weights, got {sorted(muon_roles)}")

    groups = [muon_group(cfg, muon_params)]
    for role in ROLES:
        params = [p for p in adamw_params if role_of[id(p)] == role]
        if params:
            # hidden_matrix is not a field of AdamWConf: a matrix register_muon missed steps at the plain values
            r = getattr(cfg.adamw, role, AdamWRoleConf())
            group = adamw_group(cfg, params, lr=r.lr, eps=r.epsilon, weight_decay=r.weight_decay)
            group["role"] = role
            groups.append(group)

    logger.info("roles: " + "; ".join(
        f"{g.get('role', 'Muon')} ({len(g['params'])} tensors): lr {g['lr']:.6g}, eps {g.get('eps', cfg.adamw.epsilon):.6g}, "
        f"weight_decay {g['weight_decay']:.6g}" for g in groups))
    return groups
