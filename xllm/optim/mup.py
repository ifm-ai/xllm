"""Complete(d)P param groups for Muon (MuPConf in xllm/config.py, arXiv:2512.22382).

Muon steps the hidden matrices with lr x m_L^(alpha - 1); its spectral_norm update scale already
transfers across widths. The output projections (attention wo, FFN fc2) hold the folded depth scale c:
their update is x c (update_scales) but their decay lr * wd is not, and their gradients count x c in
the clipping norm (clip_grad_scales). With their init x c (MuPConf), training matches a residual
multiplier c on the branches.

AdamW steps everything else, one group per role, its (lr, eps, weight_decay) x
    embed          1                    m_N^-1                   1
    hidden_vector  m_L^(alpha - 1)      m_N^-1 m_L^-alpha        1      norm gains and biases in the blocks
    final_norm     1                    1                        1
    readout        m_N^-1               1                        m_N
"""
from typing import Dict, List, Tuple

from torch import nn

from xllm.config import MuPConf, OptimConf
from .muon import group_hparams


def _scales(mup: MuPConf) -> Dict[str, Tuple[float, float, float]]:
    """(lr, eps, weight_decay) multipliers of every role."""
    m_n, m_l, alpha = mup.width_multiplier, mup.depth_multiplier, mup.alpha
    return {
        "muon": (m_l ** (alpha - 1), 1.0, 1.0),
        "embed": (1.0, 1 / m_n, 1.0),
        "hidden_vector": (m_l ** (alpha - 1), m_l ** -alpha / m_n, 1.0),
        "final_norm": (1.0, 1.0, 1.0),
        "readout": (1 / m_n, 1.0, m_n),
    }


def role_hparams(cfg: OptimConf, mup: MuPConf, role: str) -> Dict:
    """A param group's hyperparameters by role: the configured ones, lr / eps / weight_decay scaled."""
    lr_scale, eps_scale, wd_scale = _scales(mup)[role]
    algorithm = "muon" if role == "muon" else "adamw"
    lr = cfg.muon_lr if algorithm == "muon" else cfg.lr
    values = group_hparams(cfg, algorithm)
    return dict(values, lr=lr * lr_scale, eps=cfg.epsilon * eps_scale, weight_decay=values["weight_decay"] * wd_scale)


def out_projections(model: nn.Module) -> List[nn.Parameter]:
    """attention wo and FFN fc2 of every block: the weights that hold the folded depth scale c."""
    return [p for layer in model.layers for p in (layer.attention.wo.weight, layer.nffn.fc2.weight)]


def clip_grad_scales(model: nn.Module, mup: MuPConf) -> Dict[int, float]:
    """id(param) -> factor on its gradient in the clipping norm: c for the output projections, whose
    gradient is the unfolded weight's / c."""
    return {id(p): mup.branch_scale for p in out_projections(model)}


def param_groups(
    model: nn.Module, cfg: OptimConf, mup: MuPConf, muon_params: List[nn.Parameter]
) -> Tuple[Dict, List[Dict], List[float]]:
    """The Muon group's hyperparameters, the AdamW groups (one per role) and the Muon update scales
    (c for the output projections, 1 for the other matrices)."""
    muon = {id(p) for p in muon_params}
    out = {id(p) for p in out_projections(model)}
    if not out <= muon:
        raise ValueError("mup: attention wo and FFN fc2 must be Muon weights (register_muon)")
    unassigned = [f"layers.{name}" for name, p in model.layers.named_parameters() if id(p) not in muon and p.ndim > 1]
    if unassigned:
        raise ValueError(f"mup: no Complete(d)P role for hidden matrices that are not Muon weights: {unassigned}")
    roles = {
        "embed": list(model.embed.parameters()),
        "hidden_vector": [p for p in model.layers.parameters() if id(p) not in muon],
        "final_norm": list(model.output.final_norm.parameters()),
        "readout": [model.output.output.weight],
    }
    assigned = [id(p) for params in roles.values() for p in params] + list(muon)
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in model.parameters()}:
        raise ValueError("mup: every parameter needs exactly one Complete(d)P role")
    adamw_groups = [dict(params=params, role=role, **role_hparams(cfg, mup, role))
                    for role, params in roles.items() if params]
    update_scales = [mup.branch_scale if id(p) in out else 1.0 for p in muon_params]
    return dict(role="muon", **role_hparams(cfg, mup, "muon")), adamw_groups, update_scales
