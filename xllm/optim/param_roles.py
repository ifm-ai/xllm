from typing import Dict, List

from torch import nn


PARAM_ROLES = ("embed", "hidden_matrix", "hidden_vector", "final_norm", "readout")


def _trainable(parameters) -> List[nn.Parameter]:
    return [p for p in parameters if p.requires_grad]


def param_roles(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """Assign every trainable parameter of a dense Transformer to one Complete(d)P role.

    Roles: input embedding, hidden matrices (ndim > 1) and hidden biases/norms
    (ndim <= 1) inside the blocks, the final norm, and the readout matrix.
    """
    hidden = _trainable(model.layers.parameters())
    roles = {
        "embed": _trainable(model.embed.parameters()),
        "hidden_matrix": [p for p in hidden if p.ndim > 1],
        "hidden_vector": [p for p in hidden if p.ndim <= 1],
        "final_norm": _trainable(model.output.final_norm.parameters()),
        "readout": _trainable([model.output.output.weight]),
    }

    assigned = [id(p) for params in roles.values() for p in params]
    expected = {id(p) for p in _trainable(model.parameters())}
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError("param_roles must assign every trainable parameter to exactly one role")
    return roles
