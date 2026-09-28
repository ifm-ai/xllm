from typing import Tuple, Optional
import torch
from torch import Tensor


def logaddexp_backward(
    grad: Tensor,
    inp1: Tensor,
    inp2: Tensor,
    out: Tensor,
) -> Tuple[Tensor, Tensor]:
    grad1 = grad * torch.exp(inp1 - out)
    grad2 = grad * torch.exp(inp2 - out)
    return grad1, grad2


def logsumexp_backward(
    grad: Tensor,
    inp: Tensor,
    out: Tensor,
    dim: int,
    grad_out: Optional[Tensor] = None
) -> Tensor:
    if dim < 0:
        dim = inp.dim() + dim
    grad = grad.unsqueeze(dim)
    out = out.unsqueeze(dim)
    return torch.mul(grad, torch.exp(inp - out), out=grad_out)
