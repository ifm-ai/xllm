from typing import Any, Tuple

from torch import Tensor
from torch import nn


class BaseResidual(nn.Module):
    """
    Base Residual in ResNet: Y = H + Residual
    Ref: https://arxiv.org/abs/1512.03385
    """
    def __init__(self, rc_id: int, num_features: int,):
        super().__init__()
        self.rc_id = rc_id
        self.num_features = num_features

    def read(self, x: Tensor) -> Tuple[Tensor, Any]:
        """The sublayer's input from the residual state and the state `forward` takes back as `state=`:
        (x, None) unless the residual keeps more than the sublayer reads (hyper-connections)."""
        return x, None

    def forward(
        self, h, residual, *args, **kwargs
    ) -> Tensor:
        return h + residual

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.num_features)
