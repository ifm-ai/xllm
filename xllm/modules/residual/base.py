from torch import Tensor
from torch import nn


class BaseResidual(nn.Module):
    """
    Base Residual in ResNet: Y = H * multiplier + Residual
    Ref: https://arxiv.org/abs/1512.03385
    """
    def __init__(self, rc_id: int, num_features: int, multiplier: float = 1.0):
        super().__init__()
        self.rc_id = rc_id
        self.num_features = num_features
        self.multiplier = multiplier

    def forward(
        self, h, residual, *args, **kwargs
    ) -> Tensor:
        if self.multiplier != 1.0:
            h = h * self.multiplier
        return h + residual

    def extra_repr(self) -> str:
        if self.multiplier != 1.0:
            return 'dim={}, multiplier={}'.format(self.num_features, self.multiplier)
        return 'dim={}'.format(self.num_features)
