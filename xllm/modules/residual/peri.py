from torch import Tensor
from torch import nn

from xllm.modules.rms_norm import GroupRMSNorm


class PeriResidual(nn.Module):
    """
    Peri-LN Residual: Y = Norm(H) + Residual
    Ref: https://arxiv.org/abs/2502.02732
    """
    def __init__(self, rc_id: int, num_features: int, num_groups: int, eps: float):
        super().__init__()
        self.rc_id = rc_id
        self.num_features = num_features
        self.norm = GroupRMSNorm(
            num_features,
            num_groups=num_groups,
            elementwise_affine=True,
            eps=eps,
            memory_efficient=False,
        )

    def forward(
        self, h, residual, *args, **kwargs
    ) -> Tensor:
        return self.norm(h) + residual
