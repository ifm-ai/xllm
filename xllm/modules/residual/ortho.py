import torch
from torch import nn, Tensor
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_world_size,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.fused_ops import rejection


class OrthoResidual(nn.Module):
    """
    Orthogonal Residual in: https://arxiv.org/abs/2505.11881
    """
    def __init__(
        self,
        rc_id: int,
        model_dim: int,
        num_heads: int,
        eps: float,
    ):
        super().__init__()
        self.rc_id = rc_id

        model_parallel_world_size = get_model_parallel_world_size()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = divide_and_check_no_remainder(model_dim, num_heads)
        self.n_local_heads = divide_and_check_no_remainder(num_heads, model_parallel_world_size)

        self.norm = GroupRMSNorm(
            model_dim,
            num_groups=num_heads,
            elementwise_affine=False,
            eps=eps,
            memory_efficient=True,
        )

    def forward(
        self, h, residual, *args, **kwargs
    ) -> Tensor:
        bsz, slen, _ = h.shape
        # B*L x N x S
        k = rearrange(self.norm(residual), 'b l (n s) -> (b l) n s', n=self.n_local_heads)
        h = rearrange(h, 'b l (n s) -> (b l) n s', n=self.n_local_heads)
        # B*L x N x S
        rej = rejection(h, k)
        # B x L x D
        out = rearrange(rej, '(b l) n s -> b l (n s)', b=bsz) + residual
        return out

    def extra_repr(self) -> str:
        return 'dim={}, heads={} ({}), head_dim={}'.format(
            self.model_dim, self.num_heads, self.n_local_heads, self.head_dim
        )
