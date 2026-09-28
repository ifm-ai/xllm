from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from xllm.utils import get_init_fn
from xllm.distributed import (
    get_model_parallel_world_size,
    get_model_parallel_rank,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.modules.fused_ops import causal_conv1d


class CausalConv1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        kernel_size: int,
        bias: bool = False,
        apply_weight_normalization: bool = True,
        activation: Optional[str] = None,
        backend: str = 'triton',
    ):
        """
        Short convolution layer for efficient causal convolution operations.

        This class implements a depthwise 1D convolution with causal padding,
        designed for efficient sequence processing. It supports multiple backends (triton/fla)
        and optional activation functions.

        Args:
            num_features (int): Number of input/output features
            kernel_size (int): Size of the convolution kernel
            bias (bool, optional): Whether to include learnable bias. Defaults to False.
            apply_weight_normalization (bool, optional): Apply softmax to weight. Defaults to True.
            activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to None.
            backend (str, optional): Backend implementation ('triton' or 'fla'). Defaults to 'triton'.
        """
        super().__init__()

        self.num_features = num_features
        assert 0 < kernel_size <= 4, f"The kernel size of {kernel_size} is out of supported range in (0, 4]."
        self.kernel_size = kernel_size

        world_size = get_model_parallel_world_size()
        self.features_per_partition = divide_and_check_no_remainder(num_features, world_size)

        self.normalize_weight = apply_weight_normalization
        self.weight = nn.Parameter(torch.empty(self.features_per_partition, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.features_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if backend not in ('triton', 'fla'):
            raise ValueError(f"Unsupported causal conv1d backend: {backend}")
        self.backend = backend

        # init parameters
        self._init_parameters()

    def _init_parameters(self):
        init_fn = get_init_fn('gaussian', std=0.1)
        world_size = get_model_parallel_world_size()
        if world_size == 1:
            init_fn(self.weight)
            return None

        master_weight = torch.empty(self.num_features, self.kernel_size, dtype=self.weight.dtype, requires_grad=False)
        init_fn(master_weight)
        rank = get_model_parallel_rank()
        my_weight = torch.split(master_weight, self.features_per_partition, dim=0)[rank]
        with torch.no_grad():
            self.weight.copy_(my_weight)

        del master_weight
        return None

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        bos_mask: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (Tensor): (batch, seqlen, dim)
            initial_state (Optional[Tensor]): (batch, width - 1, dim)
            bos_mask (Optional[Tensor]): (batch, seqlen)
            output_final_state (bool): whether to output the final state of shape [batch, width - 1, dim]. Default: `False`.
            deterministic (bool, optional): use deterministic Triton gradients. Defaults to False.
        Return:
            out (Tensor): (batch, seqlen, dim)
            final_states (Optional[Tensor]): (batch, width - 1, dim)
        """

        w = F.softmax(self.weight, dim=-1, dtype=torch.float32).to(x) if self.normalize_weight else self.weight
        return causal_conv1d(
            x, w, self.bias, initial_state, bos_mask,
            output_final_state, self.activation, self.backend, deterministic
        )

    def extra_repr(self) -> str:
        return 'num_features={} ({}), width={}, bias={}, normalize_w={}, act={}, backend={}'.format(
            self.num_features, self.features_per_partition, self.kernel_size,
            self.bias is not None, self.normalize_weight, self.activation, self.backend
        )
