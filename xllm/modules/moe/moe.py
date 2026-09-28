from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from xllm.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from xllm.modules.model_parallel import (
    RowParallelLinear,
    ColumnParallelLinear,
    scatter_to_model_parallel_region,
)
from xllm.modules.moe import all_to_all_expert_parallel_region
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.residual import build_residual
from xllm.utils import get_init_fn
from .router import TopKRouter
from .expert import Expert
from .permute import fused_permute_y


class NormalizedMoE(nn.Module):
    def __init__(
        self,
        layer_id: int,
        model_dim: int,
        expert_inter_dim: int,
        num_experts: int,
        num_activated_experts: int,
        num_shared_experts: int,
        expert_backend: str = 'sequential',
        permutation_backend: str = 'torch',
        router_score_func: str = 'sigmoid',
        router_bias: bool = False,
        router_bias_update_rate: Optional[float] = None,
        router_scaling_factor: Optional[float] = None,
        dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        norm_num_groups: int = 1,
        norm_affine: bool = True,
        layernorm_eps: float = 1e-5,
        rmsnorm_eps: float = 1e-6,
        memory_efficient_norm: bool = False,
        apply_rmsnorm: bool = False,
        residual_func: str = 'base',
        residual_heads: Optional[int] = None,
        init_mode: str = 'gaussian',
        init_std: Optional[float] = None
    ):
        super().__init__()
        self.layer_id = layer_id

        self.model_dim = model_dim
        self.inter_dim = expert_inter_dim
        self.n_experts = num_experts
        self.topk = num_activated_experts
        self.n_shared_experts = num_shared_experts
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout
        self.permutation_backend = permutation_backend
        self.init_mode = init_mode
        self.init_std = init_std

        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        self.n_local_experts = divide_and_check_no_remainder(num_experts, world_size)
        self.expert_start_idx = rank * self.n_local_experts
        self.expert_end_idx = (rank + 1) * self.n_local_experts

        norm_cls = GroupRMSNorm if apply_rmsnorm else GroupLayerNorm
        norm_eps = rmsnorm_eps if apply_rmsnorm else layernorm_eps
        self.norm = norm_cls(
            model_dim,
            num_groups=norm_num_groups,
            elementwise_affine=norm_affine,
            eps=norm_eps,
            memory_efficient=memory_efficient_norm,
            disable_input_reduce=True
        )
        init_fn = get_init_fn(init_mode, dim=model_dim, std=init_std)
        # router
        self.router = TopKRouter(
            model_dim,
            self.n_experts,
            self.topk,
            bias=router_bias,
            bias_update_rate=router_bias_update_rate,
            score_func=router_score_func,
            scaling_factor=router_scaling_factor,
            init_method=init_fn
        )
        # routed experts
        self.experts = Expert(
            model_dim,
            expert_inter_dim,
            num_experts,
            backend=expert_backend,
            hidden_dropout=hidden_dropout,
            init_mode=init_mode,
            init_std=init_std
        )

        # shared experts
        ffn_hidden_dim = self.inter_dim * self.n_shared_experts
        self.fc1 = RowParallelLinear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            input_is_parallel=True,
            parallel_output=False,
            init_method=init_fn
        ) if ffn_hidden_dim > 0 else None

        self.fc2 = ColumnParallelLinear(
            ffn_hidden_dim,
            model_dim,
            bias=False,
            input_is_parallel=False,
            gather_output=False,
            init_method=get_init_fn(init_mode, dim=ffn_hidden_dim, std=init_std),
        ) if ffn_hidden_dim > 0 else None

        self.fc3 = RowParallelLinear(
            model_dim,
            ffn_hidden_dim,
            bias=False,
            input_is_parallel=True,
            parallel_output=False,
            init_method=init_fn
        ) if ffn_hidden_dim > 0 else None

        self.residual = build_residual(
            residual_func, 2 * layer_id + 2, model_dim, num_heads=residual_heads,
            num_features=model_dim, eps=rmsnorm_eps, init_std=init_std
        )

    def forward(
        self,
        x: Tensor,
        load_balancing_type: Optional[str] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, slen, _ = x.shape
        # B x L x D/TP
        residual = x
        x = self.norm(x)
        if self.norm.gather_input:
            x = scatter_to_model_parallel_region(x)

        # shared experts
        if self.fc1 is not None:
            # B x L x H*s
            hidden = F.silu(self.fc1(x)) * self.fc3(x)
            # B x L x D/TP
            y1 = self.fc2(memory_efficient_dropout(hidden, self.hidden_dropout, self.training))
        else:
            y1 = None

        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        # B x L x K
        routing_scores, routing_indices, tokens_per_expert, aux_loss = self.router(x, load_balancing_type)
        # B*L x D/TP
        x = x.view(bsz * slen, -1)
        # B*L*K
        routing_indices = routing_indices.flatten()
        # B*L*K
        sorted_indices = torch.argsort(routing_indices, stable=True)
        permute_indices = sorted_indices // self.topk
        # N
        group_sizes = tokens_per_expert[self.expert_start_idx:self.expert_end_idx].tolist()
        input_splits = tokens_per_expert.view(-1, self.n_local_experts).sum(dim=1).tolist()
        curr_bsz = input_splits[rank]
        output_splits = [curr_bsz for _ in range(world_size)]

        # routed experts
        # B*L*K x D/TP
        permuted_x = torch.index_select(x, 0, permute_indices)
        x = all_to_all_expert_parallel_region(permuted_x, input_splits, output_splits)
        # B' x D
        if curr_bsz > 0:
            x = rearrange(x, '(w b) d -> b (w d)', w=world_size)
            y2 = self.experts(x, group_sizes)
            y2 = rearrange(y2, 'b (w d) -> (w b) d', w=world_size)
        else:
            y2 = x
        y2 = all_to_all_expert_parallel_region(y2, output_splits, input_splits)
        # permute y2
        y2 = fused_permute_y(sorted_indices, y2, routing_scores, bsz, slen, self.topk, self.permutation_backend)

        # combine shared expert
        y = y1 + y2 if y1 is not None else y2
        y = memory_efficient_dropout(y, self.dropout, self.training)
        # residual
        out = self.residual(y, residual, c=y)
        return out, aux_loss

    def extra_repr(self) -> str:
        return 'mdim={}, edim={}, experts={} ({}, {}), init={} ({})'.format(
            self.model_dim, self.inter_dim, self.n_experts, self.topk,
            self.n_shared_experts, self.init_mode, self.init_std
        )
