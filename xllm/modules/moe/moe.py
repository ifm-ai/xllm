from typing import Optional, Tuple

from torch import nn, Tensor
import torch.nn.functional as F

from xllm.modules.model_parallel import (
    RowParallelLinear,
    ColumnParallelLinear,
    scatter_to_model_parallel_region,
)
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.residual import build_residual
from xllm.health_metrics import get_active_health_collector
from xllm.utils import get_init_fn
from .router import TopKRouter
from .expert import Expert
from .layout import ExpertLayout
from .dedup import gather_rows, weighted_sum
from .dispatch import build_dispatcher, dispatch_rows, combine_rows


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

        self.layout = ExpertLayout.from_process_groups(num_experts)
        self.dispatcher = build_dispatcher(self.layout, self.topk, permutation_backend)
        self.n_local_experts = self.layout.n_local
        self.expert_start_idx = self.layout.start
        self.expert_end_idx = self.layout.end

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

        # B x L x K
        routing_scores, routing_indices, tokens_per_expert, aux_loss = self.router(x, load_balancing_type)
        health_collector = get_active_health_collector()
        if health_collector is not None:
            health_collector.record_router_stats(
                "moe", self.layer_id, tokens_per_expert, routing_scores, self.router.bias, aux_loss,
            )
        # B*L x D/TP
        x = x.view(bsz * slen, -1)
        plan = self.dispatcher.plan(routing_indices.flatten(), tokens_per_expert)
        backend = self.permutation_backend

        # routed experts: each token's slice goes once to every rank holding one of its experts
        # B' x D: this rank's tokens, rebuilt
        x = dispatch_rows(x, self.dispatcher, plan)
        # P x D: one row per local (token, expert) pair, in expert order
        x = gather_rows(x, plan.tp.pair_rows, plan.tp.pair_table, backend)
        y2 = self.experts(x, plan.group_sizes)
        # B' x D: experts' outputs scaled by their routing scores and summed per token
        scores = self.dispatcher.local_scores(routing_scores, plan)
        weights = scores.reshape(-1).index_select(0, plan.tp.pair_slots.flatten()).view_as(plan.tp.pair_slots)
        y2 = weighted_sum(y2, plan.tp.pair_table, weights, plan.tp.pair_rows, plan.tp.pair_cols, backend)
        # B*L x D/TP: add up the parts returned by each rank
        y2 = combine_rows(y2, self.dispatcher, plan)
        y2 = y2.view(bsz, slen, -1)

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


def moe_layer_ids(model: nn.Module) -> list[int]:
    """Decoder layers that hold an MoE, read from the built (possibly FSDP-wrapped) model."""
    return sorted(module.layer_id for module in model.modules() if isinstance(module, NormalizedMoE))
