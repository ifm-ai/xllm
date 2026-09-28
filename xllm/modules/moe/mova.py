from typing import Optional, Tuple, Any
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from xllm.utils import get_init_fn
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.distributed import (
    get_model_parallel_world_size,
)
from xllm.modules.model_parallel import (
    RowParallelLinear,
    ColumnParallelLinear,
    GroupRowParallelLinear,
    scatter_to_model_parallel_region,
    gather_copy_model_parallel_region
)
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.residual import build_residual
from xllm.modules.multihead_attention import (
    CausalSoftmaxAttention,
    CausalSoftdeltaAttention
)
from xllm.modules.fused_ops import memory_efficient_dropout
from .router import TopKRouter
from .permute.permute_ops import fused_permute_y


class MOVAttention(nn.Module):
    """
    Mixture-of-value attention layer.
    """
    def __init__(
        self,
        layer_id: int,
        mdim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        head_dim: Optional[int],
        rope_head_dim: Optional[int],
        attn_act_func: str,
        num_values: int,
        num_activated_values: int,
        qknorm: bool,
        norm_num_groups,
        norm_affine,
        layernorm_eps: float,
        rmsnorm_eps: float,
        memory_efficient_norm: bool,
        apply_rmsnorm: bool,
        apply_attn_gate: bool,
        attn_gate_func: Optional[str],
        causal_attn_backend: Optional[str],
        value_backend: str,
        permutation_backend,
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
        router_score_func: str = 'sigmoid',
        router_bias: bool = False,
        router_bias_update_rate: Optional[float] = None,
        router_scaling_factor: Optional[float] = None,
        residual_func: str = 'base',
        residual_heads: Optional[int] = None,
        init_mode: str = 'he',
        init_std: Optional[float] = None,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.mdim = mdim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = mdim // n_heads if head_dim is None else head_dim
        self.rope_head_dim = self.head_dim if rope_head_dim is None else rope_head_dim
        assert self.n_heads % self.n_kv_heads == 0
        assert 0 <= self.rope_head_dim <= self.head_dim

        self.n_values = num_values
        self.topk = num_activated_values
        self.qknorm = qknorm
        self.attn_act_func = attn_act_func

        # Divide the weight matrix along the last dimension.
        model_parallel_world_size = get_model_parallel_world_size()
        self.local_heads = divide_and_check_no_remainder(self.n_heads, model_parallel_world_size)
        self.local_kv_heads = divide_and_check_no_remainder(self.n_kv_heads, model_parallel_world_size)

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.causal_attn_backend = causal_attn_backend
        self.value_backend = value_backend
        self.permutation_backend = permutation_backend
        self.apply_attn_gate = apply_attn_gate
        self.attn_gate_fn = attn_gate_func
        self.init_mode = init_mode
        self.init_std = init_std

        norm_cls = GroupRMSNorm if apply_rmsnorm else GroupLayerNorm
        norm_eps = rmsnorm_eps if apply_rmsnorm else layernorm_eps
        self.norm = norm_cls(
            mdim,
            num_groups=norm_num_groups,
            elementwise_affine=norm_affine,
            eps=norm_eps,
            memory_efficient=memory_efficient_norm
        )

        init_fn = get_init_fn(init_mode, dim=mdim, std=init_std)
        # router
        self.router = TopKRouter(
            mdim,
            self.n_values,
            self.topk,
            bias=router_bias,
            bias_update_rate=router_bias_update_rate,
            score_func=router_score_func,
            scaling_factor=router_scaling_factor,
            init_method=init_fn
        )

        nq = {"softmax": 1, "softdelta": 2}[attn_act_func]
        assert nq is not None
        self.wq = ColumnParallelLinear(
            mdim,
            self.n_heads * self.head_dim * nq,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )

        self.wk = ColumnParallelLinear(
            mdim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )

        self.wv = GroupRowParallelLinear(
            mdim,
            self.n_kv_heads * self.head_dim,
            self.n_values,
            backend=self.value_backend,
            input_is_parallel=True,
            parallel_output=True,
            init_method=init_fn
        )

        self.wr = ColumnParallelLinear(
            mdim,
            self.n_heads * self.head_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        ) if self.apply_attn_gate else None

        self.wg = RowParallelLinear(
            mdim,
            self.n_heads,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=init_fn
        ) if self.attn_act_func == "softdelta" else None

        self.query_norm = GroupRMSNorm(
            self.n_heads * self.head_dim * nq,
            num_groups=self.n_heads * nq,
            elementwise_affine=True,
            eps=rmsnorm_eps,
            memory_efficient=True,  # for qknorm, always use efficient memory
        ) if self.qknorm else None

        self.key_norm = GroupRMSNorm(
            self.n_kv_heads * self.head_dim,
            num_groups=self.n_kv_heads,
            elementwise_affine=True,
            eps=rmsnorm_eps,
            memory_efficient=True,  # for qknorm, always use efficient memory
        ) if self.qknorm else None

        if self.attn_act_func == "softmax":
            causal_attention_cls = CausalSoftmaxAttention
        elif self.attn_act_func == "softdelta":
            causal_attention_cls = CausalSoftdeltaAttention
        else:
            raise ValueError(f"Unknown attention activation function: {self.attn_act_func}")

        self.causal_attention = causal_attention_cls(
            self.local_heads,
            self.local_kv_heads,
            self.head_dim,
            self.rope_head_dim,
            None,
            self.attention_dropout,
            self.causal_attn_backend,
        )

        self.wo = RowParallelLinear(
            self.n_heads * self.head_dim,
            mdim,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=init_fn,
        )

        self.residual = build_residual(
            residual_func, 2 * layer_id + 1, mdim, num_heads=residual_heads,
            num_features=self.n_heads * self.head_dim, eps=rmsnorm_eps, init_std=init_std
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        segments: Optional[Any] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor]] = None,
        load_balancing_type: Optional[str] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        bsz, slen, _ = x.shape
        residual = x
        x = self.norm(x)
        sx = scatter_to_model_parallel_region(x) if self.norm.gather_input else x
        # B x L x K
        routing_scores, routing_indices, tokens_per_expert, aux_loss = self.router(sx, load_balancing_type)
        # B*L*K
        sorted_indices = torch.argsort(routing_indices.flatten(), stable=True)
        permute_indices = sorted_indices // self.topk
        # N
        group_sizes = tokens_per_expert.tolist()
        # B*L*K x D/TP
        permuted_sx = torch.index_select(sx.view(bsz * slen, -1), 0, permute_indices)
        # B*L*K x V/TP
        xv = F.silu(self.wv(permuted_sx, group_sizes))
        # permute xv
        xv = fused_permute_y(sorted_indices, xv, routing_scores, bsz, slen, self.topk, self.permutation_backend)

        # B x L x H
        g = torch.sigmoid(self.wg(sx)) if self.wg is not None else None

        # gather when grouped norm layer
        mx = gather_copy_model_parallel_region(x) if not self.norm.gather_input else x
        # QK
        xq = self.wq(mx)
        xk = self.wk(mx)
        if self.qknorm:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # B x L x (H*S)
        attn, new_cache = self.causal_attention(
            xq, xk, xv, g, freqs_cis, segments, fp32_attn_output, deterministic, cache
        )
        if self.apply_attn_gate:
            if self.attn_gate_fn == "silu":
                r = F.silu(self.wr(mx))
            elif self.attn_gate_fn == "softplus":
                r = F.softplus(self.wr(mx), beta=math.log(2))
            else:
                raise ValueError(f"Unknown attention gate function {self.attn_gate_fn}")
            attn = attn * r

        attn = memory_efficient_dropout(attn, self.hidden_dropout, self.training)
        h = self.wo(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)
        # residual
        out = self.residual(h, residual, c=attn)

        return out, aux_loss, new_cache

    def extra_repr(self) -> str:
        return 'dim={}, heads={} ({}), head_dim={} ({}), act={}, values={} ({}), backends=({}, {}, {}), gate={}, init={} ({})'.format(
            self.mdim, self.n_heads, self.n_kv_heads, self.head_dim, self.rope_head_dim,
            self.attn_act_func, self.n_values, self.topk, self.causal_attn_backend, self.value_backend,
            self.permutation_backend, self.attn_gate_fn if self.apply_attn_gate else None,
            self.init_mode, self.init_std
        )
