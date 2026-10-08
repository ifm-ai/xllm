from typing import Optional, Tuple, Any
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xllm.distributed import (
    get_model_parallel_world_size,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.causal_attention import (
    CausalSoftmaxAttention,
    CausalSoftdeltaAttention
)
from xllm.modules.residual import build_residual
from xllm.modules.model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_copy_model_parallel_region,
)
from xllm.modules.fused_ops import (
    memory_efficient_dropout,
)
from xllm.utils import get_init_fn


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
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
        qknorm: bool,
        norm_num_groups: int,
        norm_affine: bool,
        layernorm_eps: float,
        rmsnorm_eps: float,
        memory_efficient_norm: bool,
        apply_rmsnorm: bool,
        apply_attn_gate: bool,
        attn_gate_func: Optional[str],
        causal_attn_backend: Optional[str],
        dropout: float,
        attention_dropout: float,
        hidden_dropout: float,
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
        self.qknorm = qknorm
        self.attn_act_func = attn_act_func

        assert self.n_heads % self.n_kv_heads == 0
        assert 0 <= self.rope_head_dim <= self.head_dim

        # Divide the weight matrix along the last dimension.
        model_parallel_world_size = get_model_parallel_world_size()
        self.local_heads = divide_and_check_no_remainder(self.n_heads, model_parallel_world_size)
        self.local_kv_heads = divide_and_check_no_remainder(self.n_kv_heads, model_parallel_world_size)

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.causal_attn_backend = causal_attn_backend
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

        self.wv = ColumnParallelLinear(
            mdim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
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
            memory_efficient=False,
        ) if self.qknorm else None

        self.key_norm = GroupRMSNorm(
            self.n_kv_heads * self.head_dim,
            num_groups=self.n_kv_heads,
            elementwise_affine=False,
            eps=rmsnorm_eps,
            memory_efficient=True,  # for knorm, always use efficient memory
        ) if self.qknorm else None

        if self.attn_act_func == "softmax":
            causal_attention_cls = CausalSoftmaxAttention
        elif self.attn_act_func == "softdelta":
            assert not self.norm.gather_input
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
        stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Any]]:
        residual = x
        x = self.norm(x)
        if not self.norm.gather_input:
            x = gather_copy_model_parallel_region(x)

        # B x L x H
        g = torch.sigmoid(self.wg(x)) if self.wg is not None else None
        # B x L x D
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        if self.qknorm:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)
        # B x L x (H*S)
        attn, new_cache = self.causal_attention(
            xq, xk, xv, g, freqs_cis, segments, stability_control, deterministic, cache
        )
        if self.apply_attn_gate:
            if self.attn_gate_fn == "silu":
                r = F.silu(self.wr(x))
            elif self.attn_gate_fn == "softplus":
                r = F.softplus(self.wr(x), beta=math.log(2))
            else:
                raise ValueError(f"Unknown attention gate function {self.attn_gate_fn}")
            attn = attn * r

        attn = memory_efficient_dropout(attn, self.hidden_dropout, self.training)
        h = self.wo(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)
        # residual
        out = self.residual(h, residual, c=attn)

        return out, new_cache

    def extra_repr(self) -> str:
        return 'dim={}, heads={} ({}), head_dim={} ({}), act={}, backend={}, gate={}, init={} ({})'.format(
            self.mdim, self.n_heads, self.n_kv_heads, self.head_dim, self.rope_head_dim,
            self.attn_act_func, self.causal_attn_backend,
            self.attn_gate_fn if self.apply_attn_gate else None,
            self.init_mode, self.init_std
        )
