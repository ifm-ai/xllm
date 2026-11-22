from typing import Optional, Tuple, Any
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from xllm.health_metrics import get_active_health_collector
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
from xllm.modules.context_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_context_parallel_region,
    recv_from_prev_context_parallel_region,
)
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.causal_conv import CausalConv1d
from xllm.modules.residual import build_residual
from xllm.modules.multihead_attention import (
    CausalSoftmaxAttention,
    CausalSoftdeltaAttention
)
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.optim.muon_spec import register_muon
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
        apply_nope: bool,
        attn_act_func: str,
        num_values: int,
        num_activated_values: int,
        qknorm: bool,
        apply_causal_conv: bool,
        causal_conv_width: int,
        causal_conv_backend: str,
        causal_conv_weight_normalization,
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
        permutation_backend: str,
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
        self.nope = apply_nope
        assert self.n_heads % self.n_kv_heads == 0
        assert 0 <= self.rope_head_dim <= self.head_dim

        self.n_values = num_values
        self.topk = num_activated_values
        self.qknorm = qknorm
        self.attn_act_func = attn_act_func
        self.causal_conv = apply_causal_conv

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

        self.wq = ColumnParallelLinear(
            mdim,
            self.n_heads * self.head_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )

        # key & value projections
        nk = {"softmax": 1, "softdelta": 2}[attn_act_func]
        self.wk = ColumnParallelLinear(
            mdim,
            self.n_kv_heads * self.head_dim * nk,
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

        # qkv shortconvs
        self.q_conv = CausalConv1d(
            self.n_heads * self.head_dim,
            causal_conv_width,
            bias=False,
            apply_weight_normalization=causal_conv_weight_normalization,
            activation=None,
            backend=causal_conv_backend,
        ) if apply_causal_conv else None

        self.k_conv = CausalConv1d(
            self.n_kv_heads * self.head_dim * nk,
            causal_conv_width,
            bias=False,
            apply_weight_normalization=causal_conv_weight_normalization,
            activation=None,
            backend=causal_conv_backend,
        ) if apply_causal_conv else None

        self.v_conv = CausalConv1d(
            self.n_kv_heads * self.head_dim,
            causal_conv_width,
            bias=False,
            apply_weight_normalization=causal_conv_weight_normalization,
            activation='silu',
            backend=causal_conv_backend,
        ) if apply_causal_conv else None

        # qk-norm
        self.query_norm = GroupRMSNorm(
            self.n_heads * self.head_dim,
            num_groups=self.n_heads,
            elementwise_affine=True,
            eps=rmsnorm_eps,
            memory_efficient=False,
        ) if self.qknorm else None

        self.key_norm = GroupRMSNorm(
            self.n_kv_heads * self.head_dim * nk,
            num_groups=self.n_kv_heads * nk,
            elementwise_affine=True,
            eps=rmsnorm_eps,
            memory_efficient=False,
        ) if self.qknorm else None

        # attention gate projection
        rdim = {"softmax": 0, "softdelta": self.n_heads}[attn_act_func]
        if self.apply_attn_gate:
            rdim += self.n_heads * self.head_dim
        self.wr = ColumnParallelLinear(
            mdim,
            rdim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        ) if rdim > 0 else None

        if self.attn_act_func == "softmax":
            causal_attention_cls = CausalSoftmaxAttention
        elif self.attn_act_func == "softdelta":
            causal_attention_cls = CausalSoftdeltaAttention
        else:
            raise ValueError(f"Unknown attention activation function: {self.attn_act_func}")

        self.causal_attention = causal_attention_cls(
            layer_id,
            self.local_heads,
            self.local_kv_heads,
            self.head_dim,
            self.rope_head_dim,
            self.nope,
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

        # Muon (xllm/optim/muon.py): one Newton-Schulz block per head (each of a kv head's two softdelta
        # keys is its own block; wr holds a head's gate rows and, for softdelta, its forget logit), one
        # per value expert and kv head in wv; wo is whole; the router stays with AdamW
        for w in (self.wq, self.wk):
            register_muon(w, layout="(h d) m", fan_out="d", fan_in="m", d=self.head_dim, tp_dim=0)
        register_muon(self.wv, layout="(n h d) m", fan_out="d", fan_in="m",
                      h=self.n_kv_heads, d=self.head_dim, tp_dim=1)
        if self.wr is not None:
            register_muon(self.wr, layout="(h d) m", fan_out="d", fan_in="m", d=rdim // self.n_heads, tp_dim=0)
        register_muon(self.wo, tp_dim=1)

    def _receive_prev_tensors(self, x):
        bsz, seq_len, _ = x.size()
        nk = {"softmax": 1, "softdelta": 2}[self.attn_act_func]
        width = self.q_conv.kernel_size - 1
        cdim = self.local_heads * self.head_dim + self.local_kv_heads * (self.head_dim * nk + self.head_dim)
        prev_tensor = torch.empty(
            (bsz, width * cdim), dtype=x.dtype, device=x.device, requires_grad=self.training
        )
        prev_tensor = recv_from_prev_context_parallel_region(prev_tensor)
        return prev_tensor

    def _pack_prev_tensors(self, conv_state_q, conv_state_k, conv_state_v):
        # B x (W-1) x D-> B x (W-1)*D
        conv_q = conv_state_q.flatten(1)
        conv_k = conv_state_k.flatten(1)
        conv_v = conv_state_v.flatten(1)
        prev_tensor = torch.cat([conv_q, conv_k, conv_v], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, x, prev_tensor):
        bsz = x.size(0)
        nk = {"softmax": 1, "softdelta": 2}[self.attn_act_func]
        width = self.q_conv.kernel_size - 1
        n_heads = self.local_heads
        n_kv_heads = self.local_kv_heads
        kdim = self.head_dim
        vdim = self.head_dim
        conv_q_size = width * n_heads * kdim
        conv_k_size = width * n_kv_heads * kdim * nk
        conv_v_size = width * n_kv_heads * vdim
        conv_q, conv_k, conv_v = torch.split(prev_tensor, [conv_q_size, conv_k_size, conv_v_size], dim=-1)
        conv_q = conv_q.view(bsz, width, n_heads * kdim)
        conv_k = conv_k.view(bsz, width, n_kv_heads * kdim * nk)
        conv_v = conv_v.view(bsz, width, n_kv_heads * vdim)
        return conv_q, conv_k, conv_v

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segments: Optional[Any] = None,
        stability_control: int = 0,
        deterministic: bool = True,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
        load_balancing_type: Optional[str] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Any]]:
        bsz, slen, _ = x.shape
        residual = x

        x = self.norm(x)
        sx = scatter_to_model_parallel_region(x) if self.norm.gather_input else x
        # B x L x K
        routing_scores, routing_indices, tokens_per_expert, aux_loss = self.router(sx, load_balancing_type)
        health_collector = get_active_health_collector()
        if health_collector is not None:
            health_collector.record_router_stats(
                "mova", self.layer_id, tokens_per_expert, routing_scores, self.router.bias, aux_loss,
            )

        if cache is not None:
            cache_attn, cache_conv = cache
            conv_state_q, conv_state_k, conv_state_v = cache_conv
        elif should_recv_from_prev() and self.causal_conv:
            prev_tensor = self._receive_prev_tensors(x)
            conv_state_q, conv_state_k, conv_state_v = self._unpack_prev_tensors(x, prev_tensor)
            cache_attn, cache_conv = None, None
        else:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            cache_attn, cache_conv = None, None

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

        # gather when grouped norm layer
        mx = gather_copy_model_parallel_region(x) if not self.norm.gather_input else x
        # QK
        xq = self.wq(mx)
        xk = self.wk(mx)

        # causal convs on QKV
        if self.causal_conv:
            if bos_mask is not None:
                curr = bos_mask.shape[1] - slen
                bos_mask_curr = bos_mask[:, curr:]
            else:
                bos_mask_curr = None

            output_final_state = cache_conv is not None or should_send_to_next()
            # B x L x Q/TP
            xq, conv_state_q = self.q_conv(
                xq, conv_state_q, bos_mask_curr, output_final_state, deterministic
            )
            # B x L x K/TP
            xk, conv_state_k = self.k_conv(
                xk, conv_state_k, bos_mask_curr, output_final_state, deterministic
            )
            # B x L x V/TP
            xv, conv_state_v = self.v_conv(
                xv, conv_state_v, bos_mask_curr, output_final_state, deterministic
            )

            if cache is not None:
                cache_conv = (conv_state_q.detach(), conv_state_k.detach(), conv_state_v.detach())
            elif should_send_to_next():
                prev_tensor = self._pack_prev_tensors(conv_state_q, conv_state_k, conv_state_v)
                prev_tensor = send_to_next_context_parallel_region(prev_tensor)
                xv = xv + prev_tensor.to(xv).mean() * 0

        if self.qknorm:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        if self.wr is not None:
            rg = rearrange(self.wr(mx), 'b l (h s) -> b l h s', h=self.local_heads)
            # B x L x H*S
            r = rearrange(rg[:, :, :, :self.head_dim], 'b l h s -> b l (h s)') if self.apply_attn_gate else None
            # B x L x H
            g = rg[:, :, :, -1] if self.attn_act_func == 'softdelta' else None
        else:
            r = g = None

        # B x L x (H*S)
        attn, cache_attn = self.causal_attention(
            xq, xk, xv, g, None if self.nope else freqs_cis,
            bos_mask, segments, stability_control, deterministic, cache_attn
        )
        if self.apply_attn_gate:
            if self.attn_gate_fn == "silu":
                r = F.silu(r)
            elif self.attn_gate_fn == "softplus":
                r = F.softplus(r, beta=math.log(2))
            else:
                raise ValueError(f"Unknown attention gate function {self.attn_gate_fn}")
            attn = attn * r

        attn = memory_efficient_dropout(attn, self.hidden_dropout, self.training)
        h = self.wo(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)
        # residual
        out = self.residual(h, residual, c=attn)

        if cache is not None:
            cache = (cache_attn, cache_conv)

        return out, aux_loss, cache

    def extra_repr(self) -> str:
        return 'dim={}, heads={} ({}), head_dim={} ({}), act={}, values={} ({}), backends=({}, {}, {}), gate={}, init={} ({})'.format(
            self.mdim, self.n_heads, self.n_kv_heads, self.head_dim, 'nope' if self.nope else self.rope_head_dim,
            self.attn_act_func, self.n_values, self.topk, self.causal_attn_backend, self.value_backend,
            self.permutation_backend, self.attn_gate_fn if self.apply_attn_gate else None,
            self.init_mode, self.init_std
        )


def mova_layer_ids(model: nn.Module) -> list[int]:
    """Decoder layers that hold MoVA attention, read from the built (possibly FSDP-wrapped) model."""
    return sorted(module.layer_id for module in model.modules() if isinstance(module, MOVAttention))
