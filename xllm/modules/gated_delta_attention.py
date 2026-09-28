from typing import Optional, Tuple, Any
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xllm.distributed import (
    get_model_parallel_world_size,
    get_context_parallel_rank,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.modules.timestep_decay_norm import TimestepDecayNorm
from xllm.modules.causal_conv import CausalConv1d
from xllm.modules.sliding_chunk_attention import SlidingChunkAttention
from xllm.modules.adaptive_working_memory import AdaptiveWorkingMemory
from xllm.modules.residual import build_residual
from xllm.modules.model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_copy_model_parallel_region,
)
from xllm.modules.context_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_context_parallel_region,
    recv_from_prev_context_parallel_region,
)
from xllm.modules.fused_ops import (
    memory_efficient_dropout,
)
from xllm.utils import get_init_fn


class GatedDeltaAttention(nn.Module):
    """Gated Delta Attention.
    See "" for more details.
    """

    def __init__(
        self,
        layer_id: int,
        mdim: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        head_dim: Optional[int],
        v_head_dim: Optional[int],
        rope_head_dim: Optional[int],
        chunk_size: int,
        attn_act_func: str,
        attn_gate_func: str,
        awm_orthogonal_update: bool,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        causal_conv_width: int = 4,
        causal_conv_backend: str = 'triton',
        causal_conv_weight_normalization: bool = True,
        sca_backend: str = 'swift',
        timenorm_num_groups: Optional[int] = None,
        timenorm_beta1: float = 0.999,
        timenorm_beta2: float = 0.9999,
        timenorm_backend: str = 'cub',
        timenorm_eps: float = 1e-5,
        rmsnorm_eps: float = 1e-6,
        apply_bias_term: bool = False,
        memory_efficient_norm: bool = False,
        residual_func: str = 'base',
        residual_heads: Optional[int] = None,
        init_mode: str = 'he',
        init_std: Optional[float] = None,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.mdim = mdim
        self.num_heads = n_heads
        self.num_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.head_dim = mdim // n_heads if head_dim is None else head_dim
        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        self.rope_head_dim = self.head_dim if rope_head_dim is None else rope_head_dim
        self.attn_act_func = attn_act_func

        assert self.num_heads % self.num_kv_heads == 0
        assert 0 <= self.rope_head_dim <= self.head_dim

        qdim = self.head_dim * self.num_heads
        kdim = self.head_dim * self.num_kv_heads
        vdim = self.v_head_dim * self.num_kv_heads
        hdim = self.v_head_dim * self.num_heads

        self.init_mode = init_mode
        self.init_std = init_std

        # Divide the weight matrix along the last dimension.
        model_parallel_world_size = get_model_parallel_world_size()
        self.local_heads = divide_and_check_no_remainder(n_heads, model_parallel_world_size)
        self.local_kv_heads = divide_and_check_no_remainder(self.num_kv_heads, model_parallel_world_size)

        self.chunk_size = chunk_size
        self.timenorm_backend = timenorm_backend
        self.causal_conv_backend = causal_conv_backend
        self.sca_backend = sca_backend
        self.attn_gate_fn = attn_gate_func
        self.awm_orthogonal_update = awm_orthogonal_update
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self.timenorm = TimestepDecayNorm(
            mdim, timenorm_num_groups, timenorm_beta1, timenorm_beta2,
            eps=timenorm_eps, backend=timenorm_backend,
            memory_efficient=memory_efficient_norm
        )

        init_fn = get_init_fn(init_mode, dim=mdim, std=init_std)
        nq = {"softmax": 1, "softdelta": 2}[attn_act_func]
        assert nq is not None
        self.wq = ColumnParallelLinear(
            mdim,
            qdim * nq,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )
        self.wk = ColumnParallelLinear(
            mdim,
            kdim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )
        self.wv = ColumnParallelLinear(
            mdim,
            vdim,
            bias=True,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )

        self.q_conv = CausalConv1d(
            qdim * nq,
            causal_conv_width,
            bias=False,
            apply_weight_normalization=causal_conv_weight_normalization,
            activation=None,
            backend=causal_conv_backend,
        )
        self.k_conv = CausalConv1d(
            kdim,
            causal_conv_width,
            bias=False,
            apply_weight_normalization=causal_conv_weight_normalization,
            activation=None,
            backend=causal_conv_backend,
        )
        self.v_conv = CausalConv1d(
            vdim,
            causal_conv_width,
            bias=False,
            apply_weight_normalization=causal_conv_weight_normalization,
            activation='silu',
            backend=causal_conv_backend,
        )

        self.query_norm = GroupRMSNorm(
            qdim * nq,
            num_groups=self.num_heads,
            elementwise_affine=True,
            eps=rmsnorm_eps,
            memory_efficient=False,
        )
        self.key_norm = GroupRMSNorm(
            kdim,
            num_groups=self.num_kv_heads,
            elementwise_affine=True,
            eps=rmsnorm_eps,
            memory_efficient=False,
        )

        self.wr = ColumnParallelLinear(
            mdim,
            hdim,
            bias=apply_bias_term,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )

        self.wg = RowParallelLinear(
            mdim,
            self.num_heads,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=init_fn
        ) if self.attn_act_func == "softdelta" else None

        self.sliding_chunk_attention = SlidingChunkAttention(
            self.local_heads,
            self.local_kv_heads,
            self.head_dim,
            self.v_head_dim,
            self.rope_head_dim,
            self.chunk_size,
            None,
            self.attention_dropout,
            sca_backend
        )
        self.adaptive_working_memory = AdaptiveWorkingMemory(
            self.local_heads,
            self.local_kv_heads,
            self.head_dim,
            self.v_head_dim,
            self.chunk_size,
            awm_orthogonal_update,
            rmsnorm_eps,
        )

        self.attn_norm = GroupRMSNorm(
            hdim,
            num_groups=self.num_heads,
            elementwise_affine=False,
            eps=rmsnorm_eps,
            memory_efficient=True,
        )

        # output projection
        self.wo = RowParallelLinear(
            hdim,
            mdim,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=get_init_fn(init_mode, dim=hdim, std=init_std)
        )

        self.residual = build_residual(
            residual_func, 2 * layer_id + 1, mdim, num_heads=residual_heads,
            num_features=hdim, eps=rmsnorm_eps, init_std=init_std
        )

    def _receive_prev_count(self, x, bos_mask):
        bsz, seq_len, _ = x.size()
        context_rank = get_context_parallel_rank()
        if bos_mask is None:
            prev_count = torch.full((bsz,), seq_len * context_rank, dtype=torch.int64, device=x.device)
        else:
            prev_count = torch.empty(bsz, dtype=torch.int64, device=x.device)
            prev_count = recv_from_prev_context_parallel_region(prev_count)
        return prev_count

    def _receive_prev_tensors(self, x):
        bsz, seq_len, _ = x.size()
        n_groups = self.timenorm.groups_per_partition
        csize = self.chunk_size
        width = self.q_conv.kernel_size - 1
        cdim = self.local_heads * self.head_dim + self.local_kv_heads * (self.head_dim + self.v_head_dim)
        sdim = self.local_kv_heads * (self.head_dim * 3 + self.v_head_dim)
        adim = self.local_kv_heads * (self.head_dim * self.v_head_dim + self.head_dim)
        prev_tensor = torch.empty(
            (bsz, n_groups * 2 + width * cdim + csize * sdim + adim),
            dtype=torch.float32, device=x.device, requires_grad=self.training
        )
        prev_tensor = recv_from_prev_context_parallel_region(prev_tensor)
        return prev_tensor

    def _pack_prev_tensors(
        self, prev_mean, prev_var, conv_state_q, conv_state_k, conv_state_v,
        prev_sk, prev_aqk, prev_akk, prev_v, memory, log_norm_term
    ):
        # B x (W-1) x D-> B x (W-1)*D
        conv_q = conv_state_q.flatten(1)
        conv_k = conv_state_k.flatten(1)
        conv_v = conv_state_v.flatten(1)
        k1 = prev_sk.flatten(1)
        k2 = prev_aqk.flatten(1)
        k3 = prev_akk.flatten(1)
        v = prev_v.flatten(1)
        m = memory.flatten(1)
        l = log_norm_term.flatten(1)
        prev_tensor = torch.cat([prev_mean, prev_var, conv_q, conv_k, conv_v, k1, k2, k3, v, m, l], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, x, prev_tensor):
        bsz = x.size(0)
        n_groups = self.timenorm.groups_per_partition
        csize = self.chunk_size
        width = self.q_conv.kernel_size - 1
        n_heads = self.local_heads
        n_kv_heads = self.local_kv_heads
        kdim = self.head_dim
        vdim = self.v_head_dim
        conv_q_size = width * n_heads * kdim
        conv_k_size = width * n_kv_heads * kdim
        conv_v_size = width * n_kv_heads * vdim
        ksize = csize * n_kv_heads * kdim
        vsize = csize * n_kv_heads * vdim
        msize = n_kv_heads * kdim * vdim
        lntsize = n_kv_heads * kdim
        prev_mean, prev_var, conv_q, conv_k, conv_v, prev_sk, prev_aqk, prev_akk, prev_v, mem, lnt = torch.split(
            prev_tensor,
            [n_groups, n_groups, conv_q_size, conv_k_size, conv_v_size, ksize, ksize, ksize, vsize, msize, lntsize],
            dim=-1
        )
        prev_mean = prev_mean.to(x)
        prev_var = prev_var.to(x)
        conv_q = conv_q.view(bsz, width, n_heads * kdim).to(x)
        conv_k = conv_k.view(bsz, width, n_kv_heads * kdim).to(x)
        conv_v = conv_v.view(bsz, width, n_kv_heads * vdim).to(x)
        prev_sk = prev_sk.view(bsz, csize, n_kv_heads, kdim).to(x)
        prev_aqk = prev_aqk.view(bsz, n_kv_heads, csize, kdim).to(x)
        prev_akk = prev_akk.view(bsz, n_kv_heads, csize, kdim).to(x)
        prev_v = prev_v.view(bsz, csize, n_kv_heads, vdim).to(x)
        mem = mem.view(bsz, n_kv_heads, kdim, vdim).to(x)
        lnt = lnt.view(bsz, n_kv_heads, kdim)
        return prev_mean, prev_var, conv_q, conv_k, conv_v, prev_sk, prev_aqk, prev_akk, prev_v, mem, lnt

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
        fp32_attn_output: bool = False,
        deterministic: bool = True,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int],
                              Tuple[Tensor, Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Any]]:
        bsz, seq_len, _ = x.size()
        residual = x

        if cache is not None:
            cache_sca, cache_awk, cache_norm, cache_conv = cache
            prev_count, prev_mean, prev_var = cache_norm
            conv_state_q, conv_state_k, conv_state_v = cache_conv
            prev_sk, prev_v, kv_count = cache_sca
            prev_aqk, prev_akk, memory, log_norm_term = cache_awk
        elif should_recv_from_prev():
            prev_count = self._receive_prev_count(x, bos_mask)
            prev_tensor = self._receive_prev_tensors(x)
            (prev_mean, prev_var, conv_state_q, conv_state_k, conv_state_v,
             prev_sk, prev_aqk, prev_akk, prev_v, memory, log_norm_term) = self._unpack_prev_tensors(x, prev_tensor)
            kv_count = None
            cache_sca, cache_awk, cache_norm, cache_conv = None, None, None, None
        else:
            prev_count, prev_mean, prev_var = None, None, None
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            prev_sk, prev_v, kv_count = None, None, None
            prev_aqk, prev_akk, memory, log_norm_term = None, None, None, None
            cache_sca, cache_awk, cache_norm, cache_conv = None, None, None, None

        if prev_v is not None and bos_mask is not None:
            bos_mask_curr = bos_mask[:, self.chunk_size:]
        else:
            bos_mask_curr = bos_mask

        # B x L x D/TP
        x, prev_count, prev_mean, prev_var = self.timenorm(x, bos_mask_curr, prev_count, prev_mean, prev_var)
        # B x L x H
        g = torch.sigmoid(self.wg(x)) if self.wg is not None else None
        # B x L x D
        mx = gather_copy_model_parallel_region(x)

        output_final_state = cache_conv is not None or should_send_to_next()
        # B x L x Q/TP
        xq, conv_state_q = self.q_conv(
            self.wq(mx), conv_state_q, bos_mask_curr, output_final_state, deterministic
        )
        # B x L x K/TP
        xk, conv_state_k = self.k_conv(
            self.wk(mx), conv_state_k, bos_mask_curr, output_final_state, deterministic
        )
        # B x L x V/TP
        v, conv_state_v = self.v_conv(
            self.wv(mx), conv_state_v, bos_mask_curr, output_final_state, deterministic
        )

        # qk-norm
        sq = self.query_norm(xq)
        sk = self.key_norm(xk)

        # B x L x E/TP
        if self.attn_gate_fn == "silu":
            r = F.silu(self.wr(mx))
        elif self.attn_gate_fn == "softplus":
            r = F.softplus(self.wr(mx), beta=math.log(2))
        else:
            raise ValueError(f"Unknown attention gate function {self.attn_gate_fn}")

        prev_sv = prev_v
        prev_av = prev_v.transpose(1, 2) if prev_v is not None else None
        # B x L x E/TP
        sca, prev_sk, prev_v = self.sliding_chunk_attention(
            sq, sk, v, freqs_cis, prev_sk, prev_sv,
            bos_mask, segment_idx, fp32_attn_output, deterministic
        )
        awk, awk_mask, memory, log_norm_term, prev_aqk, prev_akk, _ = self.adaptive_working_memory(
            xq, xk, xk, v, memory, log_norm_term, prev_aqk, prev_akk, prev_av,
            segment_idx, prev_segment_count
        )
        # B x L x E/TP
        attn = self.attn_norm(sca + awk) * r
        attn = memory_efficient_dropout(attn, self.hidden_dropout, self.training)

        if cache is not None:
            cache_norm = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
            cache_conv = (conv_state_q.detach(), conv_state_k.detach(), conv_state_v.detach())
            prev_sk = prev_sk.detach()
            prev_v = prev_v.detach()
            prev_aqk = prev_aqk.detach()
            prev_akk = prev_akk.detach()
            memory = memory.detach() if memory is not None else memory
            log_norm_term = log_norm_term.detach() if log_norm_term is not None else log_norm_term
            kv_count = kv_count + seq_len
            if kv_count % self.chunk_size == 0:
                prev_sk = prev_sk[:, -self.chunk_size:]
                prev_v = prev_v[:, -self.chunk_size:]
                prev_aqk = prev_aqk[:, :, -self.chunk_size:]
                prev_akk = prev_akk[:, :, -self.chunk_size:]
            cache_sca = (prev_sk, prev_v, kv_count)
            cache_awk = (prev_aqk, prev_akk, memory, log_norm_term)
        elif should_send_to_next():
            if bos_mask is not None:
                send_to_next_context_parallel_region(prev_count)
            prev_tensor = self._pack_prev_tensors(
                prev_mean, prev_var, conv_state_q, conv_state_k, conv_state_v, prev_sk, prev_aqk, prev_akk, prev_v, memory, log_norm_term
            )
            prev_tensor = send_to_next_context_parallel_region(prev_tensor)
            attn = attn + prev_tensor.to(attn).mean() * 0

        # B x L x D/TP
        h = self.wo(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)
        # residual
        out = self.residual(h, residual, c=attn)

        if cache is not None:
            cache = (cache_sca, cache_awk, cache_norm, cache_conv)

        return out, cache

    def extra_repr(self) -> str:
        return 'dim={}, heads={} ({}), head_dim={} ({}), v_head_dim={}, chunk={}, backends=({}, {}, {}), gate={}, init={} ({})'.format(
            self.mdim, self.num_heads, self.num_kv_heads, self.head_dim, self.rope_head_dim, self.v_head_dim, self.chunk_size,
            self.timenorm_backend, self.causal_conv_backend, self.sca_backend, self.attn_gate_fn, self.init_mode, self.init_std
        )
