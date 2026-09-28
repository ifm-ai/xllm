#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Any
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from .rms_norm import GroupRMSNorm
from .timestep_decay_norm import TimestepDecayNorm
from .complex_exponential_moving_average import MultiHeadComplexEMA
from .sliding_chunk_attention import SlidingChunkAttention
from .adaptive_working_memory import AdaptiveWorkingMemory
from .residual import (
    BaseResidual,
)
from xllm.distributed import (
    get_model_parallel_world_size,
    get_context_parallel_rank,
)
from xllm.distributed.utils import divide_and_check_no_remainder
from .model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    gather_copy_model_parallel_region,
)
from .context_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_context_parallel_region,
    recv_from_prev_context_parallel_region,
)
from .fused_ops import (
    memory_efficient_dropout,
)
from xllm.utils import get_init_fn

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.
    See "https://arxiv.org/abs/2601.06463" for more details.
    """

    def __init__(
        self,
        layer_id: int,
        mdim: int,
        num_heads: int,
        head_dim: int,
        v_head_dim: int,
        rope_head_dim: Optional[int],
        ndim: int,
        gate_channels_per_head: int,
        attn_gate_func: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = 2048,
        cema_backend: str = 'cub',
        efficient_attn: Optional[str] = None,
        timenorm_num_groups: Optional[int] = None,
        timenorm_beta1: float = 0.999,
        timenorm_beta2: float = 0.9999,
        timenorm_backend: str = 'cub',
        rmsnorm_num_groups: int = 1,
        norm_affine: bool = True,
        timenorm_eps: float = 1e-5,
        rmsnorm_eps: float = 1e-6,
        apply_bias_term: bool = False,
        memory_efficient_norm: bool = False,
        init_mode: str = 'he',
        init_std: Optional[float] = None,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.mdim = mdim
        self.ndim = ndim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.head_channels = gate_channels_per_head

        assert v_head_dim % gate_channels_per_head == 0
        zdim = head_dim * num_heads
        hdim = v_head_dim * num_heads
        gdim = gate_channels_per_head * num_heads

        self.rope_head_dim = self.head_dim if rope_head_dim is None else rope_head_dim
        assert 0 <= self.rope_head_dim <= self.head_dim

        self.init_mode = init_mode
        self.init_std = init_std

        # Divide the weight matrix along the last dimension.
        model_parallel_world_size = get_model_parallel_world_size()
        self.local_heads = divide_and_check_no_remainder(num_heads, model_parallel_world_size)
        self.local_mdim = divide_and_check_no_remainder(mdim, model_parallel_world_size)

        self.chunk_size = chunk_size
        self.efficient_attn = efficient_attn
        self.attn_gate_fn = attn_gate_func
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self.timenorm = TimestepDecayNorm(
            mdim, timenorm_num_groups, timenorm_beta1, timenorm_beta2,
            eps=timenorm_eps, backend=timenorm_backend,
            memory_efficient=memory_efficient_norm
        )
        self.cema = MultiHeadComplexEMA(mdim, ndim, cema_backend)
        self.rmsnorm = GroupRMSNorm(
            mdim,
            num_groups=rmsnorm_num_groups,
            elementwise_affine=norm_affine,
            eps=rmsnorm_eps,
            memory_efficient=False,
        )
        self.znorm = GroupRMSNorm(
            zdim,
            num_groups=self.num_heads,
            elementwise_affine=False,
            eps=rmsnorm_eps,
            memory_efficient=True,  # for znorm, always use efficient memory
        )

        init_fn = get_init_fn(init_mode, dim=mdim, std=init_std)
        self.wv = ColumnParallelLinear(
            mdim,
            hdim,
            bias=True,
            input_is_parallel=True,
            gather_output=False,
            init_method=init_fn
        )
        self.wz = ColumnParallelLinear(
            mdim,
            zdim,
            bias=True,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
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
        self.wg = ColumnParallelLinear(
            mdim,
            gdim,
            bias=apply_bias_term,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )
        self.wh1 = ColumnParallelLinear(
            mdim,
            mdim,
            bias=apply_bias_term,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )
        # wh2 projection
        self.wh2 = RowParallelLinear(
            hdim,
            mdim,
            bias=False,
            input_is_parallel=True,
            parallel_output=True,
            init_method=get_init_fn(init_mode, dim=hdim, std=init_std)
        )
        self.sliding_chunk_attention = SlidingChunkAttention(
            self.local_heads,
            self.local_heads,
            self.head_dim,
            self.v_head_dim,
            self.rope_head_dim,
            self.chunk_size,
            1.0,
            self.attention_dropout,
            efficient_attn
        )
        self.adaptive_working_memory = AdaptiveWorkingMemory(
            self.local_heads,
            self.local_heads,
            self.head_dim,
            self.v_head_dim,
            self.chunk_size,
            False,
            rmsnorm_eps,
            False
        )
        self.gamma = Parameter(torch.zeros(4 * self.head_dim * self.local_heads))
        self.beta = Parameter(torch.zeros(4 * self.head_dim * self.local_heads))
        self.residual = BaseResidual(2 * layer_id + 1, mdim)

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
        ndim = self.cema.ndim
        dim = self.local_mdim
        csize = self.chunk_size
        cdim = self.local_heads * (self.head_dim * 3 + self.v_head_dim)
        adim = self.local_heads * (self.head_dim * self.v_head_dim + self.head_dim)
        prev_tensor = torch.empty(
            (bsz, n_groups * 2 + dim * ndim * 2 + csize * cdim + adim),
            dtype=torch.float32, device=x.device, requires_grad=self.training
        )
        prev_tensor = recv_from_prev_context_parallel_region(prev_tensor)
        return prev_tensor

    def _pack_prev_tensors(self, prev_mean, prev_var, hx, prev_sk, prev_aqk, prev_akk, prev_v, memory, log_norm_term):
        # B x D x N x 2 -> B x (D*N*2)
        h = _c2r(hx).flatten(1)
        k1 = prev_sk.flatten(1)
        k2 = prev_aqk.flatten(1)
        k3 = prev_akk.flatten(1)
        v = prev_v.flatten(1)
        m = memory.flatten(1)
        l = log_norm_term.flatten(1)
        prev_tensor = torch.cat([prev_mean, prev_var, h, k1, k2, k3, v, m, l], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, x, prev_tensor):
        bsz = x.size(0)
        n_groups = self.timenorm.groups_per_partition
        ndim = self.cema.ndim
        dim = self.local_mdim
        csize = self.chunk_size
        n_heads = self.local_heads
        zdim = self.head_dim
        vdim = self.v_head_dim
        zsize = csize * n_heads * zdim
        vsize = csize * n_heads * vdim
        msize = n_heads * zdim * vdim
        lntsize = n_heads * zdim
        prev_mean, prev_var, hx, prev_sk, prev_aqk, prev_akk, prev_v, mem, lnt = torch.split(
            prev_tensor, [n_groups, n_groups, 2 * dim * ndim, zsize, zsize, zsize, vsize, msize, lntsize], dim=-1
        )
        prev_mean = prev_mean.to(x)
        prev_var = prev_var.to(x)
        hx = _r2c(hx.view(bsz, dim, ndim, 2))
        prev_sk = prev_sk.view(bsz, csize, n_heads, zdim).to(x)
        prev_aqk = prev_aqk.view(bsz, n_heads, csize, zdim).to(x)
        prev_akk = prev_akk.view(bsz, n_heads, csize, zdim).to(x)
        prev_v = prev_v.view(bsz, csize, n_heads, vdim).to(x)
        mem = mem.view(bsz, n_heads, zdim, vdim).to(x)
        lnt = lnt.view(bsz, n_heads, zdim)
        return prev_mean, prev_var, hx, prev_sk, prev_aqk, prev_akk, prev_v, mem, lnt

    def forward(
        self,
        x: Tensor,
        freqs_cis: Optional[Tensor],
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        prev_segment_count: Optional[Tensor] = None,
        deterministic: bool = True,
        cache: Optional[Tuple[Tuple[Tensor, Tensor, int],
                              Tuple[Tensor, Tensor, Tensor, Tensor],
                              Tuple[Tensor, Tensor, Tensor],
                              Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Any]]:
        bsz, seq_len, _ = x.size()
        residual = x

        if cache is not None:
            cache_sca, cache_awk, cache_norm, hx = cache
            prev_count, prev_mean, prev_var = cache_norm
            prev_sk, prev_v, kv_count = cache_sca
            prev_aqk, prev_akk, memory, log_norm_term = cache_awk
        elif should_recv_from_prev():
            prev_count = self._receive_prev_count(x, bos_mask)
            prev_tensor = self._receive_prev_tensors(x)
            (prev_mean, prev_var, hx, prev_sk,
             prev_aqk, prev_akk, prev_v, memory, log_norm_term) = self._unpack_prev_tensors(x, prev_tensor)
            kv_count = None
            cache_sca, cache_awk, cache_norm = None, None, None
        else:
            prev_count, prev_mean, prev_var = None, None, None
            prev_sk, prev_v, kv_count = None, None, None
            prev_aqk, prev_akk, memory, log_norm_term = None, None, None, None
            cache_sca, cache_awk, cache_norm, hx = None, None, None, None

        if prev_v is not None and bos_mask is not None:
            bos_mask_curr = bos_mask[:, self.chunk_size:]
        else:
            bos_mask_curr = bos_mask

        # B x L x D
        out_tsn, prev_count, prev_mean, prev_var = self.timenorm(x, bos_mask_curr, prev_count, prev_mean, prev_var)
        # B x D x L
        out_cema, hx = self.cema(out_tsn.transpose(1, 2), hx, bos_mask_curr)

        # B x D x L -> B x L x D
        mx = self.rmsnorm(out_cema.transpose(1, 2))
        mx = memory_efficient_dropout(mx, self.hidden_dropout, self.training)
        mx = gather_copy_model_parallel_region(mx)

        beta = self.beta.view(4, -1)
        gamma = self.gamma.view(4, -1)
        gamma = (gamma + 1.0) / math.sqrt(self.head_dim)
        # B x L x S
        z = self.wz(mx)
        # B x L x H x S/H -> B x L x S
        z = self.znorm(z)
        # B x L x S -> B x L x 1 x S -> B x L x 4 x S
        z = z.unsqueeze(2) * gamma + beta
        # B x L x 4 x S -> B x L x S
        q, k, aq, ak = torch.unbind(z, dim=2)

        # B x L x E
        v = F.silu(self.wv(out_tsn))
        if self.attn_gate_fn == "silu":
            r = F.silu(self.wr(mx))
        elif self.attn_gate_fn == "softplus":
            r = F.softplus(self.wr(mx), beta=math.log(2))
        else:
            raise ValueError(f"Unknown attention gate function {self.attn_gate_fn}")
        # B x L x (H*C)
        g = self.wg(mx)

        prev_sv = prev_v
        prev_av = prev_v.transpose(1, 2) if prev_v is not None else None
        # B x L x E
        sca, prev_sk, prev_v = self.sliding_chunk_attention(
            q, k, v, freqs_cis, prev_sk, prev_sv, bos_mask, segment_idx, deterministic
        )
        awk, awk_mask, memory, log_norm_term, prev_aqk, prev_akk, _ = self.adaptive_working_memory(
            aq, ak, v, memory, log_norm_term, prev_aqk, prev_akk, prev_av, segment_idx, prev_segment_count
        )
        # B x L x H*C x V/C
        sca = sca.view(bsz, seq_len, self.local_heads * self.head_channels, -1)
        awk = awk.view(bsz, seq_len, self.local_heads * self.head_channels, -1)

        # B x L x H*C
        g = torch.sigmoid(torch.masked_fill(g, awk_mask.unsqueeze(2), value=float("-inf")))
        # B x L x E
        attn = torch.addcmul(sca, (awk - sca), g.unsqueeze(3)).view(bsz, seq_len, -1) * r

        if cache is not None:
            cache_norm = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
            hx = None if hx is None else hx.detach()
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
                prev_mean, prev_var, hx, prev_sk, prev_aqk, prev_akk, prev_v, memory, log_norm_term
            )
            prev_tensor = send_to_next_context_parallel_region(prev_tensor)
            attn = attn + prev_tensor.to(attn).mean() * 0

        # B x L x E -> B x L x D
        h = self.wh1(mx) + self.wh2(attn)
        h = memory_efficient_dropout(h, self.dropout, self.training)
        # residual
        out = self.residual(h, residual)

        if cache is not None:
            cache = (cache_sca, cache_awk, cache_norm, hx)

        return out, cache

    def extra_repr(self) -> str:
        return 'dim={}, heads={}, head_dim={} ({}), v_head_dim={} ({}), chunk={}, eff_attn={}, gate={}, init={} ({})'.format(
            self.mdim, self.num_heads, self.head_dim, self.rope_head_dim, self.v_head_dim, self.head_channels,
            self.chunk_size, self.efficient_attn, self.attn_gate_fn, self.init_mode, self.init_std
        )
