from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn

from xllm.modules import (
    GroupLayerNorm,
    GroupRMSNorm,
    TimestepDecayNorm,
)
from xllm.modules.model_parallel import (
    ColumnParallelLinear,
    vocab_parallel_cross_entropy,
    gather_copy_model_parallel_region,
)
from xllm.modules.context_parallel import (
    should_send_to_next,
    should_recv_from_prev,
    send_to_next_context_parallel_region,
    recv_from_prev_context_parallel_region,
)
from xllm.config import ModelConf
from xllm.health_metrics import get_active_health_collector
from xllm.distributed import get_context_parallel_rank
from xllm.models.fused_blocks import (
    TransformerOutputLayerFunction,
    GekkoOutputLayerFunction
)
from xllm.utils import get_init_fn
from xllm.optim.roles import register_role


class TransformerOutputLayer(nn.Module):
    def __init__(self, cfg: ModelConf):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.output_size = cfg.vocab_size
        self.apply_rmsnorm = cfg.apply_rmsnorm
        self.rmsnorm_eps = cfg.rmsnorm_eps
        self.recompute_logits = cfg.recompute_logits

        norm_cls = GroupRMSNorm if cfg.apply_rmsnorm else GroupLayerNorm
        self.norm_eps = cfg.stream_norm_eps
        if self.norm_eps is None:
            self.norm_eps = cfg.rmsnorm_eps if self.apply_rmsnorm else cfg.layernorm_eps
        self.final_norm = norm_cls(
            cfg.model_dim,
            num_groups=cfg.layernorm_num_groups,
            elementwise_affine=cfg.norm_affine,
            eps=self.norm_eps,
            memory_efficient=cfg.memory_efficient_norm
        )

        init_fn = get_init_fn('gaussian', dim=self.model_dim, std=cfg.init_logits_std)
        self.output = ColumnParallelLinear(
            self.model_dim,
            self.output_size,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=init_fn
        )
        # roles (xllm/optim/roles.py)
        register_role(self.final_norm, "weight", "final_norm")
        register_role(self.final_norm, "bias", "final_norm")
        register_role(self.output, "weight", "readout")

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor],
        mask: Optional[Tensor] = None,
    ):
        if self.training:
            assert y is not None
            fn = TransformerOutputLayerFunction
            return fn.apply(
                x,
                y,
                mask,
                self.final_norm.weight,
                self.final_norm.bias,
                self.final_norm.groups_per_partition,
                self.output.weight,
                self.norm_eps,
                self.rmsnorm_eps,
                self.apply_rmsnorm,
                self.final_norm.gather_input,
                self.recompute_logits,
            )

        x = self.final_norm(x)
        if not self.final_norm.gather_input:
            x = gather_copy_model_parallel_region(x)
        logits = self.output(x).float()
        collector = get_active_health_collector()
        if collector is not None and collector.enable_logits_stats and y is not None:
            # Diagnostics must not change tensors saved for checkpoint recomputation.
            detached_logits = logits.detach()
            valid_tokens = y.ne(-100)
            if mask is not None:
                valid_tokens.logical_and_(mask.to(torch.bool))
            token_sums = detached_logits.sum(dim=-1)
            logits_sum = token_sums.masked_fill(~valid_tokens, 0.0).sum()
            logits_count = valid_tokens.sum() * logits.shape[-1]
            token_logit_minimums, token_logit_maximums = torch.aminmax(detached_logits, dim=-1)
            logits_minimum = token_logit_minimums.masked_fill(
                ~valid_tokens, float("inf")
            ).min()
            logits_maximum = token_logit_maximums.masked_fill(
                ~valid_tokens, -float("inf")
            ).max()
            collector.record_logits_stats(
                logits_sum, logits_count, logits_minimum, logits_maximum,
            )
        if y is None:
            return logits
        else:
            loss = vocab_parallel_cross_entropy(logits, y)
            if mask is not None:
                loss = loss * mask.to(loss)
            return loss

    def num_parameters(self):
        embed_params = self.model_dim * self.output_size * 2
        # final norm
        norm_params = self.model_dim * (1 if self.apply_rmsnorm else 2)

        total_params = embed_params + norm_params
        return total_params, embed_params

    def tflops_per_token(self, seq_len: int):
        embed_flops = self.model_dim + self.output_size
        logits_flops = self.model_dim * self.output_size
        norm_flops = self.model_dim * (2 if self.apply_rmsnorm else 4)

        return embed_flops + logits_flops + norm_flops


class GekkoOutputLayer(nn.Module):
    def __init__(self, cfg: ModelConf):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.output_size = cfg.vocab_size
        self.recompute_logits = cfg.recompute_logits

        self.final_norm = TimestepDecayNorm(
            self.model_dim, cfg.timenorm_num_groups, cfg.timenorm_beta1, cfg.timenorm_beta2,
            eps=cfg.timenorm_eps, backend=cfg.timenorm_backend,
            memory_efficient=cfg.memory_efficient_norm
        )

        init_fn = get_init_fn('gaussian', dim=self.model_dim, std=cfg.init_logits_std)
        self.output = ColumnParallelLinear(self.model_dim, self.output_size, bias=False,
                                           input_is_parallel=True, gather_output=False, init_method=init_fn)
        # roles (xllm/optim/roles.py)
        register_role(self.final_norm, "weight", "final_norm")
        register_role(self.final_norm, "bias", "final_norm")
        register_role(self.output, "weight", "readout")

        self.context_parallel_rank = get_context_parallel_rank()

    def _receive_prev_count(self, x, bos_mask):
        bsz, seq_len, _ = x.size()
        if bos_mask is None:
            prev_count = torch.full((bsz,), seq_len * self.context_parallel_rank, dtype=torch.int64, device=x.device)
        else:
            prev_count = torch.empty(bsz, dtype=torch.int64, device=x.device)
            prev_count = recv_from_prev_context_parallel_region(prev_count)
        return prev_count

    def _receive_prev_tensors(self, x):
        bsz, seq_len, _ = x.size()
        n_groups = self.final_norm.groups_per_partition
        prev_tensor = torch.empty((bsz, n_groups * 2), dtype=torch.float32, device=x.device, requires_grad=self.training)
        prev_tensor = recv_from_prev_context_parallel_region(prev_tensor)
        return prev_tensor

    def _pack_prev_tensors(self, prev_mean, prev_var):
        # B x (G*2)
        prev_tensor = torch.cat([prev_mean, prev_var], dim=-1)
        return prev_tensor

    def _unpack_prev_tensors(self, prev_tensor):
        n_groups = self.final_norm.groups_per_partition
        prev_mean, prev_var = torch.split(prev_tensor, [n_groups, n_groups], dim=-1)
        prev_mean = prev_mean
        prev_var = prev_var
        return prev_mean, prev_var

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor],
        mask: Optional[Tensor] = None,
        bos_mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ):
        bsz, seq_len, _ = x.size()
        if bos_mask is not None:
            bos_mask = bos_mask[:, -seq_len:]

        if self.training:
            assert y is not None
            fn = GekkoOutputLayerFunction
            return fn.apply(
                x,
                y,
                mask,
                bos_mask,
                self.final_norm.weight,
                self.final_norm.bias,
                self.final_norm.prior_count,
                self.final_norm.prior_mean,
                self.final_norm.prior_var,
                self.final_norm.groups_per_partition,
                self.final_norm.beta1,
                self.final_norm.beta2,
                self.final_norm.backend,
                self.output.weight,
                self.final_norm.eps,
                self.recompute_logits,
            )

        if cache is not None:
            prev_count, prev_mean, prev_var = cache
        elif should_recv_from_prev():
            prev_count = self._receive_prev_count(x, bos_mask)
            prev_tensor = self._receive_prev_tensors(x)
            prev_mean, prev_var = self._unpack_prev_tensors(prev_tensor)
        else:
            prev_count, prev_mean, prev_var = None, None, None

        x, prev_count, prev_mean, prev_var = self.final_norm(x, bos_mask, prev_count, prev_mean, prev_var)

        if cache is not None:
            cache = (prev_count.detach(), prev_mean.detach(), prev_var.detach())
        elif should_send_to_next():
            if bos_mask is not None:
                send_to_next_context_parallel_region(prev_count)
            prev_tensor = self._pack_prev_tensors(prev_mean, prev_var)
            prev_tensor = send_to_next_context_parallel_region(prev_tensor)
            x = x + prev_tensor.to(x).mean() * 0

        logits = self.output(x).float()
        if y is None:
            return logits, cache
        else:
            loss = vocab_parallel_cross_entropy(logits, y)
            if mask is not None:
                loss = loss * mask.to(loss)
            return loss, cache

    def num_parameters(self):
        embed_params = self.model_dim * self.output_size * 2
        # final norm
        norm_params = self.model_dim * 2

        total_params = embed_params + norm_params
        return total_params, embed_params

    def tflops_per_token(self, seq_len: int):
        embed_flops = self.model_dim + self.output_size
        logits_flops = self.model_dim * self.output_size
        norm_flops = self.model_dim * 8

        return embed_flops + logits_flops + norm_flops
