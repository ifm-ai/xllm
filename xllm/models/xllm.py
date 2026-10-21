import abc
from typing import Union, Dict, Tuple, List
import types
import warnings
import math

import torch
from torch import nn
from torch.nn.utils import get_total_norm, clip_grads_with_norm_
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xllm.config import ModelConf
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.distributed import (
    get_context_parallel_world_size,
    get_data_parallel_world_size,
    get_model_parallel_group,
    get_model_parallel_world_size,
)


class XLLModel(nn.Module):
    def __init__(self, cfg: ModelConf, tokenizer: Tokenizer):
        super().__init__()

        self.arch = cfg.arch
        self.num_layers = cfg.num_layers
        assert cfg.vocab_size > 0
        self.vocab_size = cfg.vocab_size
        self.output_size = self.vocab_size

        self.model_dim = cfg.model_dim

        # heads & head dims
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_heads if cfg.num_kv_heads is None else cfg.num_kv_heads
        self.head_dim = cfg.model_dim // cfg.num_heads if cfg.head_dim is None else cfg.head_dim
        self.v_head_dim = self.head_dim if cfg.v_head_dim is None else cfg.v_head_dim
        self.rope_head_dim = self.head_dim if cfg.rope_head_dim is None else cfg.rope_head_dim
        assert 0 <= self.rope_head_dim <= self.head_dim
        self.attn_act_func = cfg.attn_act_func
        self.apply_attn_gate = cfg.apply_attn_gate
        self.causal_attn_backend = cfg.causal_attn_backend

        self.residual_func = cfg.residual_func
        self.residual_heads = cfg.residual_heads

        # hidden dim
        self.ffn_hidden_dim = cfg.ffn_hidden_dim
        self.swiglu = cfg.swiglu
        # MoE & MoVA
        self.num_values = cfg.num_values
        self.num_activated_values = cfg.num_activated_values
        self.num_experts = cfg.num_experts
        self.num_activated_experts = cfg.num_activated_experts
        self.num_shared_experts = cfg.num_shared_experts
        if self.num_experts == 0:
            assert cfg.num_dense_layers is None
            self.num_dense_layers = self.num_layers
        else:
            self.num_dense_layers = 0 if cfg.num_dense_layers is None else cfg.num_dense_layers
        self.expert_inter_dim = cfg.expert_inter_dim

        self.apply_rmsnorm = cfg.apply_rmsnorm

        self.dropout = cfg.dropout
        self.layerwise_ckpt = cfg.layerwise_ckpt
        self.tokenizer = tokenizer

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_id

    @torch.no_grad()
    def clip_grad_norm_(
            self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
    ) -> float:
        """
        Clip all gradients at this point in time. The norm is computed over all
        gradients together, as if they were concatenated into a single vector.
        Gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """

        parameters = self.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        else:
            is_generator = isinstance(parameters, types.GeneratorType)
            # prevent generators from being exhausted
            parameters = list(parameters)
            if is_generator and len(parameters) == 0:
                warnings.warn(
                    "`parameters` is an empty generator, no gradient clipping will occur.",
                    stacklevel=3,
                )

        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return 0.

        total_norm = get_total_norm(grads, norm_type, False, None)
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        model_parallel_world_size = get_model_parallel_world_size()
        # Reconstruct the total gradient norm depending on the norm type
        if model_parallel_world_size > 1:
            model_parallel_process_group = get_model_parallel_group()
            if norm_type == math.inf:
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=model_parallel_process_group)
            else:
                total_norm = total_norm ** norm_type
                dist.all_reduce(total_norm, group=model_parallel_process_group)
                total_norm = total_norm ** (1.0 / norm_type)

        clip_grads_with_norm_(parameters, max_norm, total_norm, None)
        total_norm_cpu = total_norm.item()
        return total_norm_cpu

    @abc.abstractmethod
    def support_multi_segments_with_cache(self) -> bool:
        raise NotImplemented

    @abc.abstractmethod
    def num_parameters(self):
        raise NotImplemented

    @abc.abstractmethod
    def tflops_per_token(self, seq_len: int):
        raise NotImplemented

    def extra_repr(self) -> str:
        repr = (
            f"world_size=({get_data_parallel_world_size()}, {get_context_parallel_world_size()}, {get_model_parallel_world_size()})"
        )
        return repr
