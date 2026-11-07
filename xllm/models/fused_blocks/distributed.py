from typing import Optional, List
import torch

from xllm.distributed import (
    get_parallel_region,
    get_context_parallel_rank,
)
from xllm.distributed.utils import (
    split_tensor_along_specific_dim,
    reshape_scattered_tensor_along_specific_dim,
    reshape_gathered_tensor_along_specific_dim
)
from xllm.modules.context_parallel.mappings import (
    _receive_tensor_from_prev,
    _send_tensor_to_next,
    _receive_grad_from_next,
    _send_grad_to_prev
)

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def recv_prev_count(x, bos_mask):
    bsz, seq_len, _ = x.size()
    if bos_mask is None:
        context_rank = get_context_parallel_rank()
        prev_count = torch.full((bsz,), seq_len * context_rank, dtype=torch.int64, device=x.device)
        handle = None
    else:
        prev_count = torch.empty(bsz, dtype=torch.int64, device=x.device)
        prev_count, handle = _receive_tensor_from_prev(None, prev_count, async_op=True)
    return prev_count, handle


def recv_prev_mean_var(x, n_groups):
    bsz, seq_len, _ = x.size()
    prev_tensor = torch.empty((bsz, n_groups, 2), dtype=x.dtype, device=x.device)
    prev_tensor, handle = _receive_tensor_from_prev(None, prev_tensor, async_op=True)
    return prev_tensor, handle


def send_mean_var_grad_to_prev(prev_mean_grad, prev_var_grad):
    # B x (G*2+D*N*2)
    prev_tensor = torch.stack([prev_mean_grad, prev_var_grad], dim=2)
    return _send_grad_to_prev(prev_tensor, async_op=True)


def send_count_to_next(prev_count, bos_mask):
    if bos_mask is None:
        return prev_count, None
    return _send_tensor_to_next(prev_count, async_op=True)


def send_mean_var_to_next(prev_mean, prev_var):
    # B x (G*2+D*N*2)
    prev_tensor = torch.stack([prev_mean, prev_var], dim=2)
    return _send_tensor_to_next(prev_tensor, async_op=True)


def recv_mean_var_grad_from_next(x, n_groups):
    bsz, seq_len, _ = x.size()
    prev_tensor = torch.empty((bsz, n_groups, 2), dtype=x.dtype, device=x.device)
    prev_tensor, handle = _receive_grad_from_next(None, prev_tensor, async_op=True)
    return prev_tensor, handle


def recv_prev_hx(x, ndim):
    bsz, _, mdim = x.size()
    hx = torch.empty((bsz, mdim, ndim, 2), dtype=torch.float32, device=x.device)
    hx, handle = _receive_tensor_from_prev(None, hx, async_op=True)
    return hx, handle


def send_hx_grad_to_prev(hx_grad):
    hx_grad = _c2r(hx_grad)
    return _send_grad_to_prev(hx_grad, async_op=True)


def send_h_to_next(h):
    h = _c2r(h)
    return _send_tensor_to_next(h, async_op=True)


def recv_h_grad_from_next(x, ndim):
    bsz, _, mdim = x.size()
    h_grad = torch.empty((bsz, mdim, ndim, 2), dtype=torch.float32, device=x.device)
    h_grad, handle = _receive_grad_from_next(None, h_grad, async_op=True)
    return h_grad, handle


def recv_prev_conv_state(x, width, dim):
    bsz = x.shape[0]
    state = torch.empty((bsz, width - 1, dim), dtype=x.dtype, device=x.device)
    state, handle = _receive_tensor_from_prev(None, state, async_op=True)
    return state, handle


def send_prev_conv_state_grad_to_prev(conv_state_grad):
    return _send_grad_to_prev(conv_state_grad, async_op=True)


def send_prev_conv_state_to_next(conv_state):
    return _send_tensor_to_next(conv_state, async_op=True)


def recv_prev_conv_state_grad_from_next(x, width, dim):
    bsz = x.shape[0]
    state_grad = torch.empty((bsz, width - 1, dim), dtype=x.dtype, device=x.device)
    state_grad, handle = _receive_grad_from_next(None, state_grad, async_op=True)
    return state_grad, handle


def recv_prev_key_or_value(x, chunk_size, n_heads, head_dim):
    bsz = x.shape[0]
    prev_kv = torch.empty((bsz, chunk_size, n_heads, head_dim), dtype=x.dtype, device=x.device)
    prev_kv, handle = _receive_tensor_from_prev(None, prev_kv, async_op=True)
    return prev_kv, handle


def send_prev_key_or_value_grad_to_prev(prev_grad):
    return _send_grad_to_prev(prev_grad, async_op=True)


def send_prev_key_or_value_to_next(prev_kv):
    return _send_tensor_to_next(prev_kv, async_op=True)


def recv_prev_key_or_value_grad_from_next(x, chunk_size, n_heads, head_dim):
    bsz = x.shape[0]
    prev_grad = torch.empty((bsz, chunk_size, n_heads, head_dim), dtype=x.dtype, device=x.device)
    prev_grad, handle = _receive_grad_from_next(None, prev_grad, async_op=True)
    return prev_grad, handle


def recv_prev_memory(x, n_heads, qk_head_dim, v_head_dim):
    bsz = x.shape[0]
    memory = torch.empty(bsz, n_heads, qk_head_dim, v_head_dim, dtype=x.dtype, device=x.device)
    memory, handle = _receive_tensor_from_prev(None, memory, async_op=True)
    return memory, handle


def recv_prev_log_norm_term(x, n_heads, qk_head_dim):
    bsz = x.shape[0]
    log_norm_term = torch.empty(bsz, n_heads, qk_head_dim, dtype=torch.float32, device=x.device)
    log_norm_term, handle = _receive_tensor_from_prev(None, log_norm_term, async_op=True)
    return log_norm_term, handle


def send_memory_grad_to_prev(memory_grad):
    return _send_grad_to_prev(memory_grad, async_op=True)


def send_log_norm_term_grad_to_prev(log_norm_term_grad):
    return _send_grad_to_prev(log_norm_term_grad, async_op=True)


def send_memory_to_next(memory):
    return _send_tensor_to_next(memory, async_op=True)


def send_log_norm_term_to_next(log_norm_term):
    return _send_tensor_to_next(log_norm_term, async_op=True)


def recv_memory_grad_from_next(x, n_heads, qk_head_dim, v_head_dim):
    bsz = x.shape[0]
    memory_grad = torch.empty(bsz, n_heads, qk_head_dim, v_head_dim, dtype=x.dtype, device=x.device)
    memory_grad, handle = _receive_grad_from_next(None, memory_grad, async_op=True)
    return memory_grad, handle


def recv_log_norm_term_grad_from_next(x, n_heads, qk_head_dim):
    bsz = x.shape[0]
    log_norm_term_grad = torch.empty(bsz, n_heads, qk_head_dim, dtype=torch.float32, device=x.device)
    log_norm_term_grad, handle = _receive_grad_from_next(None, log_norm_term_grad, async_op=True)
    return log_norm_term_grad, handle


def all_gather(x: torch.Tensor, parallel_region: str, dim: Optional[int] = None, async_op: bool = False):
    _, world_size, group = get_parallel_region(parallel_region)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return x, None

    output = torch.empty(world_size, *x.shape, dtype=x.dtype, device=x.device)
    handle = torch.distributed.all_gather_into_tensor(output, x, group=group, async_op=async_op)
    if not async_op:
        gather_dim = x.dim() - 1 if dim is None else dim
        output = reshape_gathered_tensor_along_specific_dim(output, gather_dim)

    return output, handle


def all_reduce(x: torch.Tensor, parallel_region: str, async_op: bool = False):
    _, world_size, group = get_parallel_region(parallel_region)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return x, None

    # All-reduce.
    handle = torch.distributed.all_reduce(x, group=group, async_op=async_op)

    return x, handle


def reduce_scatter(x: torch.Tensor, parallel_region: str, dim: Optional[int] = None, async_op: bool = False):
    rank, world_size, group = get_parallel_region(parallel_region)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return x, None

    # reduce-scatter.
    scatter_dim = x.dim() - 1 if dim is None else dim
    x_ = reshape_scattered_tensor_along_specific_dim(x, world_size, scatter_dim)
    output = torch.empty(*x_.shape[1:], dtype=x_.dtype, device=x_.device)
    handle = torch.distributed.reduce_scatter_tensor(output, x_, group=group, async_op=async_op)

    return output, handle


def scatter(x: torch.Tensor, parallel_region: str, dim: Optional[int] = None):
    rank, world_size, group = get_parallel_region(parallel_region)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return x

    # Split along the specific dimension.
    split_dim = x.dim() - 1 if dim is None else dim
    output = split_tensor_along_specific_dim(x, world_size, split_dim=split_dim)[rank]

    return output


def all_to_all(
    x: torch.Tensor,
    input_split_sizes: Optional[List[int]],
    output_split_sizes: Optional[List[int]],
    parallel_region: str,
    async_op: bool = False
):
    rank, world_size, group = get_parallel_region(parallel_region)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return x, None

    assert (input_split_sizes is None) == (output_split_sizes is None)
    out_shape = x.shape if output_split_sizes is None else [sum(output_split_sizes)] + list(x.shape[1:])
    output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    handle = torch.distributed.all_to_all_single(
        output, x, output_split_sizes, input_split_sizes, group=group, async_op=async_op
    )

    return output, handle
