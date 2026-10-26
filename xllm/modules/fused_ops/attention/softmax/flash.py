import os
from typing import Optional, Tuple

import torch
from torch.autograd.function import FunctionCtx

FLASH_ATTN_3 = os.getenv("ENABLE_FLASH_ATTENTION_3", "FALSE").lower() == "true"
FLASH_ATTN_4 = os.getenv("ENABLE_FLASH_ATTENTION_4", "FALSE").lower() == "true"
if FLASH_ATTN_4:
    from flash_attn.cute.interface import _flash_attn_fwd as _flash_attn_4_fwd
    from flash_attn.cute.interface import _flash_attn_bwd as _flash_attn_4_bwd
    FLASH_ATTN_4 = True
    FLASH_ATTN_3 = False
elif FLASH_ATTN_3:
    # We need to import the CUDA kernels after importing torch
    import flash_attn_3._C  # Registers operators with PyTorch
    flash_attn_gpu = torch.ops.flash_attn_3
    FLASH_ATTN_4 = False
    FLASH_ATTN_3 = True
else:
    import flash_attn_2_cuda as flash_attn_gpu
    FLASH_ATTN_4 = False
    FLASH_ATTN_3 = False


class FlashAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        scale: float,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        deterministic: bool = True,
        training: bool = True
    ) -> torch.Tensor:
        assert scale is not None
        p = dropout if training else 0.0

        if cu_seqlens_q is None:
            y, softmax_lse, rng_state = _flash_attention_fwd(
                q, k, v, scale, p, use_causal_mask,
            )
        else:
            fq = torch.flatten(q, 0, 1)
            fk = torch.flatten(k, 0, 1)
            fv = torch.flatten(v, 0, 1)
            y, softmax_lse, rng_state = _flash_attention_varlen_fwd(
                fq, fk, fv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, p, use_causal_mask,
            )
            y = y.view(q.shape)

        ctx.save_for_backward(q, k, v, y, softmax_lse, rng_state, cu_seqlens_q, cu_seqlens_k)
        # Non-Tensor attributes
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scale = scale
        ctx.dropout = dropout
        ctx.use_causal_mask = use_causal_mask
        ctx.determinstic = deterministic
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               None, None, None, None, None, None, None, None, None]:

        q, k, v, y, softmax_lse, rng_state, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        scale = ctx.scale
        dropout = ctx.dropout
        use_causal_mask = ctx.use_causal_mask
        determinstic = ctx.determinstic

        if cu_seqlens_q is None:
            q_grad, k_grad, v_grad = _flash_attention_bwd(
                y_grad, y, q, k, v, softmax_lse, scale, dropout, rng_state, use_causal_mask, determinstic,
            )
        else:
            fq = torch.flatten(q, 0, 1)
            fk = torch.flatten(k, 0, 1)
            fv = torch.flatten(v, 0, 1)
            y = torch.flatten(y, 0, 1)
            y_grad = torch.flatten(y_grad, 0, 1)
            q_grad, k_grad, v_grad = _flash_attention_varlen_bwd(
                y_grad, y, fq, fk, fv, softmax_lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                scale, dropout, rng_state, use_causal_mask, determinstic,
            )
            q_grad = q_grad.view(q.shape)
            k_grad = k_grad.view(k.shape)
            v_grad = v_grad.view(v.shape)

        return q_grad, k_grad, v_grad, None, None, None, None, None, None, None, None, None


flash_attention = FlashAttentionFunc.apply


def _flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout: float = 0.0,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.shape[3] % 8 == 0

    if FLASH_ATTN_4:
        out, softmax_lse, *rest = _flash_attn_4_fwd(
            q, k, v, softmax_scale=scale, causal=causal, return_lse=True,
            window_size_left=-1, window_size_right=-1
        )
        rng_state = None
    elif FLASH_ATTN_3:
        out, softmax_lse, *rest = flash_attn_gpu.fwd(
            q, k, v, None, None, None, None, None, None, None,  # k_new, v_new, qv, out, cu_seqlens_q/k/k_new
            None, None, None, None, None, None, None,  # seqused_q/k, max_seqlen_q/k, page_table, kv_batch_idx, leftpad_k,
            None, None, None, None, None, None,  # rotary_cos/sin, seqlens_rotary, q_descale, k_descale, v_descale,
            scale, causal, -1, -1, 0, 0.0,  # softmax_scale, causal, window_size, attn_chunk, softcap,
            True, None, 1, None, 0,  # rotary_interleaved, scheduler_metadata, num_splits, pack_gqa, sm_margin
        )
        rng_state = None
    else:
        out, softmax_lse, _, rng_state = flash_attn_gpu.fwd(
            q, k, v, None, None, dropout, scale, causal, -1, -1, 0.0, False, None
        )

    return out, softmax_lse, rng_state


def _flash_attention_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = 0.0,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.shape[2] % 8 == 0

    if FLASH_ATTN_4:
        out, softmax_lse, *rest = _flash_attn_4_fwd(
            q, k, v, softmax_scale=scale, causal=causal, return_lse=True,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            window_size_left=-1, window_size_right=-1
        )
        rng_state = None
    elif FLASH_ATTN_3:
        out, softmax_lse, *rest = flash_attn_gpu.fwd(
            q, k, v, None, None, None, None,  # k_new, v_new, qv, out,
            cu_seqlens_q, cu_seqlens_k, None, None, None,  # cu_seqlens_k_new, seqused_q/k
            max_seqlen_q, max_seqlen_k, None, None, None,  # page_table, kv_batch_idx, leftpad_k,
            None, None, None, None, None, None,  # rotary_cos/sin, seqlens_rotary, q_descale, k_descale, v_descale,
            scale, causal, -1, -1, 0, 0.0,  # softmax_scale, causal, window_size, attn_chunk, softcap,
            True, None, 1, None, 0,  # rotary_interleaved, scheduler_metadata, num_splits, pack_gqa, sm_margin
        )
        rng_state = None
    else:
        out, softmax_lse, _, rng_state = flash_attn_gpu.varlen_fwd(
            q, k, v, None, cu_seqlens_q, cu_seqlens_k,
            None, None, None, None, max_seqlen_q, max_seqlen_k,
            dropout, scale, False, causal, -1, -1, 0.0, False, None,
        )

    return out, softmax_lse, rng_state


def _flash_attention_bwd(
    out_grad: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_lse: torch.Tensor,
    scale: float,
    dropout: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
    causal: bool = True,
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert out_grad.shape[3] % 8 == 0

    q_grad, k_grad, v_grad = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if FLASH_ATTN_4:
        _flash_attn_4_bwd(
            q, k, v, out, out_grad, softmax_lse, scale, causal,
            dq=q_grad, dk=k_grad, dv=v_grad, deterministic=deterministic,
            window_size_left=-1, window_size_right=-1
        )
    elif FLASH_ATTN_3:
        flash_attn_gpu.bwd(
            out_grad, q, k, v, out, softmax_lse, q_grad, k_grad, v_grad,
            None, None, None, None, None, None,  # cu_seqlens_q/k, seqused_q/k, max_seqlen_q/k
            scale, causal, -1, -1, 0.0, deterministic, 0,
        )
    else:
        flash_attn_gpu.bwd(
            out_grad, q, k, v, out, softmax_lse, q_grad, k_grad, v_grad,
            None, dropout, scale, causal, -1, -1, 0.0, deterministic, None, rng_state,
        )

    return q_grad, k_grad, v_grad


def _flash_attention_varlen_bwd(
    out_grad: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
    causal: bool = True,
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert out_grad.shape[2] % 8 == 0

    q_grad, k_grad, v_grad = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if FLASH_ATTN_4:
        _flash_attn_4_bwd(
            q, k, v, out, out_grad, softmax_lse, scale, causal,
            dq=q_grad, dk=k_grad, dv=v_grad, deterministic=deterministic,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            window_size_left=-1, window_size_right=-1
        )
    elif FLASH_ATTN_3:
        flash_attn_gpu.bwd(
            out_grad, q, k, v, out, softmax_lse, q_grad, k_grad, v_grad,
            cu_seqlens_q, cu_seqlens_k, None, None, max_seqlen_q, max_seqlen_k,  # seqused_q/k
            scale, causal, -1, -1, 0.0, deterministic, 0,
        )
    else:
        flash_attn_gpu.varlen_bwd(
            out_grad, q, k, v, out, softmax_lse, q_grad, k_grad, v_grad,
            cu_seqlens_q, cu_seqlens_k, None, max_seqlen_q, max_seqlen_k,
            dropout, scale, False, causal, -1, -1, 0.0, deterministic, None, rng_state,
        )

    return q_grad, k_grad, v_grad


def flash_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    scale: float,
    dropout: float = 0.0,
    use_causal_mask: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens_q is None:
        y, softmax_lse, rng_state = _flash_attention_fwd(q, k, v, scale, dropout, use_causal_mask)
    else:
        fq = torch.flatten(q, 0, 1)
        fk = torch.flatten(k, 0, 1)
        fv = torch.flatten(v, 0, 1)
        y, softmax_lse, rng_state = _flash_attention_varlen_fwd(
            fq, fk, fv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, dropout, use_causal_mask
        )
        y = y.view(q.shape)

    return y, softmax_lse, rng_state


def flash_attention_bwd(
    out_grad: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    scale: float,
    dropout: float = 0.0,
    rng_state: Optional[torch.Tensor] = None,
    use_causal_mask: bool = True,
    deterministic: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens_q is None:
        q_grad, k_grad, v_grad = _flash_attention_bwd(
            out_grad, out, q, k, v, softmax_lse, scale, dropout, rng_state, use_causal_mask, deterministic,
        )
    else:
        fq = torch.flatten(q, 0, 1)
        fk = torch.flatten(k, 0, 1)
        fv = torch.flatten(v, 0, 1)
        out = torch.flatten(out, 0, 1)
        out_grad = torch.flatten(out_grad, 0, 1)
        q_grad, k_grad, v_grad = _flash_attention_varlen_bwd(
            out_grad, out, fq, fk, fv, softmax_lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            scale, dropout, rng_state, use_causal_mask, deterministic,
        )
        q_grad = q_grad.view(q.shape)
        k_grad = k_grad.view(k.shape)
        v_grad = v_grad.view(v.shape)

    return q_grad, k_grad, v_grad
