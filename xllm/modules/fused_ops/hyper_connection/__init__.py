"""Triton kernels for hyper-connection streams (xllm/modules/residual/hyper_connection.py).

Streams X [T, N, D] stay in BF16 (or FP16); FP32 lives only in registers. Coefficients are FP32 and
contiguous: a, b [T, N], R [T, N, N] with destination rows. N <= 4: the kernels hold four streams per
program and mask the ones beyond N.

The module path's ops, each an autograd Function over the raw kernels:

    hc_logits(Xf, W, eps)     rstd(Xf) * (Xf @ W)        Xf = X.flatten(-2) [T, N * D], FP32 out
    hc_read(X, a)             h  = sum_n a_n X_n          [T, D]
    hc_write(X, R, b, y)      X' = R X + b y              [T, N, D], R None = identity
    sinkhorn(z, iterations)   mHC transport from logits [..., 4, 4]

The projection is an rstd kernel and a cuBLAS BF16 matmul with FP32 accumulate and output; its backward
forms dX = dP W^T - c X in one kernel. Read and write are one kernel each way, Sinkhorn one each way.
Every op saves only its inputs, except the projection, which also keeps its FP32 rstd [T] and logits
[T, n_logits].

The raw launchers are plain functions on contiguous [T, ...] tensors, so the fused transformer block
(xllm/models/fused_blocks/transformer/transformer.py) calls them from its own forward and backward:

    logits_fwd / logits_bwd                     the projection
    ihc_coef_fwd / ihc_coef_bwd                 the iHC gates from the logits
    ohc_coef_fwd / ohc_coef_bwd                 the oHC gates and SO(4) transport from the logits
    read_fwd / read_bwd / read_da               the read, and its coefficient gradient alone
    write_fwd / write_bwd                       the write; `split` stores X' as hi + lo (see below)

The fused block's stream path keeps the streams between layers as hi + lo pairs (hi = round(X'), the streams
every other kernel reads; lo = round(X' - hi), so hi + lo holds the FP32 write to ~2^-17) and recovers them in
the backward by inverting the writes. Its own passes, one kernel over the N * D channels of every token:

    write_fwd(split, rstd)    the split write and the rstd of its hi, the next controller's normalization
    unwrite_bwd               the inverse write + the recomputed read + the write backward (dy, db, dR) in one pass
    read_proj_bwd             dX = R^T G + a dh + dP W^T - c X: the whole stream gradient of a sublayer in one pass

Modules: `projection` (the controller projection), `streams` (read / write), `fused_block` (the two-pass backward),
`coefficients` (iHC / oHC), `sinkhorn` (mHC), `common` (contracts and shared Triton helpers).
"""

from .common import MAX_STREAMS, supported
from .projection import logits_fwd, proj_bwd_terms, logits_bwd, hc_logits
from .streams import read_fwd, read_bwd, read_da, hc_read, write_fwd, write_bwd, hc_write
from .fused_block import unwrite_bwd, read_proj_bwd
from .coefficients import ihc_coef_fwd, ihc_coef_bwd, ohc_coef_fwd, ohc_coef_bwd
from .sinkhorn import sinkhorn

__all__ = [
    "MAX_STREAMS", "supported",
    "logits_fwd", "proj_bwd_terms", "logits_bwd", "hc_logits",
    "read_fwd", "read_bwd", "read_da", "hc_read", "write_fwd", "write_bwd", "hc_write",
    "unwrite_bwd", "read_proj_bwd",
    "ihc_coef_fwd", "ihc_coef_bwd", "ohc_coef_fwd", "ohc_coef_bwd",
    "sinkhorn",
]
