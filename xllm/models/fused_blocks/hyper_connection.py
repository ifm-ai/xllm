"""The hyper-connection stream path of the fused training blocks (`TransformerBlockFunction`, `GekkoBlockFunction`).

With hyper-connections (`residual_func` ihc / ohc) a layer's state is N = 4 streams X [B, L, N, D] and the two residual
adds become the stream ops of xllm/modules/fused_ops/hyper_connection/, with the sublayers' residual modules as the
controllers (xllm/modules/residual/hyper_connection.py is the module path):

    (a1, R1, b1) = controller1(X);   x  = read(X, a1);   y1 = branch(norm(x));   X1 = write(X, R1, b1, y1)
    (a2, R2, b2) = controller2(X1);  xo = read(X1, a2);  y2 = ffn(norm(xo));     X2 = write(X1, R2, b2, y2)

`FusedStreams` runs these around a block's branches: `read1`, `write1`, `write2` in the forward and `saved` for what
the block saves; `restore`, `after_ffn`, `after_branch` in the backward. The stream tensors live on the object between
the calls, and the block keeps the object on its ctx (it holds only the top layer's output streams across the step).

The streams are not saved per layer; the block saves the branch outputs y1, y2 and the controllers' logits and rstd
instead of x and xo, and the backward gets the streams back one of two ways (`residual_ckpt_interval`):

- Inverting (0): a write is invertible, X = R^T (X' - b y), so the backward recovers the streams from the layer above
  (X2 from the top layer's output or the layer above's backward through the `StreamSlot`; X1 before the FFN backward,
  X0 before the branch backward, X0 on to the layer below). The streams travel as hi + lo BF16 pairs: hi is what the
  model reads, hi + lo holds the FP32 write to ~2^-17, so the inversion reproduces the forward's streams; lo and the
  rstd of hi reach the next layer through the slot, the first layer's input (the embedding) has no lo and is saved as
  one copy. The backward of a sublayer is two passes over the streams, `unwrite_bwd` and `read_proj_bwd`.
- Recomputing (k > 0; mHC's recomputation blocks, arXiv 2512.24880 sec. 4.3.2): every k-th layer keeps its input
  streams (a saved tensor; layer 0 its embedding copy) and the forward leaves each layer's write coefficients and
  branch outputs in the slot; in the backward the top layer of each block of k layers recomputes the inputs of the
  layers below it from the kept one (`recompute_block_streams`) and leaves them in the slot, where each layer takes
  its own. No lo parts: the streams are rounded once per write as on the module path, the writes cost two passes
  instead of five, and a sublayer's backward is two passes again: `rewrite_bwd` (the forward write again, the read
  and the write's backward in one pass) or `read_write_bwd`, then `read_proj_bwd`. The first layer takes this path
  in both modes. Memory: one [T, N, D] tensor per kept layer, plus one per layer of the block being run, transiently.

mhc (Sinkhorn transport, not invertible) has no fused block. The hyper-connection module asserts a model-parallel world
size of one, so the blocks' model-parallel gathers and scatters are no-ops on that path. One backward per forward: the
stream hand-off between layers (the slot, the top layer's streams on its ctx) is consumed by the first backward, so a
second one through the same graph (retain_graph, double backward) is rejected.
"""

from typing import Optional, Tuple

import torch

from xllm.modules.fused_ops import hyper_connection as fused
from xllm.modules.residual.hyper_connection import StreamSlot, controller_params, expand_streams


def kept_streams(X_saved: torch.Tensor, N: int) -> torch.Tensor:
    """A kept input as streams [T, N, D]; the first layer keeps one copy of the embedding [T, D] (every stream)."""
    return X_saved if X_saved.dim() == 3 else expand_streams(X_saved, N)


def recompute_block_streams(slot: StreamSlot, hc_cls, first: int, top: int, N: int):
    """Recomputing mode: the input streams of layers first + 1 .. top from the kept input of layer `first` and the writes
    the forward left in the slot (coefficient logits and branch outputs: saved tensors, not copies), each left in the
    slot for its layer's backward. The forward's writes again, rounded the same way, so the streams are the forward's
    to the bit. One fresh [T, N, D] tensor per layer; the kept input is read, not written."""
    X = kept_streams(slot.take((first, 'in')), N)
    for layer in range(first, top):
        hc1_s, hc1_b, l1, y1, hc2_s, hc2_b, l2, y2 = slot.take((layer, 'writes'))
        _, R1, b1 = hc_cls.coef_fwd(l1, hc1_s, hc1_b)
        _, R2, b2 = hc_cls.coef_fwd(l2, hc2_s, hc2_b)
        X1, _, _ = fused.write_fwd(X, R1, b1, y1)
        X, _, _ = fused.write_fwd(X1, R2, b2, y2, inplace=True)
        slot.put((layer + 1, 'in'), X)


class FusedStreams:
    """One layer's stream path (see the module docstring). `hc_cls` is the layer's HyperConnection class, `slot` the
    model's StreamSlot, `eps` the controllers' normalization eps."""

    def __init__(self, hc_cls, slot: StreamSlot, layer_id: int, last_layer: bool, ckpt_interval: int, eps: float):
        self.hc_cls, self.slot, self.layer_id, self.last_layer, self.k, self.eps = hc_cls, slot, layer_id, last_layer, ckpt_interval, eps
        self.first = layer_id == 0  # the embedding: saved as one copy, no lo part
        self.split = ckpt_interval == 0  # inverting: the streams travel as hi + lo
        self.kept = ckpt_interval > 0 and layer_id % ckpt_interval == 0  # recomputing: this layer's input is kept
        self.top = ckpt_interval > 0 and (last_layer or (layer_id + 1) % ckpt_interval == 0)  # and this one recomputes
        self.X_out = None  # inverting, the top layer: its output streams (hi, lo) for the backward

    # ---- forward

    def read1(self, x: torch.Tensor, w1, b1, s1, w2, b2, s2) -> torch.Tensor:
        """The layer's input streams x [B, L, N, D] and the two controllers' parameters -> the branch input x [B, L, D]."""
        bsz, seq_len, N, D = x.shape
        self.shape = (bsz, seq_len, N, D)
        X = x.contiguous().view(bsz * seq_len, N, D)
        if self.first:
            self.slot.reset()
            X_lo, rstd_in = None, None
        else:
            X_lo, rstd_in = self.slot.take((self.layer_id, 'carry'))  # from the layer below's write: lo (inverting), rstd
        self.p1 = controller_params(w1, b1, s1, X.dtype)
        self.p2 = controller_params(w2, b2, s2, X.dtype)
        self.l1, self.rstd1, self.a1, self.R1, self.b1 = self.hc_cls.controller_fwd(X, *self.p1, self.eps, rstd_in)
        self.X, self.X_lo = X, X_lo
        return fused.read_fwd(X, self.a1).view(bsz, seq_len, D)

    def write1(self, y1: torch.Tensor) -> torch.Tensor:
        """The branch output y1 [B, L, D] into the streams -> the FFN input xo [B, L, D]."""
        bsz, seq_len, N, D = self.shape
        self.y1 = y1.contiguous().view(bsz * seq_len, D)
        self.X1, self.X1_lo, rstd_x1 = fused.write_fwd(self.X, self.R1, self.b1, self.y1, lo=self.X_lo, split=self.split,
                                                       rstd_eps=self.eps)
        # the kept input streams: the first layer's as one copy of the embedding
        self.X_saved = self.X[:, 0].contiguous() if self.first else (self.X if self.kept else None)
        self.X = self.X_lo = None
        self.l2, self.rstd2, self.a2, self.R2, self.b2 = self.hc_cls.controller_fwd(self.X1, *self.p2, self.eps, rstd_x1)
        return fused.read_fwd(self.X1, self.a2).view(bsz, seq_len, D)

    def write2(self, y2: torch.Tensor) -> torch.Tensor:
        """The FFN output y2 [B, L, D] into the streams, in place over X1 (read above, not saved) -> the layer's output
        streams [B, L, N, D]."""
        bsz, seq_len, N, D = self.shape
        self.y2 = y2.contiguous().view(bsz * seq_len, D)
        X2, X2_lo, rstd_x2 = fused.write_fwd(self.X1, self.R2, self.b2, self.y2, lo=self.X1_lo, split=self.split,
                                             rstd_eps=None if self.last_layer else self.eps, inplace=True)
        if not self.last_layer:
            self.slot.put((self.layer_id + 1, 'carry'), (X2_lo, rstd_x2))  # to the next layer's forward
        if self.kept:
            self.slot.put((self.layer_id, 'in'), self.X_saved)  # for the block's top layer's recompute (the same tensor)
        if self.k > 0 and not self.top:
            self.slot.put((self.layer_id, 'writes'), (self.p1[2], self.p1[1], self.l1, self.y1, self.p2[2], self.p2[1],
                                                      self.l2, self.y2))  # for the recompute
        # inverting: the top layer's output streams stay here, not in the saved tensors, so the backward can free them
        # right after inverting them; detached, or ctx -> X2 -> grad_fn would be a cycle
        if self.last_layer and self.split:
            self.X_out = (X2.detach(), X2_lo)
        return X2.view(bsz, seq_len, N, D)

    def saved(self) -> tuple:
        """The tensors the block saves for the backward (`restore` takes them back): the kept input, the controllers,
        their logits and rstd, the branch outputs. Nothing else of the forward stays on the object."""
        out = (self.X_saved, *self.p1, *self.p2, self.l1, self.rstd1, self.l2, self.rstd2, self.y1, self.y2)
        self._clear()
        return out

    def _clear(self):
        for name in ('X', 'X_lo', 'X1', 'X1_lo', 'X_saved', 'p1', 'p2', 'l1', 'rstd1', 'a1', 'R1', 'b1', 'l2', 'rstd2',
                     'a2', 'R2', 'b2', 'y1', 'y2', 'G2', 'dR1', 'db1', 'dR2', 'db2', 'dX1', 'grads2'):
            setattr(self, name, None)

    # ---- backward

    def restore(self, rc: tuple, out_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The saved tensors and dL/dX2 [B, L, N, D] -> (xo [B, L, D], dL/dy2 [B, L, D]): X1 and xo = read(X1, a2) back,
        with the layer's input at hand (kept, recomputed, or the first layer's embedding) as the forward's write again,
        else by inverting the second write; the write's backward (dy2, db2, dR2) in the same pass."""
        X_in, w1, b1, s1, w2, b2, s2, self.l1, self.rstd1, self.l2, self.rstd2, self.y1, self.y2 = rc
        self.p1, self.p2 = (w1, b1, s1), (w2, b2, s2)
        bsz, seq_len, N, D = out_grad.shape
        self.shape = (bsz, seq_len, N, D)
        T = bsz * seq_len
        self.G2 = out_grad.contiguous().view(T, N, D)  # autograd's tensor: read only
        self.a1, self.R1, self.b1 = self.hc_cls.coef_fwd(self.l1, s1, b1)
        self.a2, self.R2, self.b2 = self.hc_cls.coef_fwd(self.l2, s2, b2)
        slot, layer_id = self.slot, self.layer_id
        if self.k > 0:
            if self.top:  # the block's top layer: the inputs of the layers below it
                recompute_block_streams(slot, self.hc_cls, layer_id - layer_id % self.k, layer_id, N)
            if X_in is None:
                X_in = slot.take((layer_id, 'in'))
        if X_in is not None:
            self.X = kept_streams(X_in, N)
            self.X1, xo, self.dR2, self.db2, dy2 = fused.rewrite_bwd(self.X, self.R1, self.b1, self.y1, self.a2, self.R2,
                                                                     self.b2, self.y2, self.G2)
        else:
            if self.last_layer:
                # one backward per forward: the stream hand-off (X_out, the slot) is consumed by the first backward
                assert self.X_out is not None, \
                    "the hyper-connection fused block runs one backward per forward (no retain_graph / double backward)"
                (X2, X2_lo), self.X_out = self.X_out, None  # the model's output: not ours to overwrite
            else:
                X2, X2_lo = slot.take((layer_id, 'out'))
            self.X1, self.X1_lo, xo, self.dR2, self.db2, dy2 = fused.unwrite_bwd(X2, X2_lo, self.R2, self.b2, self.y2,
                                                                                  self.a2, self.G2, inplace=not self.last_layer)
            del X2, X2_lo
        return xo.view(bsz, seq_len, D), dy2.view(bsz, seq_len, D)

    def after_ffn(self, xo_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """dL/dxo [B, L, D] -> (the branch input x [B, L, D] again, dL/dy1 [B, L, D]). The FFN controller's backward (its
        dlogits need da2 from the read), the whole dX1 in one pass, then X0 (x = read(X0, a1), dy1, db1, dR1), which the
        inverting mode hands on to the layer below."""
        bsz, seq_len, N, D = self.shape
        xo_grad = xo_grad.reshape(bsz * seq_len, D)
        da2 = fused.read_da(self.X1, xo_grad)
        proj2, *self.grads2 = self.hc_cls.controller_bwd(self.X1, *self.p2, self.l2, self.rstd2, da2, self.dR2, self.db2)
        del da2
        self.dR2 = self.db2 = None
        self.dX1 = fused.read_proj_bwd(self.X1, self.R2, self.a2, self.G2, xo_grad, *proj2)
        self.G2 = None
        del xo_grad, proj2
        if self.X is not None:
            x, self.dR1, self.db1, dy1 = fused.read_write_bwd(self.X, self.a1, self.R1, self.b1, self.y1, self.dX1)
        else:
            self.X, X_lo, x, self.dR1, self.db1, dy1 = fused.unwrite_bwd(self.X1, self.X1_lo, self.R1, self.b1, self.y1,
                                                                         self.a1, self.dX1, inplace=True)
            if self.layer_id > 1:  # the first layer rewrites its X1 from the embedding copy, not from these
                self.slot.put((self.layer_id - 1, 'out'), (self.X, X_lo))
            del X_lo
        self.X1 = self.X1_lo = None
        return x.view(bsz, seq_len, D), dy1.view(bsz, seq_len, D)

    def after_branch(self, x_grad: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """dL/dx [B, L, D] -> (dL/dX0 [B, L, N, D], the controllers' gradients (w1, b1, s1, w2, b2, s2)): the branch
        controller's backward, then the whole dX0 in one pass, in place over dX1. The controller gradients are FP32;
        autograd casts them to the parameters' dtypes."""
        bsz, seq_len, N, D = self.shape
        x_grad = x_grad.reshape(bsz * seq_len, D)
        da1 = fused.read_da(self.X, x_grad)
        proj1, *grads1 = self.hc_cls.controller_bwd(self.X, *self.p1, self.l1, self.rstd1, da1, self.dR1, self.db1)
        del da1
        x_grad = fused.read_proj_bwd(self.X, self.R1, self.a1, self.dX1, x_grad, *proj1, out=self.dX1).view(bsz, seq_len, N, D)
        grads = (*grads1, *self.grads2)
        self._clear()
        return x_grad, grads
