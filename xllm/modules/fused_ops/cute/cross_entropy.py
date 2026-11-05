"""Fused softmax cross-entropy over a chunk of logits (CuTe DSL, Blackwell) for the chunked fused-linear cross
entropy: from the bf16 logits [C, V] (the output of the vocabulary GEMM) and the targets, one kernel writes

    loss[c]       = log(sum_v exp(x[c, v] - max)) - (x[c, t_c] - max)          (fp32, 0 for ignored rows)
    scores[c, v]  = softmax(x[c])[v] - [v == t_c]                              (fp32 and bf16, 0 for ignored rows)

replacing .float(), max, sub, index, exp, sum, log, div, index-sub, masked_fill_ and .to(bf16): ~12 passes over the
fp32 chunk become 1 bf16 read (+ 2 L2 re-reads) and the two score writes. A CTA of 256 threads handles one row."""
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

THREADS = 256
VEC = 8


@cute.jit
def _block_reduce(v: Float32, red: cute.Tensor, tidx: Int32, is_max: cutlass.Constexpr) -> Float32:
    for o in cutlass.range_constexpr(5):
        w = cute.arch.shuffle_sync_bfly(v, offset=1 << o)
        v = cute.arch.fmax(v, w) if is_max else v + w
    if tidx % 32 == 0:
        red[tidx // 32] = v
    cute.arch.sync_threads()
    v = red[0]
    for i in cutlass.range_constexpr(1, THREADS // 32):
        v = cute.arch.fmax(v, red[i]) if is_max else v + red[i]
    cute.arch.sync_threads()
    return v


class CeLogits:
    def __init__(self, V: int, want_loss: bool, want_grad: bool, want_fp32: bool = True):
        self.V, self.want_loss, self.want_grad, self.want_fp32 = V, want_loss, want_grad, want_fp32
        assert V % VEC == 0

    @cute.jit
    def __call__(self, tX: cute.Tensor, tT: cute.Tensor, tIgn: cute.Tensor, tLoss: cute.Tensor, tS32: cute.Tensor, tS16: cute.Tensor,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(tX.iterator, tT.iterator, tIgn.iterator, tLoss.iterator, tS32.iterator, tS16.iterator).launch(
            grid=[rows, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pT: cute.Pointer, pIgn: cute.Pointer, pLoss: cute.Pointer, pS32: cute.Pointer, pS16: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        V = self.V
        n_vec = V // VEC
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(THREADS // 32), 16)
        base = pX + Int64(row) * V
        # pass 1: row max
        m = Float32(-3.0e38)
        for i in cutlass.range(tidx, n_vec, THREADS):
            x8 = cute.make_tensor((base + i * VEC).align(16), cute.make_layout(VEC)).load().to(Float32)
            for k in cutlass.range_constexpr(VEC):
                m = cute.arch.fmax(m, x8[k])
        m = _block_reduce(m, red, tidx, True)
        # pass 2: sum of exp(x - max)
        s = Float32(0.0)
        for i in cutlass.range(tidx, n_vec, THREADS):
            x8 = cute.make_tensor((base + i * VEC).align(16), cute.make_layout(VEC)).load().to(Float32)
            for k in cutlass.range_constexpr(VEC):
                s = s + cute.math.exp(x8[k] - m, fastmath=True)
        s = _block_reduce(s, red, tidx, False)
        tgt = cute.make_tensor(pT + row, cute.make_layout(1))[0]
        ign = cute.make_tensor(pIgn + row, cute.make_layout(1))[0]
        ignored = ign != 0
        if cutlass.const_expr(self.want_loss):
            if tidx == 0:
                xt = cute.make_tensor(base + tgt, cute.make_layout(1))[0].to(Float32)
                loss = cute.math.log(s, fastmath=False) - (xt - m)
                cute.make_tensor(pLoss + row, cute.make_layout(1))[0] = 0.0 if ignored else loss
        if cutlass.const_expr(self.want_grad):
            inv = 1.0 / s
            o32 = pS32 + Int64(row) * V
            o16 = pS16 + Int64(row) * V
            for i in cutlass.range(tidx, n_vec, THREADS):
                x8 = cute.make_tensor((base + i * VEC).align(16), cute.make_layout(VEC)).load().to(Float32)
                sc = cute.make_rmem_tensor(cute.make_layout(VEC), Float32)
                for k in cutlass.range_constexpr(VEC):
                    p = cute.math.exp(x8[k] - m, fastmath=True) * inv
                    p = p - 1.0 if i * VEC + k == tgt else p
                    sc[k] = 0.0 if ignored else p
                if cutlass.const_expr(self.want_fp32):
                    cute.make_tensor((o32 + i * VEC).align(16), cute.make_layout(VEC)).store(sc.load())
                cute.make_tensor((o16 + i * VEC).align(16), cute.make_layout(VEC)).store(sc.load().to(BFloat16))


def ce_logits(logits: torch.Tensor, targets: torch.Tensor, ignore_mask: torch.Tensor, want_loss: bool, want_grad: bool,
              want_fp32: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(loss fp32 [C] or None, scores fp32 [C, V] or None, scores bf16 [C, V] or None) for bf16 logits [C, V], int64
    targets [C] (in [0, V)) and a bool ignore mask [C] (rows with loss and scores 0). want_fp32=False skips the fp32
    scores (None)."""
    C, V = logits.shape
    assert logits.dtype == torch.bfloat16 and targets.shape == (C,) and ignore_mask.shape == (C,)
    logits = logits.contiguous()
    dev = logits.device
    t = targets.to(torch.int32).contiguous()
    ign = ignore_mask.to(torch.uint8).contiguous()
    loss = torch.empty(C, dtype=torch.float32, device=dev) if want_loss else None
    s32 = torch.empty(C, V, dtype=torch.float32, device=dev) if want_grad and want_fp32 else None
    s16 = torch.empty(C, V, dtype=torch.bfloat16, device=dev) if want_grad else None
    dummy32 = loss if loss is not None else torch.empty(16, dtype=torch.float32, device=dev)
    fp32 = bool(want_fp32) and want_grad
    fn = _compiled(("ce_logits", V, want_loss, want_grad, fp32), lambda: CeLogits(V, want_loss, want_grad, fp32), lambda: (
        _fake(BFloat16), _fake(Int32, 4), _fake(Uint8, 1), _fake(Float32, 4), _fake(Float32), _fake(BFloat16), Int32(0)))
    fn(logits.reshape(-1), t, ign, (loss if want_loss else dummy32), (s32 if s32 is not None else dummy32).reshape(-1),
       (s16 if want_grad else logits).reshape(-1), C)
    return loss, s32, s16


def ce_logits_supported(logits: torch.Tensor) -> bool:
    if os.environ.get("XLLM_CUTE_CE", "1") == "0" or logits.dtype != torch.bfloat16 or not logits.is_cuda or logits.dim() != 2:
        return False
    if logits.shape[1] % VEC or logits.numel() >= 2 ** 31:
        return False
    return torch.cuda.get_device_capability(logits.device)[0] >= 9
