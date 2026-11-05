"""Fused attention output gate of the Gekko block (CuTe DSL, Blackwell): per row (b, l, h) of V channels

    forward:  attn = rmsnorm(sca_out + awk_out) * gate(rmx)            (gate: softplus(beta = ln 2) or silu)
    backward: sca_out_grad (= awk_out_grad) = rmsnorm_bwd(attn_grad * gate(rmx)), rmx_grad = gate'(rmx) attn_grad * norm

replacing the add, the small-group rmsnorm, the gate activation and the multiply (and in the backward their
recompute plus two multiplies, the memory-efficient rmsnorm backward and the activation backward): 10 / 22 passes
over [B, L, H * V] become 4 / 6. Every lane holds 8 channels (16-byte loads): rows of 256 channels take one warp,
V = 128 takes 16 lanes, larger rows V / 256 warps with a shared-memory reduction (one smem slot per reduction, all
loads issued before the first barrier). The intermediate roundings of the torch path
(bf16 sum, bf16 norm, bf16 gate) are reproduced so the numerics match it."""
import math
import os
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

try:
    from .adaptive_working_memory import _compiled, _fake
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _awm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_awm)
    _compiled, _fake = _awm._compiled, _awm._fake

WARPS = 8  # per CTA
N_RED = 3  # cross-warp reductions per kernel (at most)
LN2 = math.log(2.0)
LOG2E = 1.0 / LN2


@cute.jit
def _warp_sum(v: Float32, steps: cutlass.Constexpr = 5) -> Float32:
    """Sum over groups of 2^steps consecutive lanes (the whole warp for steps = 5)."""
    for o in cutlass.range_constexpr(steps):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << o)
    return v


@cute.jit
def _softplus(x: Float32) -> Float32:
    # torch: softplus(x, beta = ln 2, threshold = 20) = log1p(exp(beta x)) / beta = log2(1 + 2^x), x when beta x > 20
    y = cute.math.log2(1.0 + cute.math.exp2(x, fastmath=True), fastmath=True)
    return x if x > 20.0 / LN2 else y


@cute.jit
def _sigmoid_ln2(x: Float32) -> Float32:
    # sigmoid(beta x) with beta = ln 2: 1 / (1 + 2^-x)
    return cute.arch.rcp_approx(1.0 + cute.math.exp2(-x, fastmath=True))


@cute.jit
def _sigmoid(x: Float32) -> Float32:
    return cute.arch.rcp_approx(1.0 + cute.math.exp2(-x * LOG2E, fastmath=True))


@cute.jit
def _row_sum(v: Float32, red: cute.Tensor, wpr: cutlass.Constexpr, widx: Int32, lane: Int32, slot: cutlass.Constexpr,
             steps: cutlass.Constexpr) -> Float32:
    """Sum over the wpr warps of a row (red: smem [N_RED * WARPS] floats, one slot per warp of the CTA and reduction:
    every reduction of a kernel uses its own slot, so no barrier is needed after the reads - a reduction that reused
    a slot would race with the slow warps' reads of the previous one)."""
    assert 0 <= slot < N_RED, slot
    v = _warp_sum(v, steps)
    if cutlass.const_expr(wpr > 1):
        if lane == 0:
            red[slot * WARPS + widx] = v
        cute.arch.sync_threads()
        base = (widx // wpr) * wpr
        v = Float32(0.0)
        for i in cutlass.range_constexpr(wpr):
            v = v + red[slot * WARPS + base + i]
    return v


def _geometry(V: int) -> Tuple[int, int, int, int]:
    """(warps per row, lanes per row, channels per lane, rows per CTA): rows of V >= 256 take WPR = V / 256 warps of
    8 channels per lane; V = 128 takes 16 lanes (two rows per warp). Every lane loads 16 bytes per tensor. Raises
    ValueError for an unsupported V (the single source of truth for attn_gate_supported)."""
    WPR = max(1, V // 256)
    LPR = 32 if V >= 256 else 16
    E = V // (LPR * WPR)
    if V % 32 or E != 8 or WARPS % WPR:
        raise ValueError(f"attn gate: unsupported head width {V}")
    return WPR, LPR, E, WARPS * (32 // LPR) // WPR


class AttnGateFwd:
    """attn[row] = bf16(bf16(rmsnorm(bf16(sca[row] + awk[row]))) * bf16(gate(rmx[row])))."""

    def __init__(self, V: int, silu: bool):
        self.V, self.silu = V, silu
        self.WPR, self.LPR, self.E, self.RPC = _geometry(V)

    @cute.jit
    def __call__(self, tA: cute.Tensor, tB: cute.Tensor, tR: cute.Tensor, tOut: cute.Tensor, rows: Int32, eps: Float32, stream: cuda.CUstream):
        rows_per_cta = self.RPC
        self.kernel(tA.iterator, tB.iterator, tR.iterator, tOut.iterator, rows, eps).launch(
            grid=[(rows + rows_per_cta - 1) // rows_per_cta, 1, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pA: cute.Pointer, pB: cute.Pointer, pR: cute.Pointer, pOut: cute.Pointer, rows: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        V, E, WPR, LPR, RPC = self.V, self.E, self.WPR, self.LPR, self.RPC
        RPW = 32 // LPR  # rows per warp
        STEPS = {32: 5, 16: 4}[LPR]
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(N_RED * WARPS), 16)
        lane = tidx % 32
        widx = tidx // 32
        row = bidx * RPC + (widx // WPR) * RPW + lane // LPR
        col = (widx % WPR) * (LPR * E) + (lane % LPR) * E
        valid = row < rows
        if not valid:
            row = rows - 1  # keep the whole CTA in the reductions
        off = Int64(row) * V + col
        sv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        a8 = cute.make_tensor((pA + off).align(2 * E), cute.make_layout(E)).load().to(Float32)
        b8 = cute.make_tensor((pB + off).align(2 * E), cute.make_layout(E)).load().to(Float32)
        sv.store((a8 + b8).to(BFloat16).to(Float32))  # the torch path adds in bf16
        ss = Float32(0.0)
        for k in cutlass.range_constexpr(E):
            ss = ss + sv[k] * sv[k]
        rv = cute.make_rmem_tensor(cute.make_layout(E), Float32)  # all loads are issued before the first barrier
        rv.store(cute.make_tensor((pR + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
        ss = _row_sum(ss, red, WPR, widx, lane, 0, STEPS)
        rstd = cute.math.rsqrt(ss * (1.0 / V) + eps, fastmath=True)
        ov = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        for k in cutlass.range_constexpr(E):
            n = BFloat16(sv[k] * rstd).to(Float32)
            if cutlass.const_expr(self.silu):
                g = BFloat16(rv[k] * _sigmoid(rv[k])).to(Float32)
            else:
                g = BFloat16(_softplus(rv[k])).to(Float32)
            ov[k] = n * g
        if valid:
            cute.make_tensor((pOut + off).align(2 * E), cute.make_layout(E)).store(ov.load().to(BFloat16))


class AttnGateBwd:
    """From attn_grad g, sca, awk, rmx: n = bf16(rmsnorm(bf16(sca + awk))), r = bf16(gate(rmx)),
    dn = bf16(g r), dsca = rstd (dn - n mean(dn n)); dr = bf16(g n), drmx = gate'(rmx) dr; optionally also the
    recomputed attn = bf16(n r) (the wo weight gradient needs it)."""

    def __init__(self, V: int, silu: bool, want_attn: bool, H: int = 1, want_delta: bool = False):
        self.V, self.silu, self.want_attn, self.H, self.want_delta = V, silu, want_attn, H, want_delta
        self.WPR, self.LPR, self.E, self.RPC = _geometry(V)

    @cute.jit
    def __call__(self, tG: cute.Tensor, tA: cute.Tensor, tB: cute.Tensor, tR: cute.Tensor, tDs: cute.Tensor, tDr: cute.Tensor,
                 tAttn: cute.Tensor, tDelta: cute.Tensor, rows: Int32, L: Int32, eps: Float32, stream: cuda.CUstream):
        rows_per_cta = self.RPC
        self.kernel(tG.iterator, tA.iterator, tB.iterator, tR.iterator, tDs.iterator, tDr.iterator, tAttn.iterator, tDelta.iterator,
                    rows, L, eps).launch(
            grid=[(rows + rows_per_cta - 1) // rows_per_cta, 1, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pG: cute.Pointer, pA: cute.Pointer, pB: cute.Pointer, pR: cute.Pointer, pDs: cute.Pointer, pDr: cute.Pointer,
               pAttn: cute.Pointer, pDelta: cute.Pointer, rows: Int32, L: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        V, E, WPR, LPR, RPC = self.V, self.E, self.WPR, self.LPR, self.RPC
        RPW = 32 // LPR  # rows per warp
        STEPS = {32: 5, 16: 4}[LPR]
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(N_RED * WARPS), 16)
        lane = tidx % 32
        widx = tidx // 32
        row = bidx * RPC + (widx // WPR) * RPW + lane // LPR
        col = (widx % WPR) * (LPR * E) + (lane % LPR) * E
        valid = row < rows
        if not valid:
            row = rows - 1
        off = Int64(row) * V + col
        sv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        a8 = cute.make_tensor((pA + off).align(2 * E), cute.make_layout(E)).load().to(Float32)
        b8 = cute.make_tensor((pB + off).align(2 * E), cute.make_layout(E)).load().to(Float32)
        sv.store((a8 + b8).to(BFloat16).to(Float32))
        ss = Float32(0.0)
        for k in cutlass.range_constexpr(E):
            ss = ss + sv[k] * sv[k]
        rv = cute.make_rmem_tensor(cute.make_layout(E), Float32)  # all loads are issued before the first barrier
        rv.store(cute.make_tensor((pR + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
        gv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        gv.store(cute.make_tensor((pG + off).align(2 * E), cute.make_layout(E)).load().to(Float32))
        ss = _row_sum(ss, red, WPR, widx, lane, 0, STEPS)
        rstd = cute.math.rsqrt(ss * (1.0 / V) + eps, fastmath=True)
        dn = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        drv = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        av = cute.make_rmem_tensor(cute.make_layout(E), Float32)
        dot = Float32(0.0)
        for k in cutlass.range_constexpr(E):
            n = BFloat16(sv[k] * rstd).to(Float32)
            sv[k] = n
            x = rv[k]
            if cutlass.const_expr(self.silu):
                sg = _sigmoid(x)
                r = BFloat16(x * sg).to(Float32)
                dgate = sg * (1.0 + x * (1.0 - sg))
            else:
                r = BFloat16(_softplus(x)).to(Float32)
                dgate = 1.0 if x > 20.0 / LN2 else _sigmoid_ln2(x)
            dn[k] = BFloat16(gv[k] * r).to(Float32)
            av[k] = n * r
            dot = dot + dn[k] * n
            dr = BFloat16(gv[k] * n).to(Float32)
            drv[k] = dr * dgate
        dot = _row_sum(dot, red, WPR, widx, lane, 1, STEPS) * (1.0 / V)
        for k in cutlass.range_constexpr(E):
            dn[k] = rstd * (dn[k] - sv[k] * dot)
        if cutlass.const_expr(self.want_delta):
            # delta[b, h, l] = sum_v dS[v] y[v] (bf16 operands, as the SCA backward's RowDot) for the SCA backward
            dsb = dn.load().to(BFloat16).to(Float32)
            dd = Float32(0.0)
            for k in cutlass.range_constexpr(E):
                dd = dd + dsb[k] * a8[k]
            dd = _row_sum(dd, red, WPR, widx, lane, 2, STEPS)
            if valid:
                if tidx % (LPR * WPR) == 0:
                    H = self.H
                    h = row % H
                    bl = row // H
                    b = bl // L
                    l = bl - b * L
                    cute.make_tensor(pDelta + ((b * H + h) * L + l), cute.make_layout(1))[0] = dd
        if valid:
            cute.make_tensor((pDs + off).align(2 * E), cute.make_layout(E)).store(dn.load().to(BFloat16))
            cute.make_tensor((pDr + off).align(2 * E), cute.make_layout(E)).store(drv.load().to(BFloat16))
            if cutlass.const_expr(self.want_attn):
                cute.make_tensor((pAttn + off).align(2 * E), cute.make_layout(E)).store(av.load().to(BFloat16))


def attn_gate_fwd(sca_out: torch.Tensor, awk_out: torch.Tensor, rmx: torch.Tensor, H: int, gate: str, eps: float) -> torch.Tensor:
    """attn [B, L, H * V] = rmsnorm_per_head(sca_out + awk_out) * gate(rmx) (bf16; inputs [B, L, H * V])."""
    B, L, D = sca_out.shape
    V = D // H
    rows = B * L * H
    out = torch.empty_like(sca_out)
    fn = _compiled(("attn_gate_fwd", V, gate), lambda: AttnGateFwd(V, gate == "silu"), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), Int32(0), Float32(0.0)))
    fn(sca_out.contiguous().reshape(-1), awk_out.contiguous().reshape(-1), rmx.contiguous().reshape(-1), out.reshape(-1), rows, float(eps))
    return out


def attn_gate_bwd(attn_grad: torch.Tensor, sca_out: torch.Tensor, awk_out: torch.Tensor, rmx: torch.Tensor, H: int, gate: str,
                  eps: float, want_attn: bool = False, want_delta: bool = False):
    """(sca_out_grad [B, L, H * V] (also the awk_out gradient), rmx_grad [B, L, H * V], attn or None, delta or None) of
    attn_gate_fwd; delta [B, H, L] fp32 = rowsum(sca_out_grad * sca_out) is the SCA backward's delta."""
    B, L, D = sca_out.shape
    V = D // H
    rows = B * L * H
    ds = torch.empty_like(sca_out)
    dr = torch.empty_like(sca_out)
    attn = torch.empty_like(sca_out) if want_attn else None
    delta = torch.empty(B, H, L, dtype=torch.float32, device=sca_out.device) if want_delta else None
    fn = _compiled(("attn_gate_bwd", V, gate, want_attn, H if want_delta else 0), lambda: AttnGateBwd(V, gate == "silu", want_attn, H, want_delta),
                   lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16),
                            _fake(BFloat16), _fake(Float32, 4), Int32(0), Int32(0), Float32(0.0)))
    fn(attn_grad.contiguous().reshape(-1), sca_out.contiguous().reshape(-1), awk_out.contiguous().reshape(-1), rmx.contiguous().reshape(-1),
       ds.reshape(-1), dr.reshape(-1), (attn if want_attn else ds).reshape(-1),
       (delta if want_delta else torch.empty(16, dtype=torch.float32, device=sca_out.device)).reshape(-1), rows, L, float(eps))
    return ds, dr, attn, delta


def attn_gate_supported(x: torch.Tensor, H: int, gate: str, dropout: float) -> bool:
    if os.environ.get("XLLM_CUTE_GATE", "1") == "0" or x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 3 or dropout != 0:
        return False
    if gate not in ("softplus", "silu") or x.shape[-1] % H:
        return False
    if x.numel() >= 2 ** 31:
        return False
    try:
        _geometry(x.shape[-1] // H)
    except ValueError:
        return False
    return torch.cuda.get_device_capability(x.device)[0] >= 9
