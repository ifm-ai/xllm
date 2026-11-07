"""Hopper fp8 operands and GEMMs (xllm/modules/fused_ops/fp8_sm90.py) against the blockwise reference and fp32 matmuls."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "fp8_sm90", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "fp8_sm90.py"))
f8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f8)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                                reason="needs a Hopper GPU")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("kind", ["act", "grad", "weight"])
@pytest.mark.parametrize("R,C", [(128, 128), (256, 384), (8192, 2560), (1024, 4096)])
def test_quantize(R, C, kind):
    torch.manual_seed(0)
    x = torch.randn(R, C, device="cuda") * torch.exp2(torch.randint(-20, 20, (R, 1), device="cuda").float())
    x = x.to(torch.bfloat16)
    x[:128, :128] = 0  # an all-zero group / tile
    x[R - 1, C - 1] = 500 * 2.0 ** 10  # an amax with mantissa > 1.75 (saturates to 448)
    row, col = f8.quantize(x, kind=kind)
    rt, ct = f8._tiles(kind)
    assert row.tile == rt and col.tile == ct
    assert torch.equal(f8.dequantize(row), f8.quantize_reference(x, rt))
    assert torch.equal(f8.dequantize(col), f8.quantize_reference(x.t().contiguous(), ct))
    assert col.data.shape == (C, R) and col.data.is_contiguous()
    row2, none = f8.quantize(x, colwise=False, kind=kind)
    assert none is None and torch.equal(row2.data, row.data) and torch.equal(row2.scale, row.scale)
    none, col2 = f8.quantize(x, rowwise=False, kind=kind)
    assert none is None and torch.equal(col2.data, col.data) and torch.equal(col2.scale, col.scale)


def test_quantize_cat():
    torch.manual_seed(0)
    xs = [torch.randn(1024, c, device="cuda", dtype=torch.bfloat16) for c in (512, 128, 384)]
    row, col = f8.quantize_cat(xs, dim=1, kind="grad")
    cat = torch.cat(xs, 1)
    assert torch.equal(f8.dequantize(row), f8.quantize_reference(cat, row.tile))
    assert torch.equal(f8.dequantize(col), f8.quantize_reference(cat.t().contiguous(), col.tile))
    ws = [torch.randn(n, 640, device="cuda", dtype=torch.bfloat16) for n in (256, 128, 512)]
    row, col = f8.quantize_cat(ws, dim=0, kind="weight")
    stack = torch.cat(ws, 0)
    assert torch.equal(f8.dequantize(row), f8.quantize_reference(stack, True))
    assert torch.equal(f8.dequantize(col), f8.quantize_reference(stack.t().contiguous(), True))
    r0 = 0
    for w in ws:  # the forward GEMMs read row slices
        assert torch.equal(f8.dequantize(row.row_slice(r0, r0 + w.shape[0])), f8.quantize_reference(w, True))
        r0 += w.shape[0]


@pytest.mark.parametrize("wgrad_tile", ["x", "dy"])
@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (8192, 4096, 2560), (2048, 1024, 9216), (1024, 1536, 1536)])
def test_mm(M, N, K, wgrad_tile, monkeypatch):
    monkeypatch.setattr(f8, "WGRAD_TILE", wgrad_tile)
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    xr, xc = f8.quantize(x, kind="act")
    wr, wc = f8.quantize(w, kind="weight")
    dr, dc = f8.quantize(dy, kind="grad")
    tol = 0.05  # two e4m3 operands: ~4% relative rms error on random data
    assert _rel(f8.mm(xr, wr), x.float() @ w.float().t()) < tol
    assert _rel(f8.mm(xr, wr, bias=b), x.float() @ w.float().t() + b.float()) < tol
    assert _rel(f8.mm(dr, wc), dy.float() @ w.float()) < tol
    assert _rel(f8.mm(dc, xc), dy.float().t() @ x.float()) < tol
    for out, a, bb in ((f8.mm(xr, wr), xr, wr), (f8.mm(dr, wc), dr, wc), (f8.mm(dc, xc), dc, xc)):
        assert _rel(out, f8.dequantize(a) @ f8.dequantize(bb).t()) < 1e-2
    half = (N // 2) // 128 * 128  # a row slice of the weight (the forward GEMMs of a linear set)
    if half:
        ws = wr.row_slice(0, half)
        assert _rel(f8.mm(xr, ws), f8.dequantize(xr) @ f8.dequantize(ws).t()) < 1e-2


def test_linear_set():
    try:
        from xllm.modules.fused_ops import mxfp8 as mx
    except ImportError as e:
        pytest.skip(f"xllm not importable: {e}")
    torch.manual_seed(0)
    M, D = 2048, 1536
    ws = [torch.randn(n, D, device="cuda", dtype=torch.bfloat16) * 0.02 for n in (2048, 1024, 1024, 2048)]
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    ls = mx.MxLinearSet(ws)
    xr, xc = mx.mx_quantize(x)
    for i, w in enumerate(ws):
        assert _rel(ls.forward(i, xr), x.float() @ w.float().t()) < 0.05
    dys = [torch.randn(M, w.shape[0], device="cuda", dtype=torch.bfloat16) for w in ws]
    dx, dy_col = ls.dgrad(dys)
    assert _rel(dx, sum(d.float() @ w.float() for d, w in zip(dys, ws))) < 0.05
    for g, d in zip(ls.wgrad(dy_col, xc), dys):
        assert _rel(g, d.float().t() @ x.float()) < 0.05
    assert mx.mx_supported(x.device, M, D, 2048) and not mx.mx_supported(x.device, M, D, 640)


def test_unsupported_k():
    assert f8.supported(8192, 2560, 9728) and not f8.supported(8192, 640)
    a, _ = f8.quantize(torch.randn(256, 640, device="cuda", dtype=torch.bfloat16), colwise=False)
    with pytest.raises(AssertionError):
        f8.mm(a, a)


def test_lazy_weight_transpose(monkeypatch):
    torch.manual_seed(0)
    ws = [torch.randn(n, 1536, device="cuda", dtype=torch.bfloat16) * 0.02 for n in (2048, 512, 1024)]
    monkeypatch.setattr(f8, "LAZY_WEIGHT_T", True)
    row, lazy = f8.quantize_cat(ws, 0, kind="weight")
    assert isinstance(lazy, f8._LazyTransposed) and lazy._m is None and (lazy.rows, lazy.cols) == (1536, 3584)
    monkeypatch.setattr(f8, "LAZY_WEIGHT_T", False)
    row2, col = f8.quantize_cat(ws, 0, kind="weight")
    assert torch.equal(row.data.view(torch.uint8), row2.data.view(torch.uint8)) and torch.equal(row.scale, row2.scale)
    assert torch.equal(lazy.data.view(torch.uint8), col.data.view(torch.uint8)) and torch.equal(lazy.scale, col.scale)
    dy, _ = f8.quantize(torch.randn(1024, 3584, device="cuda", dtype=torch.bfloat16), colwise=False, kind="grad")
    assert torch.equal(f8.mm(dy, lazy), f8.mm(dy, col))


@pytest.mark.parametrize("M,E", [(1024, 1536), (2048, 9216)])
def test_swiglu_quantize(M, E):
    torch.manual_seed(0)
    h1 = torch.randn(M, E, device="cuda", dtype=torch.bfloat16) * 3
    h3 = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(M, E, device="cuda", dtype=torch.bfloat16) * 1e-3
    hidden = (torch.nn.functional.silu(h1.float()) * h3.float()).bfloat16()
    for got, ref in zip(f8.swiglu_quantize(h1, h3), f8.quantize(hidden, kind="act")):
        assert got.tile == ref.tile
        a, b = f8.dequantize(got), f8.dequantize(ref)
        assert (a != b).float().mean().item() < 1e-3 and _rel(a, b) < 1e-3  # sigmoid rounding: rare 1-ulp differences
    s = torch.sigmoid(h1.float())
    d1 = (g.float() * h3.float() * s * (1 + h1.float() * (1 - s))).bfloat16()
    d3 = (g.float() * h1.float() * s).bfloat16()
    for got, ref in zip(f8.swiglu_bwd_quantize(h1, h3, g), f8.quantize(torch.cat([d1, d3], 1), kind="grad")):
        assert got.tile == ref.tile
        a, b = f8.dequantize(got), f8.dequantize(ref)
        assert (a != b).float().mean().item() < 1e-3 and _rel(a, b) < 1e-3


def test_swiglu_quantize_column_halves():
    """h1 / h3 as the column halves of one (M, 2 E) GEMM output (B200 acacd71), d hidden contiguous."""
    torch.manual_seed(0)
    M, E = 1024, 1536
    h13 = torch.randn(M, 2 * E, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(M, E, device="cuda", dtype=torch.bfloat16) * 1e-3
    h1, h3 = h13[:, :E], h13[:, E:]
    for a, b in zip(f8.swiglu_quantize(h1, h3), f8.swiglu_quantize(h1.contiguous(), h3.contiguous())):
        assert torch.equal(a.data.view(torch.uint8), b.data.view(torch.uint8)) and torch.equal(a.scale, b.scale)
    for a, b in zip(f8.swiglu_bwd_quantize(h1, h3, g), f8.swiglu_bwd_quantize(h1.contiguous(), h3.contiguous(), g)):
        assert torch.equal(a.data.view(torch.uint8), b.data.view(torch.uint8)) and torch.equal(a.scale, b.scale)


def test_mm_residual_in_epilogue():
    """mm(..., residual=) through xllm_extension's cuBLASLt op (C in the epilogue) against the GEMM plus a separate add."""
    torch.manual_seed(0)
    M, N, K = 2048, 1536, 2560
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    r = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    xr, _ = f8.quantize(x, colwise=False, kind="act")
    wr, _ = f8.quantize(w, colwise=False, kind="weight")
    exact = f8.dequantize(xr) @ f8.dequantize(wr).t() + r.float()
    got = f8.mm(xr, wr, residual=r)
    sep = f8.mm(xr, wr).float() + r.float()
    assert got.dtype == torch.bfloat16 and got.shape == (M, N)
    assert _rel(got, exact) <= _rel(sep.bfloat16(), exact) * 1.01 + 1e-6  # one rounding instead of two
    assert _rel(got, exact) < 4e-3
    if f8._fused_c_op() is None:
        pytest.skip("xllm_extension without fp8_blockwise_gemm (separate add tested)")
    ws = wr.row_slice(512, 1536)  # a row slice of a stacked weight
    assert _rel(f8.mm(xr, ws, residual=r[:, :1024].contiguous()), f8.mm(xr, ws).float() + r[:, :1024].float()) < 4e-3


@pytest.mark.parametrize("M,N,K", [(512, 512, 512), (8192, 4096, 2560), (2048, 1024, 9216), (1024, 1536, 1536)])
def test_mm_deepgemm(M, N, K, monkeypatch):
    """XLLM_FP8_DEEPGEMM: the 1 x 128 x 128 x 128 GEMMs (forward, dgrad, weight gradient with WGRAD_TILE=x, a weight
    row slice, the lazily transposed weight, out=) through DeepGEMM, bitwise equal to cuBLASLt on the same operands."""
    dg = pytest.importorskip("deep_gemm")
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(n, K, device="cuda", dtype=torch.bfloat16) * 0.02 for n in (N // 2, N - N // 2)]
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(f8, "WGRAD_TILE", "x")
    xr, xc = f8.quantize(x, kind="act")
    dr, dc = f8.quantize(dy, kind="grad")
    wr, wc = f8.quantize_cat(ws, 0, kind="weight")
    pairs = [(xr, wr), (dr, wc), (dc, xc)]
    if N // 2 % 512 == 0:
        pairs.append((xr, wr.row_slice(0, N // 2)))
    monkeypatch.setattr(f8, "_DG", [None])
    ref = [f8.mm(a, b) for a, b in pairs]
    monkeypatch.setattr(f8, "_DG", [dg])
    for (a, b), r in zip(pairs, ref):
        assert not a.tile and b.tile
        got = f8.mm(a, b)
        assert got.shape == r.shape and torch.equal(got, r)
    out = torch.full_like(ref[0], float("nan"))
    assert f8.mm(xr, wr, out=out) is out and torch.equal(out, ref[0])
    assert _rel(f8.mm(dc, xc), dy.float().t() @ x.float()) < 0.05


@pytest.mark.parametrize("gate", ["softplus", "silu"])
@pytest.mark.parametrize("M,H,V", [(1024, 4, 128), (2048, 8, 256), (512, 2, 256)])
def test_gate_quantize(M, H, V, gate):
    """gate_quantize against cute attn_gate_fwd + quantize (rare 1-ulp differences from the fast-math reductions)."""
    try:
        from xllm.modules.fused_ops.cute.attn_gate import attn_gate_fwd
    except ImportError as e:
        pytest.skip(f"cute attn_gate not importable: {e}")
    torch.manual_seed(0)
    s, a, r = (torch.randn(M, H * V, device="cuda", dtype=torch.bfloat16) * c for c in (1.0, 0.5, 3.0))
    ref = attn_gate_fwd(s.view(1, M, -1), a.view(1, M, -1), r.view(1, M, -1), H, gate, 1e-6).view(M, -1)
    for got, want in zip(f8.gate_quantize(s, a, r, H, gate, 1e-6), f8.quantize(ref, kind="act")):
        assert got.tile == want.tile
        x, y = f8.dequantize(got), f8.dequantize(want)
        assert (x != y).float().mean().item() < 2e-3 and _rel(x, y) < 2e-3
    assert f8.gate_quantize(s[:, :H * 64], a[:, :H * 64], r[:, :H * 64], H, gate, 1e-6) is None  # V = 64: not handled
