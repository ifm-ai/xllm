"""MXFP8 operands and GEMMs (xllm/modules/fused_ops/mxfp8.py) against the OCP MX reference and fp32 matmuls."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "mxfp8", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "mxfp8.py"))
mx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mx)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="needs a Blackwell GPU")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("R,C", [(128, 128), (256, 384), (8192, 2560), (1024, 4096)])
def test_quantize(R, C):
    torch.manual_seed(0)
    x = torch.randn(R, C, device="cuda") * torch.exp2(torch.randint(-20, 20, (R, 1), device="cuda").float())
    x = x.to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero group
    x[1, 32] = 500 * 2.0 ** 10  # a group whose amax has mantissa > 1.75 (saturates to 448)
    row, col = mx.mx_quantize(x)
    assert torch.equal(mx.mx_dequantize(row), mx.mx_quantize_reference(x))
    assert torch.equal(mx.mx_dequantize(col), mx.mx_quantize_reference(x.t().contiguous()))
    assert col.data.shape == (C, R) and col.data.stride() == (1, C)  # the transposed view of an (R, C) buffer
    assert _rel(mx.mx_dequantize(row), x) < 0.04
    # single operands
    row2, none = mx.mx_quantize(x, colwise=False)
    assert none is None and torch.equal(row2.data, row.data) and torch.equal(row2.scale, row.scale)
    none, col2 = mx.mx_quantize(x, rowwise=False)
    assert none is None and torch.equal(col2.data, col.data) and torch.equal(col2.scale, col.scale)


def test_quantize_cat():
    torch.manual_seed(0)
    xs = [torch.randn(1024, c, device="cuda", dtype=torch.bfloat16) for c in (512, 128, 384)]
    row, col = mx.mx_quantize_cat(xs, dim=1)
    cat = torch.cat(xs, 1)
    assert torch.equal(mx.mx_dequantize(row), mx.mx_quantize_reference(cat))
    assert torch.equal(mx.mx_dequantize(col), mx.mx_quantize_reference(cat.t().contiguous()))
    r0 = 0
    for x in xs:  # rows r0.. of the colwise operand are the colwise operand of the part
        assert torch.equal(mx.mx_dequantize(col)[r0:r0 + x.shape[1]], mx.mx_quantize_reference(x.t().contiguous()))
        r0 += x.shape[1]
    ws = [torch.randn(n, 640, device="cuda", dtype=torch.bfloat16) for n in (256, 128, 512)]
    row, col = mx.mx_quantize_cat(ws, dim=0)
    stack = torch.cat(ws, 0)
    assert torch.equal(mx.mx_dequantize(row), mx.mx_quantize_reference(stack))
    assert torch.equal(mx.mx_dequantize(col), mx.mx_quantize_reference(stack.t().contiguous()))


@pytest.mark.parametrize("M,N,K", [(256, 128, 256), (8192, 4096, 2560), (2048, 1024, 9216)])
def test_mm(M, N, K):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    xr, xc = mx.mx_quantize(x)
    wr, wc = mx.mx_quantize(w)
    dr, dc = mx.mx_quantize(dy)
    tol = 0.05  # two e4m3 operands: ~4.1% relative rms error on random data
    assert _rel(mx.mx_mm(xr, wr), x.float() @ w.float().t()) < tol
    assert _rel(mx.mx_mm(xr, wr, bias=b), x.float() @ w.float().t() + b.float()) < tol
    assert _rel(mx.mx_mm(dr, wc), dy.float() @ w.float()) < tol
    assert _rel(mx.mx_mm(dc, xc), dy.float().t() @ x.float()) < tol
    # the fp8 result equals the fp32 product of the dequantised operands
    ref = mx.mx_dequantize(xr) @ mx.mx_dequantize(wr).t()
    assert _rel(mx.mx_mm(xr, wr), ref) < 1e-2


def test_linear_set():
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
    assert mx.mx_supported(x.device, M, D, 2048) and not mx.mx_supported(x.device, M, D, 100)



@pytest.mark.parametrize("M,F", [(128, 128), (2048, 1536)])
def test_swiglu_mx(M, F):
    """The cute SwiGLU kernels writing the MX operands directly (xllm/modules/fused_ops/cute/swiglu.py)."""
    import torch.nn.functional as F_
    torch.manual_seed(0)
    h1 = (torch.randn(M, F, device="cuda") * 2).to(torch.bfloat16)
    h3 = torch.randn_like(h1)
    g = torch.randn_like(h1)
    hidden = F_.silu(h1).float() * h3.float()  # bf16(silu(h1)) * h3, quantised from fp32
    row, col = mx.swiglu_fwd_mx(h1, h3)
    assert torch.equal(mx.mx_dequantize(row), mx.mx_quantize_reference(hidden))
    assert torch.equal(mx.mx_dequantize(col), mx.mx_quantize_reference(hidden.t().contiguous()))
    x = h1.float()
    sg = torch.sigmoid(x)
    s = (x * sg).to(torch.bfloat16).float()
    gs = (g.float() * h3.float()).to(torch.bfloat16).float()
    cat = torch.cat([gs * (sg * (1 + x * (1 - sg))), g.float() * s], 1)  # the kernel's formulas and roundings
    row, col = mx.swiglu_bwd_mx(h1, h3, g)
    assert torch.equal(mx.mx_dequantize(row), mx.mx_quantize_reference(cat))
    assert torch.equal(mx.mx_dequantize(col), mx.mx_quantize_reference(cat.t().contiguous()))


def test_swiglu_mx_strided_halves():
    """h1 / h3 as the column halves of one (M, 2F) GEMM output, the gradient contiguous."""
    import torch.nn.functional as F_
    torch.manual_seed(0)
    M, F = 256, 384
    h13 = (torch.randn(M, 2 * F, device="cuda") * 2).to(torch.bfloat16)
    h1, h3 = h13[:, :F], h13[:, F:]
    g = torch.randn(M, F, device="cuda", dtype=torch.bfloat16)
    hidden = F_.silu(h1).float() * h3.float()
    row, col = mx.swiglu_fwd_mx(h1, h3)
    assert torch.equal(mx.mx_dequantize(row), mx.mx_quantize_reference(hidden))
    row2, col2 = mx.swiglu_bwd_mx(h1, h3, g)
    row3, col3 = mx.swiglu_bwd_mx(h1.contiguous(), h3.contiguous(), g)
    assert torch.equal(row2.data, row3.data) and torch.equal(row2.scale, row3.scale)
    assert torch.equal(col2.data, col3.data) and torch.equal(col2.scale, col3.scale)
