"""CuTe DSL sliding chunk attention (Blackwell) against a dense PyTorch reference (forward) and its autograd
(backward). Needs an SM100 GPU and nvidia-cutlass-dsl."""
import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "cute_sca", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "cute",
                             "sliding_chunk_attention.py"))
sca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sca)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                                reason="needs a Blackwell GPU")

# (B, H, K, S, V, L, C, segment probability)
CASES = [
    (1, 1, 1, 256, 256, 128, 128, 0.0),
    (1, 2, 1, 256, 256, 512, 256, 0.0),
    (1, 4, 2, 256, 256, 1024, 256, 0.01),
    (2, 4, 2, 128, 128, 1024, 512, 0.02),
    (1, 2, 2, 256, 256, 8192, 4096, 0.001),
    (1, 4, 2, 256, 256, 8192, 4096, 0.0005),  # gekko-0.9B heads halved
    (1, 2, 1, 256, 512, 512, 256, 0.0),  # v head dim > 256: v slabs, dP from GEMMs
    (1, 2, 2, 256, 512, 1024, 256, 0.01),
    (1, 2, 2, 256, 2048, 8192, 4096, 0.0005),  # gekko-7.1B heads halved
]


def make_inputs(B, H, K, S, V, L, seg_p, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, L, H, S, device=dev, generator=g).to(torch.bfloat16)
    k = torch.randn(B, L, K, S, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(B, L, K, V, device=dev, generator=g).to(torch.bfloat16)
    dout = torch.randn(B, L, H, V, device=dev, generator=g).to(torch.bfloat16)
    seg = None
    if seg_p > 0:
        bos = torch.rand(B, L, device=dev, generator=g) < seg_p
        bos[:, 0] = True
        seg = torch.cumsum(bos.long(), 1) - 1
    return q, k, v, dout, seg


@pytest.mark.parametrize("B,H,K,S,V,L,C,seg_p", CASES)
def test_forward(B, H, K, S, V, L, C, seg_p):
    q, k, v, _, seg = make_inputs(B, H, K, S, V, L, seg_p)
    kmin = sca.segment_starts(seg, B, L, q.device)
    scale = 1.0 / S ** 0.5
    y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
    yr, lr = sca.sca_reference(q, k, v, C, scale, seg)
    assert torch.isfinite(y.float()).all()
    torch.testing.assert_close(y.float(), yr, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(lse2, lr, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("B,H,K,S,V,L,C,seg_p", CASES)
def test_backward(B, H, K, S, V, L, C, seg_p):
    q, k, v, dout, seg = make_inputs(B, H, K, S, V, L, seg_p)
    kmin = sca.segment_starts(seg, B, L, q.device)
    qmax = sca.query_limits(kmin)
    scale = 1.0 / S ** 0.5
    y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
    dq, dk, dv = sca.sca_bwd(q, k, v, y, lse2, dout, C, scale, kmin, qmax)
    qf, kf, vf = [t.float().requires_grad_(True) for t in (q, k, v)]
    yr, _ = sca.sca_reference(qf, kf, vf, C, scale, seg)
    yr.backward(dout.float())
    for g, r in ((dq, qf.grad), (dk, kf.grad), (dv, vf.grad)):
        assert torch.isfinite(g.float()).all()
        torch.testing.assert_close(g.float(), r, atol=4e-2, rtol=2e-2)


def test_segment_helpers():
    seg = torch.tensor([[0, 0, 1, 1, 1, 2, 3, 3]], device="cuda")
    kmin = sca.segment_starts(seg, 1, 8, "cuda")
    assert kmin.tolist() == [[0, 0, 2, 2, 2, 5, 6, 6]]
    qmax = sca.query_limits(kmin)
    # last query whose segment starts at or before key k
    assert qmax.tolist() == [[1, 1, 4, 4, 4, 5, 7, 7]]


# V > 256 shapes (dP from host-side GEMMs): (B, H, K, S, V, L, C, segment probability)
FP8_DP_CASES = [
    (1, 2, 1, 128, 512, 1024, 256, 0.0),
    (2, 2, 2, 256, 512, 768, 256, 0.02),
]


@pytest.mark.parametrize("pmat", ["0", "1"])
@pytest.mark.parametrize("B,H,K,S,V,L,C,seg_p", FP8_DP_CASES)
def test_backward_fp8_dp(B, H, K, S, V, L, C, seg_p, pmat, monkeypatch):
    """The MXFP8 dP GEMMs (sca_bwd(fp8_dp=True)) on both V > 256 backward paths (SCA_DV_PMAT=0: kernels reading
    dP, 1: materialized P / dS): dV is unaffected (bit-exact with the bf16-dP backward), dQ / dK within the e4m3
    noise of the reference and much closer to the reference than to zero."""
    monkeypatch.setenv("SCA_DV_PMAT", pmat)
    q, k, v, dout, seg = make_inputs(B, H, K, S, V, L, seg_p)
    kmin = sca.segment_starts(seg, B, L, q.device)
    qmax = sca.query_limits(kmin)
    scale = 1.0 / S ** 0.5
    y, lse2 = sca.sca_fwd(q, k, v, C, scale, kmin)
    assert sca._mx_dp_supported(q.device, C, V)
    dq, dk, dv = sca.sca_bwd(q, k, v, y, lse2, dout, C, scale, kmin, qmax, fp8_dp=True)
    dq16, dk16, dv16 = sca.sca_bwd(q, k, v, y, lse2, dout, C, scale, kmin, qmax)
    torch.testing.assert_close(dv, dv16, atol=0, rtol=0)
    qf, kf, vf = [t.float().requires_grad_(True) for t in (q, k, v)]
    yr, _ = sca.sca_reference(qf, kf, vf, C, scale, seg)
    yr.backward(dout.float())
    for g, g16, r in ((dq, dq16, qf.grad), (dk, dk16, kf.grad)):
        assert torch.isfinite(g.float()).all()
        err, err16 = (g.float() - r).norm() / r.norm(), (g16.float() - r).norm() / r.norm()
        assert err < 0.08 and err < 50 * err16 + 0.02, (err.item(), err16.item())
