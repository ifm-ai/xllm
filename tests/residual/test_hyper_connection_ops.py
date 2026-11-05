"""GPU tests of the hyper-connection stream ops (xllm/modules/fused_ops/hyper_connection/).

Every op runs on BF16 streams and is compared, forward and all gradients, with the FP64 torch reference evaluated on
the same (BF16-rounded) inputs; the model-level test compares the fused and the reference path of whole tiny
Transformer / Gekko models.

    python -m pytest -q tests/residual/test_hyper_connection_ops.py
"""

import datetime
import os
import types

import pytest
import torch

from xllm.modules.fused_ops import hyper_connection as fused
from xllm.modules.residual import hyper_connection as hc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.fixture(autouse=True)
def _exact_matmul():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def close(name, x, ref, tol):
    err = rel(x, ref)
    assert err < tol, f"{name}: relative error {err:.2e} >= {tol:.0e}"


def leaf(t):
    return t.detach().clone().requires_grad_(True)


def reference_read(x, a):
    return (a.unsqueeze(-1) * x).sum(-2)


def reference_write(x, R, b, y):
    res = x if R is None else R @ x
    return res + b.unsqueeze(-1) * y.unsqueeze(-2)


SHAPES = [(4, 1536), (4, 256), (3, 96), (2, 1024), (4, 2560), (4, 4096), (4, 5120), (4, 1000), (4, 1032)]


@pytest.mark.parametrize("N,D,B,L", [(*s, 2, 37) for s in SHAPES] + [(4, 1536, 4, 8192)])
def test_read(N, D, B, L):
    x = torch.randn(B, L, N, D, device='cuda').bfloat16()
    a = torch.rand(B, L, N, device='cuda')
    dh = torch.randn(B, L, D, device='cuda').bfloat16()
    xf, af = leaf(x), leaf(a)
    h = fused.hc_read(xf, af)
    h.backward(dh)
    xr, ar = leaf(x.double()), leaf(a.double())
    hr = reference_read(xr, ar)
    hr.backward(dh.double())
    close('read h', h, hr, 1e-2)
    close('read dx', xf.grad, xr.grad, 1e-2)
    close('read da', af.grad, ar.grad, 1e-5)


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("N,D,B,L", [(*s, 2, 37) for s in SHAPES] + [(4, 1536, 4, 8192)])
def test_write(N, D, B, L, identity):
    x = torch.randn(B, L, N, D, device='cuda').bfloat16()
    R = None if identity else torch.randn(B, L, N, N, device='cuda') * 0.5
    b = torch.rand(B, L, N, device='cuda') * 2
    y = torch.randn(B, L, D, device='cuda').bfloat16()
    go = torch.randn(B, L, N, D, device='cuda').bfloat16()
    xf, bf, yf = leaf(x), leaf(b), leaf(y)
    Rf = None if identity else leaf(R)
    out = fused.hc_write(xf, Rf, bf, yf)
    out.backward(go)
    xr, br, yr = leaf(x.double()), leaf(b.double()), leaf(y.double())
    Rr = None if identity else leaf(R.double())
    outr = reference_write(xr, Rr, br, yr)
    outr.backward(go.double())
    close('write out', out, outr, 1e-2)
    close('write dx', xf.grad, xr.grad, 1e-2)
    close('write db', bf.grad, br.grad, 1e-5)
    close('write dy', yf.grad, yr.grad, 1e-2)
    if not identity:
        close('write dR', Rf.grad, Rr.grad, 1e-5)


@pytest.mark.parametrize("N,D,NL", [(4, 1536, 16), (4, 1536, 8), (4, 1536, 24), (3, 128, 15), (2, 128, 8)])
def test_logits(N, D, NL, B=2, L=37):
    x = torch.randn(B, L, N * D, device='cuda').bfloat16()
    w = (torch.randn(N * D, NL, device='cuda') / (N * D) ** 0.5).bfloat16()
    g = torch.randn(B, L, NL, device='cuda')
    xf, wf = leaf(x), leaf(w)
    out = fused.hc_logits(xf, wf, 1e-6)
    out.backward(g)
    xr, wr = leaf(x.double()), leaf(w.double())
    outr = hc.rms_normalize(xr, 1e-6) @ wr
    outr.backward(g.double())
    # dP is rounded to BF16 for the tensor-core backward, as for every BF16 weight gradient
    close('logits', out, outr, 1e-5)
    close('logits dx', xf.grad, xr.grad, 1e-2)
    close('logits dW', wf.grad, wr.grad, 1e-2)


def reference_gates(logits, scale, bias):
    a = torch.sigmoid(scale[0] * logits[..., :4] + bias[:4])
    b = 2 * torch.sigmoid(scale[1] * logits[..., 4:8] + bias[4:8])
    return a, b


@pytest.mark.parametrize("B,L", [(2, 37), (4, 8192)])  # several programs: the partial sums of dscale / dbias add up
def test_ihc_coefficients(B, L):
    logits = torch.randn(B, L, 8, device='cuda')
    scale = torch.randn(2, device='cuda')
    bias = torch.randn(8, device='cuda') * 0.5
    da, db = torch.randn(B, L, 4, device='cuda'), torch.randn(B, L, 4, device='cuda')
    lf, sf, bf = leaf(logits), leaf(scale), leaf(bias)
    a, _, b = hc.fused_coefficients(hc.IdentityHyperConnection, lf, sf, bf)
    ((a * da).sum() + (b * db).sum()).backward()
    lr, sr, br = leaf(logits.double()), leaf(scale.double()), leaf(bias.double())
    ar, b_r = reference_gates(lr, sr, br)
    ((ar * da).sum() + (b_r * db).sum()).backward()
    close('ihc a', a, ar, 1e-6)
    close('ihc b', b, b_r, 1e-6)
    close('ihc dlogits', lf.grad, lr.grad, 1e-5)
    close('ihc dscale', sf.grad, sr.grad, 1e-5)
    close('ihc dbias', bf.grad, br.grad, 1e-5)


@pytest.mark.parametrize("B,L", [(2, 37), (4, 8192)])
def test_ohc_coefficients(B, L):
    logits = torch.randn(B, L, 16, device='cuda')
    scale = torch.randn(2, device='cuda')
    bias = torch.randn(16, device='cuda') * 0.5
    alpha = hc.OrthogonalHyperConnection.ROTATION_SCALE  # the class constant the kernels use
    da, dR, db = (torch.randn(B, L, 4, device='cuda'), torch.randn(B, L, 4, 4, device='cuda'),
                  torch.randn(B, L, 4, device='cuda'))
    lf, sf, bf = leaf(logits), leaf(scale), leaf(bias)
    a, R, b = hc.fused_coefficients(hc.OrthogonalHyperConnection, lf, sf, bf)
    ((a * da).sum() + (R * dR).sum() + (b * db).sum()).backward()
    lr, sr, br = leaf(logits.double()), leaf(scale.double()), leaf(bias.double())
    ar, b_r = reference_gates(lr, sr, br)
    Rr = hc.quaternion_transport(alpha * lr[..., 8:] + br[8:])
    ((ar * da).sum() + (Rr * dR).sum() + (b_r * db).sum()).backward()
    close('ohc a', a, ar, 1e-6)
    close('ohc R', R, Rr, 1e-6)
    close('ohc b', b, b_r, 1e-6)
    close('ohc dlogits', lf.grad, lr.grad, 1e-5)
    close('ohc dscale', sf.grad, sr.grad, 1e-5)
    close('ohc dbias', bf.grad, br.grad, 1e-5)


@pytest.mark.parametrize("iterations,B,L", [(1, 2, 37), (5, 2, 37), (20, 2, 37), (20, 4, 16384)])
def test_sinkhorn(iterations, B, L):  # B * L > 8 * SMs * 32 runs the multi-block backward loop
    z = torch.randn(B, L, 4, 4, device='cuda') * 2
    dout = torch.randn(B, L, 4, 4, device='cuda')
    zf = leaf(z)
    out = fused.sinkhorn(zf, iterations)
    out.backward(dout)
    zr = leaf(z.double())
    outr = hc.sinkhorn_knopp(zr, iterations)
    outr.backward(dout.double())
    close('sinkhorn', out, outr, 1e-5)
    close('sinkhorn dz', zf.grad, zr.grad, 1e-4)


# ---------------------------------------------------------------------------------------------
# whole models: the fused stream ops against the torch reference path


@pytest.fixture(scope="module")
def single_process():
    from xllm.distributed.initialize import initialize_model_parallel, model_parallel_is_initialized
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29523")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0,
                                             timeout=datetime.timedelta(seconds=600))
    if not model_parallel_is_initialized():
        initialize_model_parallel(1, 1, 600)


def model_loss_and_grads(arch, residual_func, use_fused, **kwargs):
    from xllm.config import ModelConf
    from xllm.models.gekko import Gekko
    from xllm.models.transformer import Transformer
    from xllm.utils import create_on_gpu

    cfg = ModelConf(arch=arch, num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2, head_dim=32, chunk_size=32,
                    ffn_hidden_dim=256, swiglu=True, apply_rmsnorm=True, vocab_size=256,
                    causal_attn_backend='flash' if arch == 'transformer' else None, residual_func=residual_func,
                    **kwargs)
    torch.manual_seed(0)
    with create_on_gpu():
        model = (Transformer if arch == 'transformer' else Gekko)(cfg, types.SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    model = model.cuda().to(torch.bfloat16).train()
    with torch.no_grad():  # move the controllers off their near-identity init so every coefficient matters
        for name, p in model.named_parameters():
            if '.residual.' in name and not name.endswith('weight'):
                p.add_(torch.randn_like(p) * 0.5)
    g = torch.Generator(device='cuda').manual_seed(1)
    tokens = torch.randint(3, 256, (2, 64), device='cuda', generator=g)
    supported = fused.supported
    if not use_fused:
        fused.supported = lambda x: False
    try:
        loss, _, _ = model(tokens, multi_segments=False, targets=torch.roll(tokens, -1, dims=1))
        loss = loss.float().mean()
        loss.backward()
    finally:
        fused.supported = supported
    return loss.item(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}


MOE = dict(num_experts=4, num_activated_experts=2, expert_inter_dim=64, num_dense_layers=1)


@pytest.mark.parametrize("arch,residual_func,kwargs", [
    *[(arch, rf, {}) for arch in ('transformer', 'gekko') for rf in ('ihc', 'mhc', 'ohc')],
    ('transformer', 'ohc', MOE),
    ('transformer', 'mhc', dict(num_values=4, num_activated_values=2, **MOE)),
])
def test_model_fused_ops_match_reference(single_process, arch, residual_func, kwargs):
    loss_f, grads_f = model_loss_and_grads(arch, residual_func, True, **kwargs)
    loss_r, grads_r = model_loss_and_grads(arch, residual_func, False, **kwargs)
    assert abs(loss_f - loss_r) < 2e-3 * abs(loss_r), (loss_f, loss_r)
    assert grads_f.keys() == grads_r.keys()
    # the controllers' scales and biases are sums over every token of this tiny model: cancellation leaves them a few
    # percent of BF16 noise (flash attention's backward is not even deterministic), so the strict bound covers the
    # tensors with more than a handful of elements
    errs = {n: rel(grads_f[n], grads_r[n]) for n in grads_f}
    worst = max((n for n in errs if grads_f[n].numel() > 32), key=errs.get)
    worst_small = max((n for n in errs if grads_f[n].numel() <= 32), key=errs.get)
    assert errs[worst] < 5e-2, f"{arch} {residual_func}: grad {worst} relative error {errs[worst]:.2e}"
    assert errs[worst_small] < 2.5e-1, f"{arch} {residual_func}: grad {worst_small} relative error {errs[worst_small]:.2e}"
