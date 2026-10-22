"""GPU tests of the fused hyper-connection ops (xllm/modules/fused_ops/hyper_connection.py).

Every op runs on BF16 streams and is compared, forward and all gradients, with the FP64
torch reference evaluated on the same (BF16-rounded) inputs; a model-level test compares
the fused and the reference path of whole Transformer / Gekko models.

    python tests/residual/test_hyper_connection_triton.py
"""

import datetime
import os
import types

import torch

from xllm.modules.fused_ops import hyper_connection as fused
from xllm.modules.residual import hyper_connection as hc

torch.backends.cuda.matmul.allow_tf32 = False


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def close(name, x, ref, tol):
    err = rel(x, ref)
    assert err < tol, f"{name}: relative error {err:.2e} >= {tol:.0e}"
    return err


def leaf(t):
    return t.detach().clone().requires_grad_(True)


def coefficients(B, L, G, tail, layout, fill):
    """FP32 coefficients [B, L, (G,) *tail] in a given memory layout: natural (contiguous), component
    (component-major, as the oHC kernel stores them) or slice (a non-dense slice of a larger tensor)."""
    k = 1
    for t in tail:
        k *= t
    if layout == 'natural':
        c = fill(B, L, G, *tail)
    elif layout == 'component':
        c = fill(B, L, k, G).view(B, L, *tail, G).movedim(-1, 2)
    else:
        c = fill(B, L, G, *tail[:-1], 2 * tail[-1])[..., :tail[-1]]
    return c.squeeze(2) if G == 1 and layout == 'natural' else c


def check_read(N, D, G, B=2, L=37, layout='natural'):
    x = torch.randn(B, L, N, D, device='cuda').bfloat16()
    a = coefficients(B, L, G, (N,), layout, lambda *s: torch.rand(*s, device='cuda'))
    dh = torch.randn(B, L, D, device='cuda').bfloat16()
    xf, af = leaf(x), leaf(a)
    h = fused.hc_read(xf, af)
    h.backward(dh)
    xr, ar = leaf(x.double()), leaf(a.double())
    hr = hc.hc_read(xr, ar)
    hr.backward(dh.double())
    errs = [close('read h', h, hr, 1e-2), close('read dx', xf.grad, xr.grad, 1e-2), close('read da', af.grad, ar.grad, 1e-5)]
    print(f"read   N={N} D={D} G={G} {layout}: " + " ".join(f"{e:.1e}" for e in errs))


def check_write(N, D, G, identity, B=2, L=37, layout='natural'):
    x = torch.randn(B, L, N, D, device='cuda').bfloat16()
    R = None if identity else coefficients(B, L, G, (N, N), layout, lambda *s: torch.randn(*s, device='cuda') * 0.5)
    b = coefficients(B, L, G, (N,), layout, lambda *s: torch.rand(*s, device='cuda') * 2)
    y = torch.randn(B, L, D, device='cuda').bfloat16()
    go = torch.randn(B, L, N, D, device='cuda').bfloat16()
    xf, bf, yf = leaf(x), leaf(b), leaf(y)
    Rf = None if identity else leaf(R)
    out = fused.hc_write(xf, Rf, bf, yf)
    out.backward(go)
    xr, br, yr = leaf(x.double()), leaf(b.double()), leaf(y.double())
    Rr = None if identity else leaf(R.double())
    outr = hc.hc_write(xr, Rr, br, yr)
    outr.backward(go.double())
    errs = [close('write out', out, outr, 1e-2), close('write dx', xf.grad, xr.grad, 1e-2),
            close('write db', bf.grad, br.grad, 1e-5), close('write dy', yf.grad, yr.grad, 1e-2)]
    if not identity:
        errs.append(close('write dR', Rf.grad, Rr.grad, 1e-5))
    print(f"write  N={N} D={D} G={G} {layout} {'R=I' if identity else 'R  '}: " + " ".join(f"{e:.1e}" for e in errs))


def check_logits(N, D, NL, B=2, L=37):
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
    errs = [close('logits', out, outr, 1e-5), close('logits dx', xf.grad, xr.grad, 1e-2),
            close('logits dW', wf.grad, wr.grad, 1e-2)]
    print(f"logits N={N} D={D} NL={NL}: " + " ".join(f"{e:.1e}" for e in errs))


def reference_ohc(logits, scale, bias, rates, alpha):
    """FP64 oHC coefficients with a group axis: a, R, b [B, L, G, ...]."""
    l = logits.unsqueeze(-2)  # [B, L, 1, 16]
    a = torch.sigmoid(scale[0] * l[..., :4] + bias[:, :4])
    b = 2 * torch.sigmoid(scale[1] * l[..., 4:8] + bias[:, 4:8])
    R = hc.quaternion_transport(alpha * l[..., 8:] * rates.view(-1, 1) + bias[:, 8:])
    return a, R, b


def check_ohc_coefficients(G, B=2, L=37):
    logits = torch.randn(B, L, 16, device='cuda')
    scale = torch.randn(2, device='cuda')
    bias = torch.randn(G, 16, device='cuda') * 0.5
    rates = torch.rand(G, device='cuda') + 0.5
    alpha = torch.tensor(0.7, device='cuda')
    da, dR, db = (torch.randn(B, L, G, 4, device='cuda'), torch.randn(B, L, G, 4, 4, device='cuda'),
                  torch.randn(B, L, G, 4, device='cuda'))
    lf, sf, bf = leaf(logits), leaf(scale), leaf(bias)
    a, R, b = fused.ohc_coefficients(lf, sf, bf, rates, alpha)
    ((a * da).sum() + (R * dR).sum() + (b * db).sum()).backward()
    lr, sr, br = leaf(logits.double()), leaf(scale.double()), leaf(bias.double())
    ar, Rr, b_r = reference_ohc(lr, sr, br, rates.double(), alpha.double())
    ((ar * da).sum() + (Rr * dR).sum() + (b_r * db).sum()).backward()
    errs = [close('ohc a', a, ar, 1e-6), close('ohc R', R, Rr, 1e-6), close('ohc b', b, b_r, 1e-6),
            close('ohc dlogits', lf.grad, lr.grad, 1e-5), close('ohc dscale', sf.grad, sr.grad, 1e-5),
            close('ohc dbias', bf.grad, br.grad, 1e-5)]
    print(f"ohc    G={G}: " + " ".join(f"{e:.1e}" for e in errs))


def check_sinkhorn(iterations, B=2, L=37):  # B * L > 8 * SMs * 32 runs the multi-block backward loop
    z = torch.randn(B, L, 4, 4, device='cuda') * 2
    dout = torch.randn(B, L, 4, 4, device='cuda')
    zf = leaf(z)
    out = fused.sinkhorn(zf, iterations)
    out.backward(dout)
    zr = leaf(z.double())
    outr = hc.sinkhorn_knopp(zr, iterations)
    outr.backward(dout.double())
    errs = [close('sinkhorn', out, outr, 1e-5), close('sinkhorn dz', zf.grad, zr.grad, 1e-4)]
    print(f"sinkhorn iters={iterations}: " + " ".join(f"{e:.1e}" for e in errs))


def init_single_process():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29523")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0,
                                         timeout=datetime.timedelta(seconds=600))
    from xllm.distributed import initialize_model_parallel
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
            if '_hc.' in name and not name.endswith('weight'):
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


def check_model(arch, residual_func, **kwargs):
    loss_f, grads_f = model_loss_and_grads(arch, residual_func, True, **kwargs)
    loss_r, grads_r = model_loss_and_grads(arch, residual_func, False, **kwargs)
    assert abs(loss_f - loss_r) < 2e-3 * abs(loss_r), (loss_f, loss_r)
    assert grads_f.keys() == grads_r.keys()
    worst = max(grads_f, key=lambda n: rel(grads_f[n], grads_r[n]))
    err = rel(grads_f[worst], grads_r[worst])
    assert err < 5e-2, f"{arch} {residual_func}: grad {worst} relative error {err:.2e}"
    print(f"model  {arch:11s} {residual_func} {kwargs}: loss {loss_f:.5f} vs {loss_r:.5f}, worst grad {worst} {err:.1e}")


def main():
    torch.manual_seed(0)
    for N, D, G in [(4, 1536, 1), (4, 1536, 16), (4, 1536, 64), (4, 256, 1), (3, 96, 4), (2, 1024, 1),
                    (4, 2560, 4), (4, 4096, 1), (4, 5120, 64)]:
        for layout in ('natural', 'component', 'slice'):
            check_read(N, D, G, layout=layout)
            check_write(N, D, G, identity=False, layout=layout)
            check_write(N, D, G, identity=True, layout=layout)
    check_read(4, 1536, 64, B=4, L=8192, layout='component')  # production-size token count
    check_write(4, 1536, 64, identity=False, B=4, L=8192, layout='component')
    for N, D, NL in [(4, 1536, 16), (4, 1536, 8), (4, 1536, 24), (3, 128, 15), (2, 128, 8)]:
        check_logits(N, D, NL)
    for G in [1, 3, 16, 64]:
        check_ohc_coefficients(G)
    for iterations in [1, 5, 20]:
        check_sinkhorn(iterations)
    check_sinkhorn(20, B=4, L=16384)
    init_single_process()
    for arch in ('transformer', 'gekko'):
        for residual_func in ('ihc', 'mhc', 'ohc'):
            check_model(arch, residual_func)
        check_model(arch, 'ohc', residual_groups=8, residual_freq_base=2.0)
    check_model('transformer', 'mhc', residual_streams=2)  # fused read / write, torch Sinkhorn
    torch.distributed.destroy_process_group()
    print("pass")


if __name__ == '__main__':
    main()
