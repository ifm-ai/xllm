"""GPU tests of the hyper-connection path of the fused training block (TransformerBlockFunction in
xllm/models/fused_blocks/transformer/transformer.py).

The stream ops on hi + lo streams: a split write and its inverse, exact to the hi + lo precision; the two-pass backward
ops (`write_fwd(split, rstd_eps)`, `unwrite_bwd`, `rewrite_bwd`, `read_write_bwd`, `read_da`, `read_proj_bwd`) against the single-op kernels they
replace, the fused dX no further from the FP64 value than the single-op path's. Then whole tiny Transformers with
ihc / ohc: the fused block (`fused_block=True`) against the module path (`fused_block=False`, same Triton stream ops,
BF16 streams rounded at every write), loss and every parameter gradient, with and without the recompute switches, at
a channel count that uses a single kernel tile and at the production D = 1536 (three 512 tiles). The two paths differ
by BF16 rounding of the streams, so every gradient is checked against the module path's own sensitivity to BF16-level
forward noise (`bf16_floor`). A depth sweep checks the controller gradients layer by layer: with the streams recovered
from hi + lo they stay at that floor down to the first layer. The recomputing mode (`residual_ckpt_interval` k > 0:
every k-th layer keeps its input, the block's top layer recomputes the rest in the backward) is checked the same way
at several intervals, blocks of one layer to the whole model, with a partial last block. Last, the activation memory
held between forward and backward: the two branch outputs per layer plus, inverting, the top layer's (hi, lo) streams
and the first layer's embedding copy, or, recomputing, the kept inputs; nothing else of the stream path.

    python -m pytest -q tests/residual/test_hyper_connection_fused_block.py
"""

import datetime
import os
import types

import pytest
import torch

from xllm.config import ModelConf
from xllm.distributed.initialize import initialize_model_parallel, model_parallel_is_initialized
from xllm.models.transformer import Transformer
from xllm.modules.fused_ops import hyper_connection as fused
from xllm.modules.residual.hyper_connection import quaternion_transport
from xllm.utils import create_on_gpu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.fixture(autouse=True)
def _exact_matmul():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32


@pytest.fixture(scope="module")
def single_process():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29523")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0,
                                             timeout=datetime.timedelta(seconds=600))
    if not model_parallel_is_initialized():
        initialize_model_parallel(1, 1, 600)


def rel(x, ref):
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def unwrite(hi, lo, R, b, y, inplace=False):
    """The inverse write alone, through `unwrite_bwd` with zero gates and a zero output gradient (lo None: hi alone)."""
    lo_in = torch.zeros_like(hi) if lo is None else lo
    a0 = torch.zeros(hi.shape[0], hi.shape[1], device=hi.device)
    return fused.unwrite_bwd(hi, lo_in, R, b, y, a0, torch.zeros_like(hi), inplace=inplace)[:2]


# ---------------------------------------------------------------------------------------------
# the stream ops of the fused path


@pytest.mark.parametrize("has_r", [True, False])
def test_split_write_and_its_inverse(has_r, D=1536, T=4096):
    """hi = the singly rounded write, hi + lo = the FP32 write to ~2^-17; the inverse from (hi, lo) returns the input to
    ~1e-5 (the hi + lo precision), from hi alone with the 2^-9 single-rounding error; in place the same."""
    X = torch.randn(T, 4, D, device='cuda').bfloat16()
    X_lo = (torch.randn(T, 4, D, device='cuda') * 2 ** -9 * X.float().abs()).bfloat16()
    y = torch.randn(T, D, device='cuda').bfloat16()
    b = 2 * torch.rand(T, 4, device='cuda')
    R = quaternion_transport(torch.randn(T, 8, device='cuda') * 0.3).contiguous() if has_r else None
    X32 = X.float() + X_lo.float()
    exact = (X32 if R is None else R @ X32) + b.unsqueeze(-1) * y.float().unsqueeze(-2)
    hi, lo, _ = fused.write_fwd(X, R, b, y, lo=X_lo, split=True)
    # hi is the rounded write; torch's matmul sums the 4 terms in another order, so a few ties round the other way
    assert (hi != exact.bfloat16()).float().mean().item() < 1e-3, "hi is the rounded write"
    assert rel(hi.float() + lo.float(), exact) < 3e-5
    assert torch.equal(fused.write_fwd(X, R, b, y, split=True)[0], fused.write_fwd(X, R, b, y)[0]), "lo None = plain write"
    X_c = X.clone()
    out_ip, lo_ip, _ = fused.write_fwd(X_c, R, b, y, inplace=True)
    assert out_ip.data_ptr() == X_c.data_ptr() and lo_ip is None and torch.equal(X_c, fused.write_fwd(X, R, b, y)[0]), \
        "plain write in place"
    xh, xl = unwrite(hi, lo, R, b, y)
    assert rel(xh.float() + xl.float(), X32) < 3e-5
    xh1, xl1 = unwrite(hi, None, R, b, y)
    assert rel(xh1.float() + xl1.float(), X32) < 3e-3
    hi_c, lo_c = hi.clone(), lo.clone()
    xh2, xl2 = unwrite(hi_c, lo_c, R, b, y, inplace=True)
    assert xh2.data_ptr() == hi_c.data_ptr() and torch.equal(xh2, xh) and torch.equal(xl2, xl), "in-place inversion"


@pytest.mark.parametrize("has_r", [True, False])
@pytest.mark.parametrize("N", [4, 3])
def test_two_pass_backward_ops(N, has_r, D=1536, T=4096, NL=16):
    """The fused block's two-pass backward against the single-op kernels it replaces. `unwrite_bwd` = the inverse write
    + read_fwd + write_bwd (same FP32 expressions: hi, lo, h, dy to BF16 ties, db, dR to FP32 rounding); `read_da` =
    read_bwd's da; the split write's rstd = the rstd kernel's, and `logits_fwd` given it = `logits_fwd` alone;
    `write_bwd(need_dx=False)` = write_bwd without dx; `read_proj_bwd` = the FP64 dX = R^T G + a dh + dP W^T - c X
    rounded once, at least as close to the FP64 value as the single-op path (its three shares rounded to BF16 and added
    in BF16). Both in place and fresh."""
    X = torch.randn(T, N, D, device='cuda').bfloat16()
    X_lo = (torch.randn(T, N, D, device='cuda') * 2 ** -9 * X.float().abs()).bfloat16()
    y = torch.randn(T, D, device='cuda').bfloat16()
    b = 2 * torch.rand(T, N, device='cuda')
    a = torch.rand(T, N, device='cuda')
    R = quaternion_transport(torch.randn(T, 8, device='cuda') * 0.3)[:, :N, :N].contiguous() if has_r else None
    G = torch.randn(T, N, D, device='cuda').bfloat16()
    dh = torch.randn(T, D, device='cuda').bfloat16()
    W = (torch.randn(N * D, NL, device='cuda') / (N * D) ** 0.5).bfloat16()
    dl = torch.randn(T, NL, device='cuda')
    # forward: the split write with its rstd, and the projection given that rstd
    hi, lo, _ = fused.write_fwd(X, R, b, y, lo=X_lo, split=True)
    hi2, lo2, rstd_w = fused.write_fwd(X, R, b, y, lo=X_lo, split=True, rstd_eps=1e-6)
    assert torch.equal(hi, hi2) and torch.equal(lo, lo2), "split write unchanged by the rstd"
    logits, rstd = fused.logits_fwd(hi.view(T, N * D), W, 1e-6)
    assert rel(rstd_w, rstd) < 1e-5
    logits_w, rstd_w2 = fused.logits_fwd(hi.view(T, N * D), W, 1e-6, rstd_w)
    assert rstd_w2 is rstd_w and rel(logits_w, logits) < 1e-5
    hi_p, lo_p, rstd_p = fused.write_fwd(X, R, b, y, rstd_eps=1e-6)  # the plain write's rstd (the recomputing mode)
    assert lo_p is None and rel(rstd_p, fused.logits_fwd(hi_p.view(T, N * D), W, 1e-6)[1]) < 1e-5
    # backward pass 1: inverse write + read + write backward
    xh, xl = unwrite(hi, lo, R, b, y)
    h = fused.read_fwd(xh, a)
    dx_w, dR, db, dy = fused.write_bwd(xh, R, b, y, G)
    if dx_w is None:
        dx_w = G  # identity transport: dX = dX'
    dx_n, dR_n, db_n, dy_n = fused.write_bwd(xh, R, b, y, G, need_dx=False)
    assert dx_n is None and torch.equal(db_n, db) and torch.equal(dy_n, dy) and (dR is None or torch.equal(dR_n, dR))
    xh2, xl2, h2, dR2, db2, dy2 = fused.unwrite_bwd(hi, lo, R, b, y, a, G)
    # the FP32 X agrees to FP32 rounding; its FMA contraction differs between the two kernels, so a few elements in a
    # million round to the other BF16 neighbour in hi, which shows in h and dR (sums over hi) at ~1e-5
    assert (xh2 != xh).float().mean().item() < 1e-4
    assert rel(xh2.float() + xl2.float(), xh.float() + xl.float()) < 1e-6
    for got, want in ((h2, h), (dy2, dy), (db2, db)) + (((dR2, dR),) if has_r else ()):
        assert rel(got, want) < 1e-4
    hi_c, lo_c = hi.clone(), lo.clone()
    out = fused.unwrite_bwd(hi_c, lo_c, R, b, y, a, G, inplace=True)
    assert out[0].data_ptr() == hi_c.data_ptr() and torch.equal(out[0], xh2) and torch.equal(out[1], xl2), "in place"
    # the same pass from the layer's input: the write below again (write_fwd on hi alone) + read + write_bwd
    y0 = torch.randn(T, D, device='cuda').bfloat16()
    b0 = 2 * torch.rand(T, N, device='cuda')
    R0 = quaternion_transport(torch.randn(T, 8, device='cuda') * 0.3)[:, :N, :N].contiguous() if has_r else None
    x1_ref = fused.write_fwd(X, R0, b0, y0)[0]
    x1, h3, dR3, db3, dy3 = fused.rewrite_bwd(X, R0, b0, y0, a, R, b, y, G)
    assert (x1 != x1_ref).float().mean().item() < 1e-4, "the write again (FMA contraction may differ)"
    h_ref = fused.read_fwd(x1, a)
    _, dR_ref, db_ref, dy_ref = fused.write_bwd(x1, R, b, y, G, need_dx=False)
    for got, want in ((h3, h_ref), (dy3, dy_ref), (db3, db_ref)) + (((dR3, dR_ref),) if has_r else ()):
        assert rel(got, want) < 1e-5
    # and from the streams themselves: read + write_bwd
    h4, dR4, db4, dy4 = fused.read_write_bwd(xh, a, R, b, y, G)
    for got, want in ((h4, h), (dy4, dy), (db4, db)) + (((dR4, dR),) if has_r else ()):
        assert rel(got, want) < 1e-5
    # da alone
    dx_r, da = fused.read_bwd(xh, a, dh)
    assert rel(fused.read_da(xh, dh), da) < 1e-6
    # backward pass 2: the whole dX
    dP, c = fused.proj_bwd_terms(rstd, logits, dl, N * D, hi.dtype)
    wt = W.t().contiguous()
    dx_c, _, _, _ = fused.logits_bwd(xh.view(T, N * D), W, rstd, logits, dl, need_dw=False)
    Xd, Gd = xh.double(), G.double()
    ref = (Gd if R is None else torch.einsum('tnm,tnd->tmd', R.double(), Gd)) + a.double().unsqueeze(-1) * dh.double().unsqueeze(-2)
    ref = ref + (dP.double() @ wt.double()).view(T, N, D) - c.double()[:, None, None] * Xd
    dx = fused.read_proj_bwd(xh, R, a, G, dh, dP, wt, c)
    old = dx_r.clone().add_(dx_w).add_(dx_c.view(T, N, D))  # the single-op path: BF16 shares added in BF16
    err_new, err_old = rel(dx, ref), rel(old, ref)
    assert err_new < 2e-3 and err_new <= err_old, (err_new, err_old)
    G_c = G.clone()
    dx2 = fused.read_proj_bwd(xh, R, a, G_c, dh, dP, wt, c, out=G_c)
    assert dx2.data_ptr() == G_c.data_ptr() and torch.equal(dx2, dx), "in place over G"


# ---------------------------------------------------------------------------------------------
# whole models: the fused block against the module path


def build(residual_func, fused_block, model_dim, num_layers, **kwargs):
    heads = model_dim // 32
    cfg = ModelConf(arch='transformer', num_layers=num_layers, model_dim=model_dim, num_heads=heads,
                    num_kv_heads=max(1, heads // 2), head_dim=32, chunk_size=32, ffn_hidden_dim=2 * model_dim, swiglu=True,
                    apply_rmsnorm=True, vocab_size=256, causal_attn_backend='flash', residual_func=residual_func,
                    fused_block=fused_block, **kwargs)
    torch.manual_seed(0)
    with create_on_gpu():
        model = Transformer(cfg, types.SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    model = model.cuda().to(torch.bfloat16).train()
    with torch.no_grad():  # move the controllers off their near-identity init so every coefficient matters
        for name, p in model.named_parameters():
            if '.residual.' in name and not name.endswith('weight'):
                p.add_(torch.randn_like(p) * 0.5)
    return model


def loss_and_grads(model, B=2, L=64, noise=0.0):
    """`noise` > 0 perturbs the embedding by that relative amount (the module path's BF16 sensitivity, see `bf16_floor`)."""
    g = torch.Generator(device='cuda').manual_seed(1)
    tokens = torch.randint(3, 256, (B, L), device='cuda', generator=g)
    if noise:
        def hook(mod, inp, out):
            return (out.float() * (1 + noise * torch.randn(out.shape, device=out.device, generator=g))).to(out.dtype)
        model.embed.register_forward_hook(hook)
    loss, _, _ = model(tokens, multi_segments=False, targets=torch.roll(tokens, -1, dims=1))
    loss = loss.float().mean()
    loss.backward()
    return loss.item(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}


def bf16_floor(residual_func, model_dim, num_layers, grads_r):
    """How far every gradient of the module path moves under BF16-level forward noise: the module path rerun with the
    embedding perturbed by 2^-9 * sqrt(2 L) relatively (one BF16 rounding per write, 2 writes per layer). The fused block
    carries hi + lo instead of rounding at every write, so its gradients differ from the module path's by noise of this
    size; a gradient with heavy token cancellation (e.g. the first layer's FFN controller weight at 128 tokens) moves
    by 10% under it while the typical one moves by 1-2%. Both paths are deterministic here (rerunning either gives
    bitwise equal gradients), so this is the whole floor of the comparison."""
    _, grads_n = loss_and_grads(build(residual_func, False, model_dim, num_layers), noise=2 ** -9 * (2 * num_layers) ** 0.5)
    return {n: rel(grads_n[n], grads_r[n]) for n in grads_r}


def bound(floor):
    return max(5e-2, 3 * floor)


PRODUCTION = dict(recompute_q=True, recompute_fc1_out=True, recompute_fc3_out=True)


@pytest.mark.parametrize("model_dim,num_layers,kwargs", [
    (128, 2, {}), (128, 2, PRODUCTION), (128, 2, dict(recompute_v=True, **PRODUCTION)),
    (1536, 1, PRODUCTION), (1536, 3, PRODUCTION),
])
@pytest.mark.parametrize("residual_func", ['ihc', 'ohc'])
def test_fused_block_matches_module_path(single_process, residual_func, model_dim, num_layers, kwargs):
    """Loss and every gradient of the fused block vs the module path. The main weights (attention, FFN, norms) and the
    controller projections must be within 5e-2 or within 3x their BF16 floor (`bf16_floor`): the two paths differ in the
    streams' carry (hi + lo across layers vs BF16 rounding at every write), so they agree to the BF16 level, not
    bitwise. The controllers' scales / biases are token sums with cancellation and are bounded loosely."""
    loss_f, grads_f = loss_and_grads(build(residual_func, True, model_dim, num_layers, **kwargs))
    loss_r, grads_r = loss_and_grads(build(residual_func, False, model_dim, num_layers))
    assert abs(loss_f - loss_r) < 2e-3 * abs(loss_r), (loss_f, loss_r)
    assert grads_f.keys() == grads_r.keys(), set(grads_f) ^ set(grads_r)
    errs = {n: rel(grads_f[n], grads_r[n]) for n in grads_f}
    floor = bf16_floor(residual_func, model_dim, num_layers, grads_r)
    for n in errs:
        if '.residual.' not in n or n.endswith('weight'):
            assert errs[n] < bound(floor[n]), f"grad {n} relative error {errs[n]:.2e}, BF16 floor {floor[n]:.2e}"
        else:
            assert errs[n] < 2.5e-1, f"grad {n} relative error {errs[n]:.2e}"


@pytest.mark.parametrize("ckpt", [1, 2, 3])
@pytest.mark.parametrize("residual_func", ['ihc', 'ohc'])
def test_recomputing_mode_matches_module_path(single_process, residual_func, ckpt, model_dim=1536, num_layers=3):
    """`residual_ckpt_interval` 1 (every layer kept), 2 (a block of two and a partial one), 3 (one block, the whole
    model recomputed from the embedding): the same bounds as the inverting mode."""
    loss_f, grads_f = loss_and_grads(build(residual_func, True, model_dim, num_layers, residual_ckpt_interval=ckpt, **PRODUCTION))
    loss_r, grads_r = loss_and_grads(build(residual_func, False, model_dim, num_layers))
    assert abs(loss_f - loss_r) < 2e-3 * abs(loss_r), (loss_f, loss_r)
    errs = {n: rel(grads_f[n], grads_r[n]) for n in grads_f}
    floor = bf16_floor(residual_func, model_dim, num_layers, grads_r)
    for n in errs:
        if '.residual.' not in n or n.endswith('weight'):
            assert errs[n] < bound(floor[n]), f"grad {n} relative error {errs[n]:.2e}, BF16 floor {floor[n]:.2e}"
        else:
            assert errs[n] < 2.5e-1, f"grad {n} relative error {errs[n]:.2e}"


@pytest.mark.parametrize("ckpt", [0, 5])
@pytest.mark.parametrize("residual_func", ['ihc', 'ohc'])
def test_depth_sweep(single_process, residual_func, ckpt, model_dim=256, num_layers=28):
    """The main-weight and controller-weight gradients at every depth stay at the BF16 floor when the streams are
    recovered from hi + lo (with singly rounded streams the first layers drifted to 10-20%), and when they are
    recomputed in blocks of five layers (the last block of three)."""
    _, grads_r = loss_and_grads(build(residual_func, False, model_dim, num_layers))
    _, grads_f = loss_and_grads(build(residual_func, True, model_dim, num_layers, residual_ckpt_interval=ckpt))
    errs = {n: rel(grads_f[n], grads_r[n]) for n in grads_f}
    floor = bf16_floor(residual_func, model_dim, num_layers, grads_r)
    for n in errs:
        if '.residual.' not in n or n.endswith('weight'):
            assert errs[n] < bound(floor[n]), f"grad {n} relative error {errs[n]:.2e}, BF16 floor {floor[n]:.2e}"


@pytest.mark.parametrize("ckpt", [0, 1, 2])
@pytest.mark.parametrize("residual_func", ['ihc', 'ohc'])
def test_held_activations(single_process, residual_func, ckpt, model_dim=1536, num_layers=4, B=2, L=2048):
    """Activation memory held between forward and backward: the base fused block holds x and xo (2 D per token per
    layer), the hyper-connection block y1 and y2 (2 D) plus, inverting, the top layer's (hi, lo) streams (2 x 4 D) and
    the first layer's embedding copy (D); recomputing every k-th layer, the embedding copy (D) and the other kept
    layers' streams (4 D each)."""
    def held(func, **kwargs):
        model = build(func, True, model_dim, num_layers, **PRODUCTION, **kwargs)
        tokens = torch.randint(3, 256, (B, L), device='cuda')
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        loss, _, _ = model(tokens, multi_segments=False, targets=torch.roll(tokens, -1, dims=1))
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        loss.float().mean().backward()
        del model, loss
        torch.cuda.empty_cache()
        return after - before
    extra = held(residual_func, residual_ckpt_interval=ckpt) - held('base')
    streams = 2 * 4 + 1 if ckpt == 0 else 1 + 4 * (-(-num_layers // ckpt) - 1)
    expected = B * L * streams * model_dim * 2
    assert abs(extra - expected) < 0.15 * expected + 24 * 2**20, (extra, expected)
