"""fp8 all-gather extension (xllm/modules/fused_ops/fp8_fsdp.py): shard quantisation against the full weight, the wrapper
semantics FSDP2 relies on, and the fp8 GEMMs on all-gathered weights against the quantised bf16 weights."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                                reason="needs a Hopper GPU")


def _mods():
    try:
        from xllm.modules.fused_ops import fp8_fsdp, fp8_sm90, mxfp8
    except ImportError as e:
        pytest.skip(f"xllm not importable: {e}")
    return fp8_fsdp, fp8_sm90, mxfp8


def _gathered(fp8_fsdp, w, world):
    """What FSDP2 does with the extension: pre_all_gather on each row shard, concatenation, post_all_gather."""
    parts = [fp8_fsdp.Fp8ShardParam(s.contiguous()).fsdp_pre_all_gather(None, None, None, None, None)[0]
             for s in w.chunk(world, 0)]
    data = torch.cat([p[0] for p in parts], 0)
    scale = torch.cat([p[1] for p in parts], 0)
    out, inner = fp8_fsdp.Fp8ShardParam(w[:1]).fsdp_post_all_gather((data, scale), None, torch.bfloat16)
    return out, inner


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shards_match_full_weight(dtype):
    fp8_fsdp, f8, _ = _mods()
    torch.manual_seed(0)
    w = (torch.randn(2048, 1536, device="cuda") * 0.02).to(dtype)
    out, _ = _gathered(fp8_fsdp, w, 8)
    full, _ = f8.quantize(w.bfloat16(), colwise=False, kind="weight")
    assert isinstance(out, fp8_fsdp.Fp8Weight) and out.shape == w.shape and out.dtype == torch.bfloat16
    assert torch.equal(out._data.view(torch.uint8), full.data.view(torch.uint8)) and torch.equal(out._scale, full.scale)
    assert torch.equal(out.dequantize(), f8.dequantize(full).bfloat16())


def test_wrapper_ops():
    fp8_fsdp, _, _ = _mods()
    t = torch.randn(512, 256, device="cuda")
    p = fp8_fsdp.Fp8ShardParam(t.clone())
    assert isinstance(p.view(-1), fp8_fsdp.Fp8ShardParam) and isinstance(p[128:256], fp8_fsdp.Fp8ShardParam)
    assert isinstance(p.new_zeros(4, 4), fp8_fsdp.Fp8ShardParam)
    s = p.chunk(2, 0)
    assert all(isinstance(c, fp8_fsdp.Fp8ShardParam) for c in s)
    q = p.add(1)  # other ops: plain results
    assert type(q) is torch.Tensor and torch.equal(q, t + 1)
    r = p.copy_(t * 2)
    assert r is p and torch.equal(p._t, t * 2)
    torch._foreach_mul_([p._t], 0.5)  # what the optimizer does (on the wrapped tensor)
    assert torch.equal(p._t, t)


def test_linear_set_on_gathered_weights():
    fp8_fsdp, f8, mx = _mods()
    torch.manual_seed(0)
    M, D = 2048, 1536
    ws = [torch.randn(n, D, device="cuda", dtype=torch.bfloat16) * 0.02 for n in (2048, 1024, 1024, 2048)]
    gathered = [_gathered(fp8_fsdp, w, 8)[0] for w in ws]
    ref, got = mx.MxLinearSet(ws), mx.MxLinearSet(gathered)
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    xr, xc = mx.mx_quantize(x)
    for i in range(len(ws)):
        assert torch.equal(got.forward(i, xr), ref.forward(i, xr))
    dys = [torch.randn(M, w.shape[0], device="cuda", dtype=torch.bfloat16) for w in ws]
    dx_ref, dyc = ref.dgrad(dys)
    dx_got, _ = got.dgrad(dys)
    assert torch.equal(dx_got, dx_ref)
    for a, b in zip(got.wgrad(dyc, xc), ref.wgrad(dyc, xc)):
        assert torch.equal(a, b)
    # a non-fp8 op on the gathered weight dequantises it
    assert torch.equal(torch.nn.functional.linear(x, gathered[1]), torch.nn.functional.linear(x, gathered[1].dequantize()))


def test_shard_quantisation_cached_until_update():
    fp8_fsdp, _, _ = _mods()
    p = fp8_fsdp.Fp8ShardParam(torch.randn(256, 512, device="cuda") * 0.02)
    a = p.fsdp_pre_all_gather(None, None, None, None, None)[0]
    b = p.fsdp_pre_all_gather(None, None, None, None, None)[0]
    assert a[0] is b[0]  # unchanged shard: the cached operands
    torch._foreach_mul_([p._t], 2.0)  # an in-place op that bumps the version counter
    c = p.fsdp_pre_all_gather(None, None, None, None, None)[0]
    assert c[0] is not a[0] and torch.equal(c[1], a[1] * 2)
    # torch._fused_adamw_ does not bump version counters: an optimizer step invalidates through the step hook
    fp8_fsdp._register_step_hook()
    q = torch.nn.Parameter(p._t.clone())
    opt = torch.optim.AdamW([q], lr=1e-2, fused=True)
    q.grad = torch.ones_like(q)
    torch._foreach_copy_([p._t], [q.detach()])
    d = p.fsdp_pre_all_gather(None, None, None, None, None)[0]
    st, v = fp8_fsdp._STEP[0], p._t._version
    torch._fused_adamw_([p._t], [q.grad], [torch.zeros_like(q)], [torch.zeros_like(q)], [], [torch.ones((), device="cuda")],
                        amsgrad=False, lr=1e-2, beta1=0.9, beta2=0.95, weight_decay=0.0, eps=1e-8, maximize=False)
    assert p._t._version == v  # the reason for the hook
    opt.step()
    assert fp8_fsdp._STEP[0] == st + 1
    e = p.fsdp_pre_all_gather(None, None, None, None, None)[0]
    assert e[0] is not d[0]
