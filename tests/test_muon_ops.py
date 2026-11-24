"""xllm_extension's Muon ops: the fused momentum and update against the same formulas in fp64, each fp32 result
within one ulp of the exact value (fused multiply-adds round once) and each bf16 output within one bf16 ulp;
the batched copy exactly, between the layouts Muon copies."""
import pytest
import torch
import xllm_extension.ops as xllm_ops

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# (count, rows, cols) per tensor, as Muon packs them: one shard, heads, tall blocks, odd sizes
BLOCKS = [(1, 96, 160), (6, 64, 96), (3, 100, 40), (2, 33, 70), (1, 0, 8)]


def views(device):
    """bf16 blocks of packed buffers (muon._pack), the last one 2 bytes off 16-byte alignment (no vector access)."""
    out = [torch.empty(shape, dtype=torch.bfloat16, device=device) for shape in BLOCKS]
    count, rows, cols = BLOCKS[1]
    out.append(torch.empty(count * rows * cols + 1, dtype=torch.bfloat16, device=device)[1:].view(BLOCKS[1]))
    return out


def assert_within_ulp(x, exact, extra=0.0):
    """|x - exact| <= one ulp of x, in x's dtype (fp32: 24 significant bits, bf16: 8), plus `extra`."""
    bits = {torch.float32: 24, torch.bfloat16: 8}[x.dtype]
    tol = torch.ldexp(torch.ones_like(exact), torch.frexp(x.float())[1].double() - bits) + extra
    assert ((x.double() - exact).abs() <= tol).all(), ((x.double() - exact).abs() / tol).max()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_momentum(device, nesterov, grad_dtype):
    torch.manual_seed(0)
    mu = 0.95
    outs = views(device)
    ms = [torch.randn(o.numel(), device=device) for o in outs]
    gs = [torch.randn(o.numel(), device=device).to(grad_dtype) for o in outs]
    exact_ms = [float(torch.tensor(mu, dtype=torch.float32)) * m.double() + g.double() for m, g in zip(ms, gs)]
    xllm_ops.muon_momentum_(ms, gs, outs, mu, nesterov)
    for m, exact, g, o in zip(ms, exact_ms, gs, outs):
        assert_within_ulp(m, exact)
        # the output is the bf16 rounding of an fp32 value (itself an fma of the stored m when nesterov)
        u = float(torch.tensor(mu, dtype=torch.float32)) * m.double() + g.double() if nesterov else m.double()
        assert_within_ulp(o.reshape(-1), u)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lr, wd", [(0.02, 0.1), (3e-4, 0.06)])
def test_update(device, lr, wd):
    torch.manual_seed(0)
    us = views(device)
    for u in us:
        u.copy_(torch.randn(u.shape))
    ps = [torch.randn(u.numel(), device=device) for u in us]
    scales = [0.5 + i for i in range(len(us))]
    f32 = lambda v: float(torch.tensor(v, dtype=torch.float32))  # the kernels take lr, wd and scales in fp32
    inner = [f32(wd) * p.double() + f32(s) * u.reshape(-1).double() for p, u, s in zip(ps, us, scales)]
    exact = [p.double() - f32(lr) * i for p, i in zip(ps, inner)]
    xllm_ops.muon_update_(ps, us, scales, lr, wd)
    for p, e, i in zip(ps, exact, inner):
        # one rounding of p - lr inner, and lr times the fp32 rounding of inner = wd p + s u
        assert_within_ulp(p, e, extra=f32(lr) * i.abs() * 2 ** -24)


def test_many_tensors_split_over_launches():
    """More tensors than one launch's argument list holds (512)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    us = [torch.randn(3, 8, 5, device="cuda").bfloat16() for _ in range(600)]
    ps = [torch.randn(120, device="cuda") for _ in us]
    exact = [p.double() - 0.125 * u.reshape(-1).double() for p, u in zip(ps, us)]
    xllm_ops.muon_update_(ps, us, [1.0] * len(us), 0.125, 0.0)
    for p, e in zip(ps, exact):
        assert_within_ulp(p, e)


def copy_pairs(device, src_dtype, dst_dtype):
    """(dst, src) views of one shape: slices of shards (some columns) and contiguous all-to-all buffers, as Muon
    copies between, plus transposed blocks and unaligned buffers for the other paths."""
    pairs = []
    for count, rows, cols in BLOCKS:
        shape = (count, rows, cols)
        packed = torch.randn(count, min(rows, cols), max(rows, cols), device=device).to(src_dtype)
        src = packed.mT if rows > cols else packed
        flat = torch.empty(count * rows * cols, dtype=dst_dtype, device=device).view(shape)
        wide = torch.empty(count, rows, cols + 7, dtype=dst_dtype, device=device)[..., 3:3 + cols]  # some columns
        tall = torch.empty(count, cols, rows, dtype=dst_dtype, device=device).mT
        unaligned = torch.empty(count * rows * cols + 3, dtype=dst_dtype, device=device)[3:].view(shape)  # contiguous
        contiguous_src = flat.to(src_dtype).copy_(src)
        pairs += [(flat, src), (wide, src), (tall, src), (tall.clone(), contiguous_src), (unaligned, contiguous_src)]
    return pairs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("src_dtype, dst_dtype", [(torch.bfloat16, torch.bfloat16), (torch.float16, torch.bfloat16),
                                                  (torch.bfloat16, torch.float16)])
def test_batched_copy(device, src_dtype, dst_dtype):
    torch.manual_seed(0)
    pairs = copy_pairs(device, src_dtype, dst_dtype)
    xllm_ops.batched_copy_([d for d, _ in pairs], [s for _, s in pairs])
    for d, s in pairs:
        torch.testing.assert_close(d, s.to(dst_dtype), rtol=0, atol=0)


def test_batched_copy_split_over_launches():
    """More copies than one launch's argument list holds (192)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    src = [torch.randn(40, 9, device="cuda").bfloat16().mT for _ in range(300)]
    dst = [torch.empty(9, 40, device="cuda", dtype=torch.bfloat16) for _ in src]
    xllm_ops.batched_copy_(dst, src)
    for d, s in zip(dst, src):
        torch.testing.assert_close(d, s, rtol=0, atol=0)
