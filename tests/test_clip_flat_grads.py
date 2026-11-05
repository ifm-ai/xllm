import pytest
import torch

from xllm.distributed.fsdp_patches import _flat_grad_buffers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _views(buf, sizes, pad):
    """Contiguous views of ``buf`` of the given numels, ``pad`` zero elements between them (FSDP2's layout)."""
    out, off = [], 0
    for n in sizes:
        out.append(buf[off:off + n]); off += n + pad
    return out


def test_flat_buffers_norm_and_scale():
    torch.manual_seed(0)
    sizes = [1000, 64, 2560, 7]
    for pad in (0, 3):
        buf = torch.zeros(sum(sizes) + pad * len(sizes), device="cuda", dtype=torch.bfloat16)
        views = _views(buf, sizes, pad)
        for v in views:
            v.copy_(torch.randn(v.shape, device="cuda"))
        buf2 = torch.zeros(500, device="cuda", dtype=torch.bfloat16)
        views2 = _views(buf2, [200, 300], 0)
        views2[0].fill_(2.0); views2[1].fill_(-1.0)
        grads = views + views2
        flats = _flat_grad_buffers(grads)
        assert flats is not None and len(flats) == 2
        ref = torch.stack(torch._foreach_norm(grads, 2, dtype=torch.float32)).square().sum().sqrt()
        got = torch.stack(torch._foreach_norm(flats, 2, dtype=torch.float32)).square().sum().sqrt()
        exact = torch.cat([g.float().flatten() for g in grads]).norm()
        assert torch.allclose(got, ref, rtol=1e-5) and torch.allclose(got, exact, rtol=1e-4)
        torch._foreach_mul_(flats, 0.5)
        assert views2[0][0].item() == 1.0 and views2[1][-1].item() == -0.5
    # a non-contiguous view, an overlap, or a storage mostly outside the views: fall back
    assert _flat_grad_buffers([buf[::2]]) is None
    assert _flat_grad_buffers([buf[0:100], buf[50:150]]) is None
    big = torch.zeros(10000, device="cuda", dtype=torch.bfloat16)
    assert _flat_grad_buffers([big[0:100]]) is None
