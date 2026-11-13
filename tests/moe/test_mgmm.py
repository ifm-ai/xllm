import numpy as np
import torch
from torch import Tensor
import fire

from xllm.modules.fused_ops import (
    multi_group_matmul_fwd,
    multi_group_matmul_bwd,
    multi_group_matmul_dgrad,
    multi_group_matmul_wgrad,
)
from xllm.modules.fused_ops.mgmm.multi_group_matmul import _multi_group_matmul


def manual_mgmm(
    inp1: Tensor,
    inp2: Tensor,
    group_sizes,
    transpose
):
    n_bsz = inp2.shape[0]
    assert len(group_sizes) == n_bsz

    inp1_list = torch.split(inp1, group_sizes, dim=0)
    out_list = []

    for i in range(len(inp1_list)):
        inp2_i = inp2[i].t() if transpose else inp2[i]
        out = torch.mm(inp1_list[i], inp2_i)
        out_list.append(out)

    out = torch.cat(out_list, dim=0)
    return out


def pytorch_mgmm_fwd(
    inp1: Tensor,
    inp2: Tensor,
    group_sizes,
    transpose,
    backend,
):
    return multi_group_matmul_fwd(inp1, inp2, group_sizes, transpose, backend)


def pytorch_mgmm_bwd(
    y_grad: Tensor,
    inp2: Tensor,
    inp1: Tensor,
    group_sizes,
    transpose,
    backend,
):
    return multi_group_matmul_bwd(y_grad, inp2, inp1, group_sizes, transpose, backend)


def test(bsz: int, n_experts: int, d_in: int, d_out: int, transpose: bool, dtype: str):
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    assert bsz > n_experts
    bounds = np.sort(np.random.choice(range(1, bsz), n_experts - 1, replace=True))
    group_sizes = [0 for _ in range(n_experts)]
    prev = 0
    for i in range(n_experts - 1):
        group_sizes[i] = bounds[i] - prev
        prev = bounds[i]
    group_sizes[n_experts - 1] = bsz - prev
    assert all([gs >= 0 for gs in group_sizes])
    assert sum(group_sizes) == bsz

    with torch.no_grad():
        inp1 = torch.randn(bsz, d_in, requires_grad=False, dtype=pt_dtype, device="cuda")
        if transpose:
            inp2 = torch.randn(n_experts, d_out, d_in, requires_grad=False, dtype=pt_dtype, device="cuda")
        else:
            inp2 = torch.randn(n_experts, d_in, d_out, requires_grad=False, dtype=pt_dtype, device="cuda")

    inp1 = inp1.clone().detach().requires_grad_(True)
    inp2 = inp2.clone().detach().requires_grad_(True)

    y_manual = manual_mgmm(inp1.double(), inp2.double(), group_sizes, transpose)
    y_manual_flat = y_manual.flatten()
    num_elem = y_manual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.double, device="cuda") / bsz
    loss = y_manual_flat @ weight
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(pt_dtype)
    inp1_grad = inp1.grad
    inp2_grad = inp2.grad

    # sequential mgmm
    with torch.no_grad():
        atol = {"fp32": 1.5e-3, "bf16": 1e-2, "fp16": 5e-3}[dtype]
        rtol = {"fp32": 1e-4, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        y_seq = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'sequential')
        inp1_grad_seq, inp2_grad_seq = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'sequential')
        torch.testing.assert_close(y_seq, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass sequential fwd test")
        torch.testing.assert_close(inp1_grad_seq, inp1_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(inp2_grad_seq, inp2_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass sequential bwd test")

        # y_nest = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'nested')
        # inp1_grad_nest, inp2_grad_nest = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'nested')
        # torch.testing.assert_close(y_nest, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        # print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass nested fwd test")
        # torch.testing.assert_close(inp1_grad_nest, inp1_grad, rtol=rtol, atol=atol)
        # torch.testing.assert_close(inp2_grad_nest, inp2_grad, rtol=rtol, atol=atol)
        # print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass nested bwd test")
        #
        y_pt = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'torch')
        inp1_grad_pt, inp2_grad_pt = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'torch')
        torch.testing.assert_close(y_pt, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass pytorch fwd test")
        torch.testing.assert_close(inp1_grad_pt, inp1_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(inp2_grad_pt, inp2_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass pytorch bwd test")

        y_te = pytorch_mgmm_fwd(inp1, inp2, group_sizes, transpose, 'te')
        inp1_grad_te, inp2_grad_te = pytorch_mgmm_bwd(y_grad, inp2, inp1, group_sizes, transpose, 'te')
        torch.testing.assert_close(y_te, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass te fwd test")
        torch.testing.assert_close(inp1_grad_te, inp1_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(inp2_grad_te, inp2_grad, rtol=rtol, atol=atol)
        print(f"B={bsz}, N={n_experts}, d=({d_in}, {d_out}), transpose={transpose}, dtype={dtype}: pass te bwd test")


# group sizes for the split checks: random, empty groups at the start, middle and end, one group, all empty
SPLIT_CASES = [
    # (group sizes, d_in, d_out)
    ([3, 2], 4, 5),
    ([0, 5, 0, 8], 7, 10),
    ([4, 0, 0, 9, 0], 16, 16),
    ([40, 0, 33, 0, 51, 0, 60, 0], 32, 64),
    ([0, 0, 0, 77], 64, 32),
    ([128], 64, 48),
    ([0, 0, 0], 32, 16),
    ([0], 16, 16),
    ([300, 0, 211, 517, 0, 101, 0, 400], 256, 512),
    ([1024, 0, 2048, 512, 0, 0, 1536, 977], 1024, 512),
]
SPLIT_BACKENDS = ['te', 'torch', 'sequential', 'nested']


def busy_stream(device):
    # unrelated work queued on the stream, which also reuses freed blocks of the caching allocator
    a = torch.randn(1024, 1024, device=device)
    for _ in range(4):
        a = torch.tanh(a @ a.t() / 32)
    junk = [torch.full((1 << 18,), float('nan'), device=device) for _ in range(8)]
    del junk
    return a.sum()


def equal(name: str, a, b) -> bool:
    if a is None or b is None:
        ok = a is None and b is None
    else:
        ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if not ok:
        print(f"FAIL {name}")
    return ok


def check_mgmm_split(group_sizes, d_in: int, d_out: int, transpose: bool, pt_dtype, backend: str) -> bool:
    """dgrad and wgrad called alone, in either order or with other work between them, must be bitwise equal to
    the backend's joint backward (_multi_group_matmul with a 3rd input, which MultiGroupMatMulFunc and
    multi_group_matmul_bwd use)."""
    n, bsz = len(group_sizes), sum(group_sizes)
    inp1 = torch.randn(bsz, d_in, dtype=pt_dtype, device="cuda")
    inp2 = torch.randn(n, *((d_out, d_in) if transpose else (d_in, d_out)), dtype=pt_dtype, device="cuda")
    out_grad = torch.randn(bsz, d_out, dtype=pt_dtype, device="cuda")
    tag = f"{backend} gs={group_sizes} d=({d_in}, {d_out}) transpose={transpose} {pt_dtype}"
    try:
        ref1, ref2 = _multi_group_matmul(out_grad, inp2, inp1, group_sizes, not transpose, backend)
    except Exception as e:  # a case the backend does not support: the split must not support it either
        try:
            multi_group_matmul_bwd(out_grad, inp2, inp1, group_sizes, transpose, backend)
        except Exception:
            print(f"skip {tag}: joint and split both raise {type(e).__name__}")
            return True
        print(f"FAIL {tag}: joint raises {type(e).__name__} but split does not")
        return False

    ok = True
    g1, g2 = multi_group_matmul_bwd(out_grad, inp2, inp1, group_sizes, transpose, backend)
    ok &= equal(f"{tag}: bwd dgrad", g1, ref1)
    ok &= equal(f"{tag}: bwd wgrad", g2, ref2)
    # wgrad first
    g2 = multi_group_matmul_wgrad(out_grad, inp2, inp1, group_sizes, transpose, backend)
    g1 = multi_group_matmul_dgrad(out_grad, inp2, group_sizes, transpose, backend)
    ok &= equal(f"{tag}: wgrad-first dgrad", g1, ref1)
    ok &= equal(f"{tag}: wgrad-first wgrad", g2, ref2)
    # dgrad, other work on the stream, then wgrad
    g1 = multi_group_matmul_dgrad(out_grad, inp2, group_sizes, transpose, backend)
    other = torch.randn_like(out_grad)
    multi_group_matmul_bwd(other, inp2, inp1, group_sizes, transpose, backend)
    busy_stream(inp1.device)
    del other
    g2 = multi_group_matmul_wgrad(out_grad, inp2, inp1, group_sizes, transpose, backend)
    ok &= equal(f"{tag}: deferred dgrad", g1, ref1)
    ok &= equal(f"{tag}: deferred wgrad", g2, ref2)
    torch.cuda.synchronize()
    return ok


def split(seed: int):
    """Bitwise checks of the split backward, in bf16 and fp32; exits with 1 if any fails."""
    torch.backends.cuda.matmul.allow_tf32 = False
    n_checks, n_fail = 0, 0
    for pt_dtype in [torch.bfloat16, torch.float32]:
        torch.manual_seed(seed)
        for group_sizes, d_in, d_out in SPLIT_CASES:
            for backend in SPLIT_BACKENDS:
                for transpose in [False, True]:
                    n_checks += 1
                    n_fail += not check_mgmm_split(group_sizes, d_in, d_out, transpose, pt_dtype, backend)
        print(f"mgmm split {pt_dtype}: {n_checks} cases, {n_fail} failed")
    print(f"{'FAIL' if n_fail else 'PASS'}: {n_fail} of {n_checks} cases failed")
    if n_fail:
        raise SystemExit(1)


def main(seed: int, dtype: str = 'bf16', split_only: bool = False):
    """--split_only runs only the bitwise checks of the split backward (both dtypes), e.g.
        python tests/moe/test_mgmm.py --seed 0 --split_only
    """
    if split_only:
        return split(seed)
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, N, D1, D2 in [
        [5, 2, 4, 5],
        [13, 3, 7, 10],
        [128, 8, 16, 16],
        [253, 7, 21, 23],
        [512, 8, 32, 64],
        [1024, 16, 256, 1024],
        [2048, 32, 512, 256],
        [2048, 32, 512, 2048],
        [4834, 24, 4096, 14336],
        [8257, 64, 14336, 4096],
        [31456, 64, 4096, 8192]
    ]:
        test(B, N, D1, D2, False, dtype)
        test(B, N, D1, D2, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
