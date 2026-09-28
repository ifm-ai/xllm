import pytest
import torch
from xllm.modules.moe.permute import (
    fused_y_perm_fwd,
    fused_y_perm_bwd,
)
from xllm.models.fused_blocks.utils import (
    fused_permute_y_bwd_y_grad,
    fused_permute_y_bwd_scores_grad
)


def _baseline_y_perm(sorted_indices, y2, routing_scores, B, L, K):
    inverse_indices = torch.argsort(sorted_indices, stable=True)
    y2 = torch.index_select(y2, 0, inverse_indices)
    return (y2.view(B, L, K, -1) * routing_scores.unsqueeze(3)).sum(dim=2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11, 13, 17, 19, 42])
@pytest.mark.parametrize("B,L,K,D", [
    (1, 3, 3, 17),
    (1, 512, 2, 256),
    (2, 1024, 4, 512),
    (3, 53, 5, 141),
    (3, 1024, 4, 512),
    (4, 2048, 6, 512),
    (8, 4096, 6, 1034),
    (8, 4096, 7, 1024),
    (16, 8192, 8, 2048),
])
def test_fused_y_perm_correctness(dtype, B, L, K, D, seed):
    torch.manual_seed(seed)
    N = B * L * K

    with torch.no_grad():
        sorted_indices = torch.randperm(N, device="cuda")
        y2 = torch.randn(N, D, device="cuda", dtype=dtype)
        routing_scores = torch.randn(B, L, K, device="cuda", dtype=torch.float32).softmax(dim=-1).to(dtype)

        baseline = _baseline_y_perm(sorted_indices, y2.double(), routing_scores.double(), B, L, K).to(dtype)
        result, _ = fused_y_perm_fwd(sorted_indices, y2, routing_scores, B, L, K)

        atol = {torch.bfloat16: 1e-3, torch.float16: 1e-4, torch.float32: 1e-6}[dtype]
        rtol = {torch.bfloat16: 1e-2, torch.float16: 1e-3, torch.float32: 1e-5}[dtype]
        torch.testing.assert_close(result, baseline, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", ['torch', 'triton'])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7, 11, 13, 17, 19, 42])
@pytest.mark.parametrize("B,L,K,D", [
    (1, 3, 3, 17),
    (1, 512, 2, 256),
    (2, 1024, 4, 512),
    (3, 53, 5, 141),
    (3, 1024, 4, 512),
    (4, 2048, 6, 512),
    (8, 4096, 6, 1034),
    (8, 4096, 7, 1024),
    (8, 8192, 8, 1126),
])
def test_fused_y_perm_backward(dtype, B, L, K, D, seed, backend):
    torch.manual_seed(seed)
    N = B * L * K
    sorted_indices = torch.randperm(N, device="cuda")

    # Reference path
    y2_ref = torch.randn(N, D, device="cuda", dtype=dtype, requires_grad=True)
    rs_ref = torch.randn(B, L, K, device="cuda", dtype=torch.float32).softmax(dim=-1).to(dtype)
    rs_ref = rs_ref.clone().detach().requires_grad_(True)
    ref_out = _baseline_y_perm(sorted_indices, y2_ref.double(), rs_ref.double(), B, L, K)
    loss = ref_out.sum()
    ref_out.retain_grad()
    loss.backward()

    out_grad = ref_out.grad.to(dtype)
    y2_grad_ref = y2_ref.grad.to(dtype)
    rs_grad_ref = rs_ref.grad.to(dtype)

    # fused y perm bwd
    with torch.no_grad():
        grad_atol = {torch.bfloat16: 1e-3, torch.float32: 1e-6}[dtype]
        grad_rtol = {torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype]
        score_atol = {torch.bfloat16: 1e-3, torch.float32: 5e-5}[dtype]
        score_rtol = {torch.bfloat16: 1e-2, torch.float32: 1e-4}[dtype]

        grad_y2, inverse_indices = fused_permute_y_bwd_y_grad(out_grad, sorted_indices, rs_ref, B, L, K, backend)
        grad_scores = fused_permute_y_bwd_scores_grad(out_grad, inverse_indices, y2_ref, B, L, K, backend)

        torch.testing.assert_close(grad_y2, y2_grad_ref, rtol=grad_rtol, atol=grad_atol)
        torch.testing.assert_close(grad_scores, rs_grad_ref, rtol=score_rtol, atol=score_atol)

        if backend == 'triton':
            grad_y2, grad_scores = fused_y_perm_bwd(out_grad, inverse_indices, y2_ref, rs_ref, B, L, K)

            torch.testing.assert_close(grad_y2, y2_grad_ref, rtol=grad_rtol, atol=grad_atol)
            torch.testing.assert_close(grad_scores, rs_grad_ref, rtol=score_rtol, atol=score_atol)
