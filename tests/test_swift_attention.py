import math
import numpy as np
import torch
import torch.nn.functional as F
import fire

from xllm.modules.fused_ops import (
    swift_efficient_attention_fwd,
    swift_efficient_attention_bwd,
)


def manual_swift_attention(
    query, key, value, q_segment_idx, k_segment_idx, n_heads
):
    bsz, qlen, _, _ = query.shape
    kvlen = key.shape[1]

    # B x L1 x L2
    attn_mask = torch.full((bsz, qlen, kvlen), float("-inf"), device=query.device)
    attn_mask = torch.triu(attn_mask, diagonal=(kvlen - qlen + 1)).type_as(query)

    seg_mask = torch.ne(q_segment_idx.unsqueeze(2), k_segment_idx.unsqueeze(1))
    attn_mask = attn_mask.masked_fill(seg_mask, value=float("-inf"))

    # B x H x L x D
    xq = query.transpose(1, 2)
    xk = key.transpose(1, 2)
    xv = value.transpose(1, 2)
    # B x H x L x L
    scores = torch.matmul(xq, xk.transpose(2, 3))
    scores = scores + attn_mask.unsqueeze(1)
    scores = F.softmax(scores, dim=-1, dtype=xq.dtype)
    # B x H x L x S -> B x L x H x S
    output = torch.matmul(scores, xv)
    output = output.transpose(1, 2)
    return output


def test(B: int, L1: int, L2: int, H: int, D: int, dtype: torch.dtype):
    assert L1 <= L2
    bos_ratio = 0.2

    with torch.no_grad():
        query = torch.randn(B, L1, H, D, requires_grad=False, dtype=dtype, device="cuda")
        key = torch.randn(B, L2, H, D, requires_grad=False, dtype=dtype, device="cuda")
        value = torch.randn(B, L2, H, D * 4, requires_grad=False, dtype=dtype, device="cuda")

        bos_mask = torch.rand(B, L2, requires_grad=False, device='cuda') < bos_ratio
        k_segment_idx = torch.cumsum(bos_mask, dim=-1)
        q_segment_idx = k_segment_idx[:, L2 - L1:]

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        value = F.silu(value) + 0.1

    q = query.clone().detach().requires_grad_(True)
    k = key.clone().detach().requires_grad_(True)
    v = value.clone().detach().requires_grad_(True)

    y_mannual = manual_swift_attention(
        q.double(), k.double(), v.double(), q_segment_idx, k_segment_idx, H
    )

    y_mannual_flat = y_mannual.flatten()
    num_elem = y_mannual_flat.shape[0]
    weight = torch.randn(num_elem, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(L1)
    # weight = torch.ones(num_elem, requires_grad=False, dtype=torch.double, device="cuda")
    loss = y_mannual_flat @ weight
    y_mannual.retain_grad()
    loss.backward()
    y_grad = y_mannual.grad.to(dtype)
    q_grad = q.grad
    k_grad = k.grad
    v_grad = v.grad

    with torch.no_grad():
        y_swift, w = swift_efficient_attention_fwd(q, k, v, q_segment_idx, k_segment_idx, 1.0, 0.0, True)

        atol = 1e-6
        rtol = 1e-4
        torch.testing.assert_close(y_swift, y_mannual.to(dtype), rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, dtype={dtype}: pass fwd test")

        q_grad_swift, k_grad_swift, v_grad_swift = swift_efficient_attention_bwd(
            y_grad, q, k, v, w, 1.0, True
        )

        atol = 1e-6
        rtol = 1e-4
        torch.testing.assert_close(q_grad_swift, q_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_grad_swift, k_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(v_grad_swift, v_grad, rtol=rtol, atol=atol)
        print(f"B={B}, L=({L1}, {L2}), H={H}, D={D}, dtype={dtype}: pass bwd test")


def main(seed: int):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L1, L2, H, D in [
        [1, 4, 4, 1, 4],
        [1, 2048, 2048, 4, 256],
        [1, 2048, 4096, 4, 256],
        [2, 2048, 5000, 1, 256],
        [3, 3072, 4096, 4, 128],
        [4, 8192, 8192, 4, 256],
    ]:
        test(B, L1, L2, H, D, torch.float32)


if __name__ == "__main__":
    fire.Fire(main)
