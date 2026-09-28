import math
import numpy as np
import torch
import fire

from xllm.modules.fused_ops import rejection_fwd, rejection_bwd
from xllm.modules.rms_norm import group_rms_norm


def manual_rejection(x, h):
    dim = h.shape[-1]
    inv_scale = 1.0 / float(dim)
    # [B, *, S] x [B, *,  S] -> [B, *, 1]
    alpha = torch.sum(x * h, dim=-1, keepdim=True) * inv_scale
    y = torch.addcmul(x, alpha, h, value=-1.0)
    return y


def test(B: int, L: int, H: int, S: int, dtype: str):
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        x = torch.randn(B, L, H * S, requires_grad=False, dtype=pt_dtype, device="cuda")
        h = torch.randn(B, L, H * S, requires_grad=False, dtype=pt_dtype, device="cuda")
        h = group_rms_norm(h, None, H, 1e-6)

        x = x.view(B, L, H, S)
        h = h.view(B, L, H, S)

    x = x.clone().detach().requires_grad_(True)
    h = h.clone().detach().requires_grad_(True)

    y_manual = manual_rejection(x.double(), h.double())
    y_manual_flat = y_manual.flatten()
    num_elem_y = y_manual_flat.shape[0]
    weight_y = torch.randn(num_elem_y, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(L)
    loss = y_manual_flat @ weight_y
    y_manual.retain_grad()
    loss.backward()
    y_grad = y_manual.grad.to(pt_dtype)
    x_grad = x.grad
    h_grad = h.grad

    with torch.no_grad():
        atol = {"fp32": 1e-6, "bf16": 5e-3, "fp16": 1e-3}[dtype]
        rtol = {"fp32": 1e-5, "bf16": 1e-2, "fp16": 1e-3}[dtype]

        y, alpha = rejection_fwd(x, h)
        torch.testing.assert_close(y, y_manual.to(pt_dtype), rtol=rtol, atol=atol)
        print(f"B={B}, L={L}, H={H}, S={S}, dtype={dtype}: pass fwd test")

        atol = {"fp32": 2e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
        rtol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]

        x_rej_grad, h_rej_grad = rejection_bwd(y_grad, x, h, alpha)
        torch.testing.assert_close(x_rej_grad, x_grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(h_rej_grad, h_grad, rtol=rtol, atol=atol)
        print(f"B={B}, L={L}, H={H}, S={S}, dtype={dtype}: pass bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for B, L, H, S in [
        [1, 8, 1, 2],
        [2, 8, 2, 4],
        [5, 128, 4, 128],
        [2, 1024, 4, 256],
        [2, 8192, 4, 128],
        [2, 16384, 4, 64],
        [1, 32768, 8, 64],
    ]:
        test(B, L, H, S, dtype)


if __name__ == "__main__":
    fire.Fire(main)
