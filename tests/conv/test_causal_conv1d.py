import math
import numpy as np

import torch
import torch.nn.functional as F
from xllm.modules.fused_ops import (
    causal_conv1d_fwd,
    causal_conv1d_bwd,
)
from einops import rearrange
import fire


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    activation=None,
):
    """
    x: (batch, seqlen, dim)
    weight: (dim, width)
    bias: (dim)
    initial_states: (batch, width - 1, dim)

    out: (batch, dim, seqlen)
    final_states: (batch, width - 1, dim)
    """
    if activation not in [None, "silu"]:
        raise NotImplementedError("activation must be None or silu")
    seqlen = x.shape[1]
    dim, width = weight.shape
    if initial_states is None:
        x = F.pad(x, (0, 0, width - 1, 0), mode="constant", value=0)
    else:
        x = torch.cat([initial_states, x], dim=1)

    final_states = x[:, 1 - width:]

    x = rearrange(x, 'b l d -> b d l')
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = rearrange(out[..., :seqlen], 'b d l -> b l d')
    out = out if activation is None else F.silu(out)
    return out, final_states


def causal_conv1d_bos_mask_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    bos_mask=None,
    activation=None,
):
    """
    x: (batch, seqlen, dim)
    weight: (dim, width)
    bias: (dim)
    initial_states: (batch, width - 1, dim)
    bos_mask: (batch, seqlen)

    out: (batch, dim, seqlen)
    final_states: (batch, width - 1, dim)
    """
    if activation not in [None, "silu"]:
        raise NotImplementedError("activation must be None or silu")

    bsz, seqlen, dim = x.shape
    width = weight.shape[1]
    if initial_states is None:
        hx = torch.zeros(bsz, dim, width - 1, device=x.device, dtype=x.dtype)
    else:
        hx = initial_states.transpose(1, 2).contiguous()

    y = []
    for t in range(seqlen):
        if bos_mask is not None:
            curr_mask = bos_mask[:, t]
            hx = hx.masked_fill(curr_mask.view(bsz, 1, 1), 0)

        # B x D x 1
        xt = x[:, t].unsqueeze(2)
        # B x D x W
        xx = torch.cat([hx, xt], dim=-1)
        # B x D
        yt = torch.einsum('bdw,dw->bd', xx, weight)
        if bias is not None:
            yt = yt + bias
        # B x D x W-1
        hx = xx[..., 1:]
        y.append(yt)

    # B x L x D
    out = torch.stack(y, dim=1)
    out = out if activation is None else F.silu(out)
    final_states = hx.transpose(1, 2).contiguous()
    return out, final_states


def test(B: int, L: int, D: int, W: int, init_state: bool, activation: str|None, multi_segment: bool, dtype: str):
    bos_ratio = 0.1
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    with torch.no_grad():
        x = torch.randn(B, L, D, requires_grad=False, dtype=pt_dtype, device="cuda")
        init_states = torch.randn(B, W - 1, D, requires_grad=False, dtype=pt_dtype, device="cuda") if init_state else None
        weight = torch.randn(D, W, requires_grad=False, dtype=pt_dtype, device="cuda")
        bias = torch.randn(D, requires_grad=False, dtype=pt_dtype, device="cuda")

    x = x.clone().detach().requires_grad_(True)
    if init_state:
        init_states = init_states.clone().detach().requires_grad_(True)
    bos_mask = (torch.rand(B, L) < bos_ratio).to("cuda") if multi_segment else None
    weight = weight.clone().detach().requires_grad_(True)
    bias = bias.clone().detach().requires_grad_(True)
    if multi_segment:
        y_ref, h_ref = causal_conv1d_bos_mask_ref(
            x.double(), weight.double(), bias.double(), init_states.double() if init_state else None, bos_mask, activation,
        )
    else:
        y_ref, h_ref = causal_conv1d_ref(
            x.double(), weight.double(), bias.double(), init_states.double() if init_state else None, activation
        )

    y_ref_flat = y_ref.flatten()
    num_elem_y = y_ref_flat.shape[0]
    weight_y = torch.randn(num_elem_y, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(B * L)
    h_ref_flat = h_ref.flatten()
    num_elem_h = h_ref_flat.shape[0]
    weight_h = torch.randn(num_elem_h, 1, requires_grad=False, dtype=torch.double, device="cuda") / math.sqrt(B * L)
    loss = y_ref_flat @ weight_y + h_ref_flat @ weight_h
    y_ref.retain_grad()
    h_ref.retain_grad()
    loss.backward()
    y_grad = y_ref.grad.to(pt_dtype)
    h_grad = h_ref.grad.to(pt_dtype)
    x_grad = x.grad
    hx_grad = init_states.grad if init_state else None
    w_grad = weight.grad
    b_grad = bias.grad

    with torch.no_grad():
        for backend in ['fla', 'triton']:
            y_kernel, h_kernel, _ = causal_conv1d_fwd(
                x, weight, bias, init_states, bos_mask, output_final_state=True, activation=activation, backend=backend
            )
            atol = {"fp32": 2e-6, "bf16": 1e-3, "fp16": 1e-4}[dtype]
            rtol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(y_kernel, y_ref.to(pt_dtype), rtol=rtol, atol=atol)
            torch.testing.assert_close(h_kernel, h_ref.to(pt_dtype), rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, D={D}, W={W}, init_state={init_state}, activation={activation}, "
                  f"multiseg={multi_segment}, dtype={dtype} backend={backend}: pass fwd test")

            x_kernel_grad, hx_kernel_grad, w_kernel_grad, b_kernel_grad = causal_conv1d_bwd(
                y_grad, h_grad, x, weight, bias, init_states, bos_mask, activation=activation, backend=backend
            )
            atol = {"fp32": 2e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            rtol = {"fp32": 1e-6, "bf16": 1e-2, "fp16": 1e-3}[dtype]
            torch.testing.assert_close(x_kernel_grad, x_grad, rtol=rtol, atol=atol)
            if init_state:
                torch.testing.assert_close(hx_kernel_grad, hx_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(w_kernel_grad, w_grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(b_kernel_grad, b_grad, rtol=rtol, atol=atol)
            print(f"B={B}, L={L}, D={D}, W={W}, init_state={init_state}, activation={activation}, "
                  f"multiseg={multi_segment}, dtype={dtype} backend={backend}: pass bwd test")


def main(seed: int, dtype: str):
    print(f"Initializing random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for B in [1, 2, 3, 4, 5, 8]:
        for L in [3, 4, 11, 32, 128, 255, 512, 733, 1024, 4096, 8192, 16384, 32768, 65536]:
            for D in [8, 64, 123, 256, 1024]:
                for W in [2, 3, 4, 8]:
                    if B * L * D > 65536 * 1024 * 16:
                        continue
                    for activation in [None, 'silu']:
                        test(B, L, D, W, False, activation,False, dtype)
                        test(B, L, D, W, True, activation, False, dtype)
                        test(B, L, D, W, False, activation, True, dtype)
                        test(B, L, D, W, True, activation, True, dtype)


if __name__ == "__main__":
    fire.Fire(main)
