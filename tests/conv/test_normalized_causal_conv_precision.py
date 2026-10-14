"""Real-kernel regression for CP-sensitive normalized convolution gradients."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from xllm.models.fused_blocks.utils import causal_conv1d_bwd, causal_conv1d_fwd
from xllm.modules.causal_conv import CausalConv1d
from xllm.modules.fused_ops.conv.triton.causal_conv1d_reference import (
    causal_conv1d_reference_bwd,
    causal_conv1d_reference_fwd,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('deterministic', [False, True])
@pytest.mark.parametrize('activation', [None, 'silu'])
def test_normalized_conv_preserves_small_gradients_across_splits(
    monkeypatch, deterministic, activation,
):
    implementation = importlib.import_module(
        'xllm.modules.fused_ops.conv.triton.triton_causal_conv1d_bwd'
    )

    def reject_fallback(*args, **kwargs):
        raise AssertionError('mixed FP32 weights/BF16 activations left the Triton path')

    monkeypatch.setattr(implementation, '_reference_backward', reject_fallback)
    torch.manual_seed(901)
    length, dim, width = 32768, 64, 4
    layer = CausalConv1d(dim, width, activation=activation).cuda().bfloat16()
    # Adjacent input values are similar: unnormalized gradients have a large
    # common component, but softmax removes it, leaving small kernel gradients.
    x = (0.25 + 0.005 * torch.randn(1, length, dim, device='cuda')).bfloat16()
    dout = ((0.01 + 0.005 * torch.randn_like(x.float())) / length).bfloat16()
    bos = torch.zeros(1, length, device='cuda', dtype=torch.bool)
    bos[:, [0, 73, length // 2 - 2, length // 2 + 5]] = True
    probabilities = F.softmax(layer.weight, -1, dtype=torch.float32)
    expected_raw = causal_conv1d_reference_bwd(
        dout, None, x, probabilities, bos_mask=bos, activation=activation,
    )[2]
    expected = torch.ops.aten._softmax_backward_data(
        expected_raw, probabilities, -1, torch.float32,
    ).to(layer.weight).float()

    outputs, _ = layer(x, bos_mask=bos, deterministic=deterministic)
    regular = torch.autograd.grad(outputs, layer.weight, dout)[0].float()
    fused_outputs, _, _ = causal_conv1d_fwd(
        x, layer.weight, None, None, bos, False, activation, True, 'triton', deterministic,
    )
    fused = causal_conv1d_bwd(
        dout, None, x, layer.weight, None, None, bos,
        activation, True, 'triton', deterministic,
    )[2].float()
    torch.testing.assert_close(outputs, fused_outputs, rtol=0, atol=0)
    if deterministic:
        torch.testing.assert_close(regular, fused, rtol=0, atol=0)

    # Simulate the four CP pieces with identical inputs/upstream gradients.
    # No collective is needed here; the model test covers actual CP/FSDP2.
    partials = []
    for start in range(0, length, length // 4):
        stop = start + length // 4
        state = None if start == 0 else x[:, start - width + 1:start].clone()
        if state is not None:
            for lag in range(1, width):
                state[:, -lag] *= ~bos[:, start - lag + 1:start].any(dim=1, keepdim=True)
        partials.append(causal_conv1d_bwd(
            dout[:, start:stop].contiguous(), None,
            x[:, start:stop].contiguous(), layer.weight, None, state,
            bos[:, start:stop].contiguous(), activation, True, 'triton', deterministic,
        )[2].float())
    split = torch.stack(partials).sum(0)
    for actual in (regular, fused, split):
        error = (actual - expected).norm()
        # An FP32 reference, not another rounded CP1 gradient, is the oracle.
        assert error <= 0.01 * expected.norm() + 1e-8 * expected.numel() ** 0.5
        assert (actual - expected).abs().max() <= 0.03 * expected.abs().max() + 1e-8


@pytest.mark.parametrize('deterministic', [False, True])
@pytest.mark.parametrize('width', [2, 3, 4])
def test_fp32_conv_weights_keep_fast_path_and_optional_state(
    monkeypatch, deterministic, width,
):
    implementation = importlib.import_module(
        'xllm.modules.fused_ops.conv.triton.triton_causal_conv1d_bwd'
    )

    def reject_fallback(*args, **kwargs):
        raise AssertionError('supported mixed precision call used the fallback')

    monkeypatch.setattr(implementation, '_reference_backward', reject_fallback)
    torch.manual_seed(123)
    layer = CausalConv1d(65, width, bias=True, activation='silu').cuda().bfloat16()
    x = torch.randn(2, 513, 65, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    state = torch.randn(2, width - 1, 65, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    bos = torch.zeros(2, 513, device='cuda', dtype=torch.bool)
    bos[:, [255, 256, 511]] = True
    dout, dfinal = torch.randn_like(x), torch.randn_like(state)
    out, final = layer(x, state, bos, True, deterministic)
    actual = torch.autograd.grad((out, final), (x, state, layer.weight, layer.bias), (dout, dfinal))
    probabilities = F.softmax(layer.weight, -1, dtype=torch.float32)
    ref_out, ref_final = causal_conv1d_reference_fwd(
        x, probabilities, layer.bias, state, bos, 'silu',
    )
    dx, ds, dw, db = causal_conv1d_reference_bwd(
        dout, dfinal, x, probabilities, layer.bias, state, bos, 'silu',
    )
    dw = torch.ops.aten._softmax_backward_data(dw, probabilities, -1, torch.float32).to(layer.weight)
    for result, expected in zip((out, final, *actual), (ref_out, ref_final, dx, ds, dw, db)):
        torch.testing.assert_close(result.float(), expected.float(), rtol=0.03, atol=0.03)
