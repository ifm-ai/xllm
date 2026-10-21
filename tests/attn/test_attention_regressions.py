"""Focused CUDA regressions for shared attention helpers; no optimizer dependency."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from xllm.models.fused_blocks import utils
from xllm.modules.causal_attention import CausalSoftmaxAttention
from xllm.modules.causal_conv import CausalConv1d
from xllm.modules.fused_ops.conv.triton.causal_conv1d_reference import causal_conv1d_reference_bwd


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('paired', [False, True])
def test_partial_rope_preserves_input(paired):
    torch.manual_seed(17)
    q = torch.randn(1, 16, 4, 128, device='cuda', dtype=torch.bfloat16)
    k = q[:, :, :1].clone()
    originals = (q.clone(), k.clone())
    phase = torch.randn(16, 32, device='cuda')
    frequencies = torch.polar(torch.ones_like(phase), phase)
    for backward in (False, True):
        outputs = utils.apply_ropes(q, k, frequencies, 128, 64, backward) if paired else (
            utils.apply_rope(q, frequencies, 128, 64, backward),
            utils.apply_rope(k, frequencies, 128, 64, backward),
        )
        for actual, saved, rotated in zip((q, k), originals, outputs):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            torch.testing.assert_close(rotated[..., 64:], saved[..., 64:], rtol=0, atol=0)
            assert not torch.equal(rotated[..., :64], saved[..., :64])


def test_packed_xattn_trims_metadata_in_forward_and_backward():
    torch.manual_seed(17)
    q = torch.randn(2, 128, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 256, 1, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    bos = torch.zeros(2, 256, device='cuda', dtype=torch.bool)
    bos[:, [0, 128, 173]] = True
    segments = bos.cumsum(1, dtype=torch.int64)
    tail_bos = bos[:, -128:]
    scale = 128 ** -0.5
    expected = utils.xattn_fwd(q, k[:, -128:], v[:, -128:], scale, tail_bos, segments[:, -128:], 0, True)
    actual = utils.flash_attention_fwd(q, k, v, bos, (segments, 128), scale, 0, 0, 'xattn', True)
    for result, reference in zip(actual[:3], expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    dy = torch.randn_like(actual[0])
    gradients = utils.flash_attention_bwd(dy, actual[1], q, k, v, actual[2], bos, (segments, 128), scale, 0, None, 0, True, 'xattn')
    reference = utils.xattn_bwd(dy, q, k[:, -128:], v[:, -128:], expected[1], expected[2], scale, tail_bos, segments[:, -128:], 0, True)
    attention = CausalSoftmaxAttention(4, 1, 128, 0, scale, 0, 'xattn')
    # Exercise the module with pre-trimmed KV and the original packed metadata.
    output, _ = attention(q.flatten(2), k[:, -128:].flatten(2), v[:, -128:].flatten(2), None, None, bos, (segments, 128))
    torch.testing.assert_close(output, expected[0].flatten(2), rtol=0, atol=0)
    regular = torch.autograd.grad(output, (q, k, v), dy.flatten(2))
    regular = (regular[0], regular[1][:, -128:], regular[2][:, -128:])
    for fused, plain, target in zip(gradients, regular, reference):
        torch.testing.assert_close(fused, target, rtol=0, atol=0)
        torch.testing.assert_close(plain, target, rtol=0, atol=0)


@pytest.mark.parametrize('deterministic', [False, True])
def test_normalized_conv_keeps_small_gradients_in_fp32(monkeypatch, deterministic):
    implementation = importlib.import_module('xllm.modules.fused_ops.conv.triton.triton_causal_conv1d_bwd')

    def reject_fallback(*args, **kwargs):
        raise AssertionError('supported mixed-precision call left the Triton path')

    monkeypatch.setattr(implementation, '_reference_backward', reject_fallback)
    torch.manual_seed(901)
    length, width = 32768, 4
    layer = CausalConv1d(64, width, activation='silu').cuda().bfloat16()
    # Nearly constant inputs expose cancellation in the softmax derivative.
    x = (0.25 + 0.005 * torch.randn(1, length, 64, device='cuda')).bfloat16()
    dy = ((0.01 + 0.005 * torch.randn_like(x.float())) / length).bfloat16()
    bos = torch.zeros(1, length, device='cuda', dtype=torch.bool)
    bos[:, [0, 73, length // 2 - 2, length // 2 + 5]] = True
    probabilities = F.softmax(layer.weight, -1, dtype=torch.float32)
    raw = causal_conv1d_reference_bwd(dy, None, x, probabilities, bos_mask=bos, activation='silu')[2]
    expected = torch.ops.aten._softmax_backward_data(raw, probabilities, -1, torch.float32).to(layer.weight).float()
    output, _ = layer(x, bos_mask=bos, deterministic=deterministic)
    regular = torch.autograd.grad(output, layer.weight, dy)[0].float()
    fused_output, _, _ = utils.causal_conv1d_fwd(x, layer.weight, None, None, bos, False, 'silu', True, 'triton', deterministic)
    fused = utils.causal_conv1d_bwd(dy, None, x, layer.weight, None, None, bos, 'silu', True, 'triton', deterministic)[2].float()
    torch.testing.assert_close(output, fused_output, rtol=0, atol=0)
    partials = []
    for start in range(0, length, length // 4):
        stop = start + length // 4
        state = x[:, start - width + 1:start].clone() if start else None
        if state is not None:
            for lag in range(1, width):
                state[:, -lag] *= ~bos[:, start - lag + 1:start].any(1, keepdim=True)
        partials.append(utils.causal_conv1d_bwd(
            dy[:, start:stop].contiguous(), None, x[:, start:stop].contiguous(),
            layer.weight, None, state, bos[:, start:stop].contiguous(),
            'silu', True, 'triton', deterministic,
        )[2].float())
    for actual in (regular, fused, torch.stack(partials).sum(0)):
        delta = actual - expected
        assert delta.norm() <= 0.01 * expected.norm() + 1e-8 * expected.numel() ** 0.5
        assert delta.abs().max() <= 0.03 * expected.abs().max() + 1e-8
