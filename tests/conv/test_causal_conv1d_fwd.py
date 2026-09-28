"""Forward correctness tests for the Triton causal conv1d backend."""

import pytest
import torch
import torch.nn.functional as F

from xllm.modules.causal_conv import CausalConv1d
from xllm.modules.fused_ops import causal_conv1d_fwd


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="causal conv Triton kernels require CUDA",
)


def _causal_conv1d_reference(
    x,
    weight,
    bias=None,
    initial_state=None,
    bos_mask=None,
    activation=None,
):
    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    state = (
        x.new_zeros(batch, width - 1, dim)
        if initial_state is None
        else initial_state
    )
    outputs = []

    for t in range(seqlen):
        if bos_mask is not None:
            state = state.masked_fill(bos_mask[:, t, None, None], 0)
        window = torch.cat((state, x[:, t : t + 1]), dim=1)
        out = (window * weight.t().unsqueeze(0)).sum(dim=1)
        if bias is not None:
            out = out + bias
        if activation in ("silu", "swish"):
            out = F.silu(out)
        outputs.append(out)
        state = window[:, 1:]

    return torch.stack(outputs, dim=1), state


def _assert_close(actual, expected, *, rtol, atol):
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize(
    "use_bias,use_initial_state,use_bos_mask",
    [
        pytest.param(False, False, False, id="plain"),
        pytest.param(True, True, False, id="state"),
        pytest.param(False, False, True, id="segments"),
        pytest.param(True, True, True, id="state-and-segments"),
    ],
)
def test_triton_forward_matches_reference(
    width,
    activation,
    use_bias,
    use_initial_state,
    use_bos_mask,
):
    torch.manual_seed(1234)
    batch, seqlen, dim = 2, 67, 65
    x = torch.randn(batch, seqlen, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = (
        torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        if use_bias
        else None
    )
    initial_state = (
        torch.randn(
            batch,
            width - 1,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if use_initial_state
        else None
    )
    bos_mask = None
    if use_bos_mask:
        bos_mask = torch.zeros(batch, seqlen, device="cuda", dtype=torch.bool)
        bos_mask[0, 0] = True
        bos_mask[0, seqlen // 3] = True
        bos_mask[1, seqlen // 2] = True

    expected_out, expected_final_state = _causal_conv1d_reference(
        x.float(),
        weight.float(),
        bias.float() if bias is not None else None,
        initial_state.float() if initial_state is not None else None,
        bos_mask,
        activation=activation,
    )
    actual_out, actual_final_state, _ = causal_conv1d_fwd(
        x,
        weight,
        bias,
        initial_state,
        bos_mask,
        output_final_state=True,
        activation=activation,
        backend="triton",
    )

    _assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    _assert_close(actual_final_state, expected_final_state, rtol=0, atol=0)


def test_triton_forward_reference_fallback():
    torch.manual_seed(5678)
    batch, seqlen, dim, width = 1, 17, 13, 5
    x = torch.randn(batch, seqlen, dim, device="cuda")
    weight = torch.randn(dim, width, device="cuda")
    bias = torch.randn(dim, device="cuda")
    initial_state = torch.randn(batch, width - 1, dim, device="cuda")
    bos_mask = torch.zeros(batch, seqlen, device="cuda", dtype=torch.bool)
    bos_mask[:, 7] = True

    expected_out, expected_final_state = _causal_conv1d_reference(
        x, weight, bias, initial_state, bos_mask, activation="silu"
    )
    actual_out, actual_final_state, _ = causal_conv1d_fwd(
        x,
        weight,
        bias,
        initial_state,
        bos_mask,
        output_final_state=True,
        activation="silu",
        backend="triton",
    )

    _assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    _assert_close(actual_final_state, expected_final_state, rtol=0, atol=0)


def test_triton_forward_reference_fallback_without_bos():
    torch.manual_seed(2468)
    batch, seqlen, dim, width = 1, 17, 13, 4
    x = torch.randn(batch, seqlen, dim, device="cuda", dtype=torch.float16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.float16)
    bias = torch.randn(dim, device="cuda", dtype=torch.float16)
    initial_state = torch.randn(
        batch, width - 1, dim, device="cuda", dtype=torch.float16
    )

    expected_out, expected_final_state = _causal_conv1d_reference(
        x.float(),
        weight.float(),
        bias.float(),
        initial_state.float(),
        activation="silu",
    )
    actual_out, actual_final_state, _ = causal_conv1d_fwd(
        x,
        weight,
        bias,
        initial_state,
        output_final_state=True,
        activation="silu",
        backend="triton",
    )

    _assert_close(actual_out, expected_out, rtol=2e-3, atol=2e-3)
    _assert_close(actual_final_state, expected_final_state, rtol=0, atol=0)


def test_causal_conv1d_module_with_bias_and_forward():
    torch.manual_seed(1357)
    batch, seqlen, dim, width = 2, 32, 16, 4
    layer = CausalConv1d(
        dim,
        width,
        bias=True,
        activation="silu",
        backend="triton",
    ).to(device="cuda", dtype=torch.bfloat16)
    assert layer.bias.shape == (dim,)

    x = torch.randn(batch, seqlen, dim, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(
        batch, width - 1, dim, device="cuda", dtype=torch.bfloat16
    )
    bos_mask = torch.zeros(batch, seqlen, device="cuda", dtype=torch.bool)
    bos_mask[0, 11] = True
    bos_mask[1, 23] = True

    effective_weight = F.softmax(layer.weight, dim=-1, dtype=torch.float32).to(x)
    expected_out, expected_final_state = _causal_conv1d_reference(
        x.float(),
        effective_weight.float(),
        layer.bias.float(),
        initial_state.float(),
        bos_mask,
        activation="silu",
    )
    actual_out, actual_final_state = layer(
        x,
        initial_state,
        bos_mask,
        output_final_state=True,
    )

    _assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    _assert_close(actual_final_state, expected_final_state, rtol=0, atol=0)


@pytest.mark.parametrize("width,seqlen", [(2, 1), (3, 2), (4, 3), (4, 8), (4, 16), (4, 19), (4, 22)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_triton_forward_is_repeatable(width, seqlen, dtype, use_initial_state):
    torch.manual_seed(7531)
    batch, dim = 2, 65
    x = torch.randn(batch, seqlen, dim, device="cuda", dtype=dtype)
    weight = torch.randn(dim, width, device="cuda", dtype=dtype)
    bias = torch.randn(dim, device="cuda", dtype=dtype)
    initial_state = (
        torch.randn(batch, width - 1, dim, device="cuda", dtype=dtype)
        if use_initial_state else None
    )
    bos_mask = torch.zeros(batch, seqlen, device="cuda", dtype=torch.bool)
    bos_mask[0, 0] = True
    bos_mask[0, -1] = True
    # The second batch exercises the optimized tile branch without any resets.
    args = (x, weight, bias, initial_state, bos_mask, True, "silu", "triton")
    expected = causal_conv1d_fwd(*args)[:2]
    reference = _causal_conv1d_reference(
        x.float(), weight.float(), bias.float(),
        initial_state.float() if initial_state is not None else None,
        bos_mask, "silu",
    )
    _assert_close(expected[0], reference[0], rtol=2e-2, atol=2e-2)
    _assert_close(expected[1], reference[1], rtol=0, atol=0)
    for _ in range(10):
        actual = causal_conv1d_fwd(*args)[:2]
        for actual_tensor, expected_tensor in zip(actual, expected):
            assert torch.equal(
                actual_tensor.contiguous().view(torch.uint8),
                expected_tensor.contiguous().view(torch.uint8),
            )


def test_reference_fallback_without_bos_uses_deterministic_convolution(monkeypatch):
    torch.manual_seed(1359)
    x = torch.randn(2, 17, 13, device="cuda", dtype=torch.float16)
    weight = torch.randn(13, 5, device="cuda", dtype=torch.float16)
    initial_state = torch.randn(2, 4, 13, device="cuda", dtype=torch.float16)

    convolution = torch.ops.aten._convolution
    calls = []

    def check_deterministic(*args, **kwargs):
        assert kwargs["deterministic"] is True
        assert kwargs["benchmark"] is False
        calls.append(True)
        return convolution(*args, **kwargs)

    monkeypatch.setattr(torch.ops.aten, "_convolution", check_deterministic)
    expected = _causal_conv1d_reference(
        x.float(), weight.float(), initial_state=initial_state.float(), activation="silu",
    )
    first = None
    for _ in range(5):
        actual = causal_conv1d_fwd(
            x, weight, initial_state=initial_state,
            output_final_state=True, activation="silu", deterministic=True,
        )[:2]
        _assert_close(actual[0], expected[0], rtol=2e-3, atol=2e-3)
        _assert_close(actual[1], expected[1], rtol=0, atol=0)
        if first is not None:
            for value, previous in zip(actual, first):
                assert torch.equal(
                    value.contiguous().view(torch.uint8),
                    previous.contiguous().view(torch.uint8),
                )
        else:
            first = actual
    assert len(calls) == 5
