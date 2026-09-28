"""Backward correctness tests for the Triton causal conv1d backend."""

import pytest
import torch
import torch.nn.functional as F

from xllm.modules.causal_conv import CausalConv1d
from xllm.modules.fused_ops import causal_conv1d, causal_conv1d_bwd
from xllm.modules.fused_ops.conv.triton.causal_conv1d_reference import causal_conv1d_reference_bwd


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


def _bf16_tensor(*shape):
    return (0.5 * torch.randn(*shape, device="cuda")).to(torch.bfloat16)


def _assert_close(actual, expected, *, rtol=2e-2, atol=2e-2):
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("batch,seqlen,dim", [(2, 128, 256), (2, 256, 64)])
def test_triton_backward_matches_autograd_reference(batch, seqlen, dim):
    torch.manual_seed(1234)
    width = 4
    x = _bf16_tensor(batch, seqlen, dim)
    weight = _bf16_tensor(dim, width)
    bias = _bf16_tensor(dim)
    initial_state = _bf16_tensor(batch, width - 1, dim)
    out_grad = _bf16_tensor(batch, seqlen, dim)
    final_state_grad = _bf16_tensor(batch, width - 1, dim)
    bos_mask = torch.zeros(batch, seqlen, dtype=torch.bool, device="cuda")
    bos_mask[0, 0] = True
    bos_mask[0, seqlen // 3] = True
    bos_mask[1, seqlen // 2] = True
    bos_mask[1, seqlen - 2] = True

    ref_x = x.float().requires_grad_()
    ref_weight = weight.float().requires_grad_()
    ref_bias = bias.float().requires_grad_()
    ref_initial_state = initial_state.float().requires_grad_()
    ref_out, ref_final_state = _causal_conv1d_reference(
        ref_x,
        ref_weight,
        ref_bias,
        ref_initial_state,
        bos_mask,
        activation="silu",
    )
    expected = torch.autograd.grad(
        (ref_out, ref_final_state),
        (ref_x, ref_weight, ref_bias, ref_initial_state),
        (out_grad.float(), final_state_grad.float()),
    )

    actual_x = x.clone().requires_grad_()
    actual_weight = weight.clone().requires_grad_()
    actual_bias = bias.clone().requires_grad_()
    actual_initial_state = initial_state.clone().requires_grad_()
    actual_out, actual_final_state = causal_conv1d(
        actual_x,
        actual_weight,
        actual_bias,
        actual_initial_state,
        bos_mask,
        True,
        "silu",
        "triton",
    )
    autograd_grads = torch.autograd.grad(
        (actual_out, actual_final_state),
        (actual_x, actual_weight, actual_bias, actual_initial_state),
        (out_grad, final_state_grad),
    )
    for actual, reference in zip(autograd_grads, expected):
        _assert_close(actual, reference)

    direct_grads = causal_conv1d_bwd(
        out_grad,
        final_state_grad,
        x,
        weight,
        bias,
        initial_state,
        bos_mask,
        activation="silu",
        backend="triton",
    )
    direct_in_autograd_order = (
        direct_grads[0],
        direct_grads[2],
        direct_grads[3],
        direct_grads[1],
    )
    _assert_identical_grads(direct_in_autograd_order, autograd_grads)


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(
    "use_bias,use_initial_state,use_bos_mask,use_final_state_grad",
    [
        pytest.param(False, False, False, False, id="plain"),
        pytest.param(True, True, False, True, id="state"),
        pytest.param(False, True, True, False, id="state-and-segments"),
        pytest.param(True, False, True, True, id="segments"),
        pytest.param(True, True, True, True, id="all-features"),
    ],
)
def test_triton_backward_contract_matrix(
    width,
    activation,
    deterministic,
    use_bias,
    use_initial_state,
    use_bos_mask,
    use_final_state_grad,
):
    torch.manual_seed(2468)
    batch, seqlen, dim = 2, 37, 33
    x = _bf16_tensor(batch, seqlen, dim)
    weight = _bf16_tensor(dim, width)
    bias = _bf16_tensor(dim) if use_bias else None
    initial_state = (
        _bf16_tensor(batch, width - 1, dim)
        if use_initial_state
        else None
    )
    scale = seqlen**-0.5
    out_grad = _bf16_tensor(batch, seqlen, dim) * scale
    final_state_grad = (
        _bf16_tensor(batch, width - 1, dim) * scale
        if use_final_state_grad
        else None
    )
    bos_mask = None
    if use_bos_mask:
        bos_mask = torch.zeros(
            batch,
            seqlen,
            dtype=torch.bool,
            device="cuda",
        )
        bos_mask[0, 0] = True
        bos_mask[0, seqlen // 3] = True
        bos_mask[1, seqlen // 2] = True

    ref_x = x.float().requires_grad_()
    ref_weight = weight.float().requires_grad_()
    ref_bias = bias.float().requires_grad_() if bias is not None else None
    ref_initial_state = (
        initial_state.float().requires_grad_()
        if initial_state is not None
        else None
    )
    ref_out, ref_final_state = _causal_conv1d_reference(
        ref_x,
        ref_weight,
        ref_bias,
        ref_initial_state,
        bos_mask,
        activation,
    )

    reference_inputs = [ref_x]
    if ref_initial_state is not None:
        reference_inputs.append(ref_initial_state)
    reference_inputs.append(ref_weight)
    if ref_bias is not None:
        reference_inputs.append(ref_bias)

    if final_state_grad is None:
        expected = torch.autograd.grad(
            ref_out,
            reference_inputs,
            out_grad.float(),
        )
    else:
        expected = torch.autograd.grad(
            (ref_out, ref_final_state),
            reference_inputs,
            (out_grad.float(), final_state_grad.float()),
        )

    expected_iter = iter(expected)
    expected_x_grad = next(expected_iter)
    expected_initial_state_grad = (
        next(expected_iter) if initial_state is not None else None
    )
    expected_weight_grad = next(expected_iter)
    expected_bias_grad = next(expected_iter) if bias is not None else None

    actual = causal_conv1d_bwd(
        out_grad,
        final_state_grad,
        x,
        weight,
        bias,
        initial_state,
        bos_mask,
        activation=activation,
        backend="triton",
        deterministic=deterministic,
    )
    _assert_close(actual[0], expected_x_grad, rtol=3e-2, atol=3e-2)
    _assert_close(actual[2], expected_weight_grad, rtol=3e-2, atol=3e-2)
    if expected_initial_state_grad is None:
        assert actual[1] is None
    else:
        _assert_close(
            actual[1],
            expected_initial_state_grad,
            rtol=3e-2,
            atol=3e-2,
        )
    if expected_bias_grad is None:
        assert actual[3] is None
    else:
        _assert_close(actual[3], expected_bias_grad, rtol=3e-2, atol=3e-2)


def test_backward_without_bias_or_final_grad_uses_triton(monkeypatch):
    import importlib

    implementation = importlib.import_module(
        "xllm.modules.fused_ops.conv.triton.triton_causal_conv1d_bwd"
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("supported BF16 call used the PyTorch fallback")

    monkeypatch.setattr(implementation, "_reference_backward", fail_if_called)

    batch, seqlen, dim, width = 1, 64, 64, 4
    x = _bf16_tensor(batch, seqlen, dim).requires_grad_()
    weight = _bf16_tensor(dim, width).requires_grad_()
    out_grad = _bf16_tensor(batch, seqlen, dim)
    bos_mask = torch.zeros(batch, seqlen, dtype=torch.bool, device="cuda")
    bos_mask[:, seqlen // 2] = True

    out, final_state = causal_conv1d(
        x,
        weight,
        None,
        None,
        bos_mask,
        False,
        "silu",
        "triton",
    )
    assert final_state is None
    x_grad, weight_grad = torch.autograd.grad(
        out,
        (x, weight),
        out_grad,
    )
    assert x_grad.shape == x.shape
    assert weight_grad.shape == weight.shape


def test_triton_backward_reference_fallback():
    torch.manual_seed(4321)
    batch, seqlen, dim, width = 2, 7, 5, 3
    x = torch.randn(batch, seqlen, dim, device="cuda")
    weight = torch.randn(dim, width, device="cuda")
    out_grad = torch.randn_like(x)

    ref_x = x.clone().requires_grad_()
    ref_weight = weight.clone().requires_grad_()
    ref_out, _ = _causal_conv1d_reference(ref_x, ref_weight)
    expected_x, expected_weight = torch.autograd.grad(
        ref_out,
        (ref_x, ref_weight),
        out_grad,
    )

    x_grad, initial_state_grad, weight_grad, bias_grad = causal_conv1d_bwd(
        out_grad,
        None,
        x,
        weight,
        backend="triton",
    )
    torch.testing.assert_close(x_grad, expected_x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight_grad, expected_weight, rtol=1e-5, atol=1e-5)
    assert initial_state_grad is None
    assert bias_grad is None


def test_triton_backward_reference_fallback_with_state_and_resets():
    torch.manual_seed(8765)
    batch, seqlen, dim, width = 2, 9, 7, 3
    x = torch.randn(batch, seqlen, dim, device="cuda")
    weight = torch.randn(dim, width, device="cuda")
    bias = torch.randn(dim, device="cuda")
    initial_state = torch.randn(batch, width - 1, dim, device="cuda")
    out_grad = torch.randn_like(x)
    final_state_grad = torch.randn_like(initial_state)
    bos_mask = torch.zeros(batch, seqlen, dtype=torch.bool, device="cuda")
    bos_mask[0, 0] = True
    bos_mask[0, 4] = True
    bos_mask[1, 7] = True

    ref_x = x.clone().requires_grad_()
    ref_weight = weight.clone().requires_grad_()
    ref_bias = bias.clone().requires_grad_()
    ref_initial_state = initial_state.clone().requires_grad_()
    ref_out, ref_final_state = _causal_conv1d_reference(
        ref_x,
        ref_weight,
        ref_bias,
        ref_initial_state,
        bos_mask,
        activation="silu",
    )
    expected = torch.autograd.grad(
        (ref_out, ref_final_state),
        (ref_x, ref_initial_state, ref_weight, ref_bias),
        (out_grad, final_state_grad),
    )

    actual = causal_conv1d_bwd(
        out_grad,
        final_state_grad,
        x,
        weight,
        bias,
        initial_state,
        bos_mask,
        activation="silu",
        backend="triton",
    )
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=1e-5,
            atol=1e-5,
        )


def _assert_identical_grads(actual, expected):
    assert len(actual) == len(expected)
    for actual_grad, expected_grad in zip(actual, expected):
        if expected_grad is None:
            assert actual_grad is None
        else:
            assert actual_grad.dtype == expected_grad.dtype
            assert actual_grad.shape == expected_grad.shape
            assert torch.equal(
                actual_grad.contiguous().view(torch.uint8),
                expected_grad.contiguous().view(torch.uint8),
            )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("use_optional_inputs", [False, True])
def test_triton_backward_is_repeatable(width, activation, use_optional_inputs):
    torch.manual_seed(9753)
    # Include a partial chunk and more than 32 partials in the final reduction.
    batch, seqlen, dim = 3, 3073, 65
    x = _bf16_tensor(batch, seqlen, dim)
    weight = _bf16_tensor(dim, width)
    bias = _bf16_tensor(dim) if use_optional_inputs else None
    initial_state = _bf16_tensor(batch, width - 1, dim) if use_optional_inputs else None
    out_grad = _bf16_tensor(batch, seqlen, dim) * seqlen**-0.5
    final_grad = _bf16_tensor(batch, width - 1, dim) if use_optional_inputs else None
    bos_mask = None
    if use_optional_inputs:
        bos_mask = torch.zeros(batch, seqlen, device="cuda", dtype=torch.bool)
        bos_mask[0, 0] = True
        bos_mask[:, [255, 256, 511, seqlen - 2]] = True
    args = (out_grad, final_grad, x, weight, bias, initial_state, bos_mask)
    snapshots = tuple(value.clone() if value is not None else None for value in args)
    expected = causal_conv1d_bwd(*args, activation=activation, deterministic=True)
    reference = causal_conv1d_reference_bwd(
        out_grad.float(),
        final_grad.float() if final_grad is not None else None,
        x.float(),
        weight.float(),
        bias.float() if bias is not None else None,
        initial_state.float() if initial_state is not None else None,
        bos_mask,
        activation,
    )
    for actual_grad, reference_grad in zip(expected, reference):
        if reference_grad is not None:
            _assert_close(actual_grad, reference_grad)

    for _ in range(10):
        # Interleave other values of the same shape to expose stale scratch.
        causal_conv1d_bwd(
            -out_grad, final_grad, -x, weight, bias, initial_state, bos_mask,
            activation=activation,
            deterministic=True,
        )
        actual = causal_conv1d_bwd(*args, activation=activation, deterministic=True)
        _assert_identical_grads(actual, expected)
    _assert_identical_grads(args, snapshots)


@pytest.mark.parametrize("deterministic", [None, True, False])
@pytest.mark.parametrize("global_deterministic", [True, False])
def test_triton_deterministic_option_reaches_kernel(
    monkeypatch, deterministic, global_deterministic
):
    import importlib

    torch.manual_seed(7532)
    implementation = importlib.import_module(
        "xllm.modules.fused_ops.conv.triton.triton_causal_conv1d_bwd"
    )
    original = implementation._direct_backward
    selected_modes = []

    def record_mode(*args, **kwargs):
        selected_modes.append(args[-1])
        return original(*args, **kwargs)

    monkeypatch.setattr(implementation, "_direct_backward", record_mode)
    options = {} if deterministic is None else {"deterministic": deterministic}
    expected_mode = deterministic is True or global_deterministic
    x = _bf16_tensor(2, 513, 65).requires_grad_()
    layer = CausalConv1d(
        65, 4, bias=True, activation="silu", **options,
    ).to(device="cuda", dtype=torch.bfloat16)
    grad = _bf16_tensor(*x.shape)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(global_deterministic)
        for _ in range(3):
            out, _ = layer(x)
            actual = torch.autograd.grad(out, (x, layer.weight, layer.bias), grad)
            if expected_mode:
                if len(selected_modes) > 1:
                    _assert_identical_grads(actual, expected)
                expected = actual
        # Also exercise the keyword argument of the functional autograd API.
        out, _ = causal_conv1d(x, layer.weight, **options)
        torch.autograd.grad(out, (x, layer.weight), grad)
        causal_conv1d_bwd(
            grad, None, x, layer.weight, **options,
        )
        if global_deterministic:
            out, _ = causal_conv1d(x, layer.weight, **options)
            torch.use_deterministic_algorithms(False)
            torch.autograd.grad(out, (x, layer.weight), grad)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
    assert selected_modes == [expected_mode] * (6 if global_deterministic else 5)


@pytest.mark.parametrize("execution", ["streams", "graphs"])
@pytest.mark.parametrize("deterministic", [True, False])
def test_triton_backward_workspace_isolation(execution, deterministic):
    torch.manual_seed(8642)
    x = _bf16_tensor(2, 513, 65)
    weight = _bf16_tensor(65, 4)
    grads = [_bf16_tensor(*x.shape) * 513**-0.5 for _ in range(2)]

    def run(index):
        return causal_conv1d_bwd(
            grads[index], None, x, weight, activation="silu",
            deterministic=deterministic,
        )

    expected = [run(index) for index in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    graphs = []
    outputs = []
    current = torch.cuda.current_stream()
    if execution == "graphs":
        # Capture both graphs on the same stream, then replay on separate ones.
        # A cache keyed only by capture stream would give them shared scratch.
        streams[0].wait_stream(current)
        for index in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=streams[0]):
                outputs.append(run(index))
            graphs.append(graph)
        current.wait_stream(streams[0])

    for _ in range(5):
        if execution == "streams":
            outputs = []
        for index, stream in enumerate(streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                if execution == "graphs":
                    graphs[index].replay()
                else:
                    outputs.append(run(index))
        for stream in streams:
            current.wait_stream(stream)
        for actual, reference in zip(outputs, expected):
            if deterministic:
                _assert_identical_grads(actual, reference)
            else:
                for actual_grad, reference_grad in zip(actual, reference):
                    if reference_grad is not None:
                        _assert_close(actual_grad, reference_grad)
