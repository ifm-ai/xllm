"""Group RMS norm and group layer norm ops against a float64 reference: accuracy, determinism and robustness."""

import pytest
import torch

import xllm_extension

ops = xllm_extension.ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# (rows, groups, width): every kernel family of the Blackwell dispatcher (narrow groups packed into warps, one group
# per team, wide groups split over blocks or clusters) plus widths and group counts no vector width divides.
SHAPES = [
    (128, 64, 24), (129, 64, 40), (2048, 128, 64), (128, 32, 128), (2048, 64, 128), (1024, 16, 256),
    (128, 8, 512), (2048, 4, 1024), (128, 4, 2048), (2048, 1, 4096), (128, 1, 8192), (3, 2, 12288),
    (128, 1, 65536), (1, 1, 65536), (257, 3, 37), (5, 7, 65), (64, 5, 353), (32, 1, 1765), (2, 1, 8193),
    (1, 1, 1), (7, 9, 3),
]

# Group norms traced from model training calls (rows, groups, width, norm, affine, eps), all bfloat16.
MODEL_SHAPES = [
    (65536, 64, 128, 'layer', True, 1e-5), (262144, 1, 1536, 'rms', True, 1e-6),
    (262144, 64, 24, 'rms', False, 1e-6), (262144, 64, 64, 'rms', True, 1e-6),
    (65536, 1, 4096, 'rms', True, 1e-6), (65536, 2, 1280, 'rms', True, 1e-6),
    (65536, 32, 128, 'rms', True, 1e-6), (65536, 32, 80, 'rms', False, 1e-6),
    (65536, 4, 1024, 'rms', True, 1e-6), (65536, 64, 128, 'rms', True, 1e-6),
    (65536, 64, 40, 'rms', False, 1e-6), (65536, 64, 64, 'rms', False, 1e-6),
    (65536, 8, 128, 'rms', False, 1e-6),
]


# The float32 statistics of inputs whose mean is far from zero next to their spread (400x for 'offset', 1000x for
# 'huge') carry an error that grows with that ratio, in any kernel: float32 outputs of such inputs are held to the base
# tolerances times a quarter of the ratio, about three times the largest error of both the Blackwell and the portable
# kernels.
CONDITIONING = {'randn': 1, 'outlier': 1, 'offset': 100, 'huge': 250}


def tolerance(dtype, name, kind='randn'):
    """(atol, rtol) per output; the statistics are float32 for every input dtype."""
    scale = CONDITIONING[kind] if dtype == torch.float32 else 1
    if name in ('mean', 'rstd'):
        return 1e-5 * scale, 1e-4 * scale
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:  # the tolerances of the repo's own group-norm tests for 16-bit outputs
        atol, rtol = (7e-3, 1e-2) if name in ('dgamma', 'dbeta') else (3e-3, 1e-2)
    return atol * scale, rtol * scale


def inputs(rows, groups, width, dtype, kind='randn', seed=0):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    channels = groups * width
    x = torch.randn(1, rows, channels, device='cuda', generator=generator)
    if kind == 'offset':  # a common offset 400x the spread: one-pass variance formulas cancel here
        x = x * 0.05 + 20.0
    elif kind == 'outlier':  # nearly constant groups with one far element each
        x = 1.5 + x * 1e-3
        x.view(rows, groups, width)[:, :, 0] += 40.0
    elif kind == 'huge':  # float32 values whose squares overflow while the centred variance does not
        x = x * 1.0e16 + 1.0e19
    dy = torch.randn(1, rows, channels, device='cuda', generator=generator) / rows ** 0.5
    gamma = torch.randn(channels, device='cuda', generator=generator).clamp(min=-0.9) + 1
    beta = torch.randn(channels, device='cuda', generator=generator)
    return x.to(dtype), dy.to(dtype), gamma.to(dtype), beta.to(dtype)


def misaligned(t):
    """A contiguous copy of t that starts one element past an aligned address."""
    return torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)[1:].view(t.shape).copy_(t)


def reference(norm, x, dy, gamma, beta, groups, eps):
    """Forward outputs and gradients in float64."""
    rows, channels = x.shape[1], x.shape[2]
    width = channels // groups
    xd = x.double().reshape(rows, groups, width)
    mean = xd.mean(-1, keepdim=True) if norm == 'layer' else None
    centred = xd - mean if norm == 'layer' else xd
    rstd = torch.rsqrt((centred * centred).mean(-1, keepdim=True) + eps)
    xhat = centred * rstd
    g = gamma.double().reshape(1, groups, width) if gamma is not None else None
    y = xhat * g if g is not None else xhat
    if beta is not None:
        y = y + beta.double().reshape(1, groups, width)
    out = {'y': y.reshape(x.shape), 'rstd': rstd.reshape(rows, groups)}
    if norm == 'layer':
        out['mean'] = mean.reshape(rows, groups)
    out.update(gradients(norm, xhat, rstd, dy, gamma, beta, groups))
    return out


def gradients(norm, xhat, rstd, dy, gamma, beta, groups):
    """dx, dgamma and dbeta in float64 from the normalized input xhat [rows, groups, width]."""
    rows, _, width = xhat.shape
    g = gamma.double().reshape(1, groups, width) if gamma is not None else None
    dyd = dy.double().reshape(rows, groups, width)
    dxhat = dyd * g if g is not None else dyd
    projection = (dxhat * xhat).mean(-1, keepdim=True)
    dx = dxhat - xhat * projection - (dxhat.mean(-1, keepdim=True) if norm == 'layer' else 0)
    out = {'dx': (rstd * dx).reshape(dy.shape)}
    if g is not None:
        out['dgamma'] = (dyd * xhat).sum(0).reshape(-1)
        if beta is not None:
            out['dbeta'] = dyd.sum(0).reshape(-1)
    return out


def recovered(y, gamma, beta, groups):
    """The normalized input a memory-efficient backward recovers from the stored output y, in float64.

    y is rounded to the input dtype when it is stored, so this, not the exact xhat, is what such a backward can
    differentiate."""
    rows, width = y.shape[1], y.shape[2] // groups
    xhat = y.double().reshape(rows, groups, width)
    if beta is not None:
        xhat = xhat - beta.double().reshape(1, groups, width)
    if gamma is not None:
        xhat = xhat / gamma.double().reshape(1, groups, width)
    return xhat


def forward(norm, x, groups, gamma, beta, eps):
    channels = x.size(-1)
    if norm == 'layer':
        y, mean, rstd = (ops.group_layer_norm_fwd_affine(x, channels, groups, gamma, beta, eps) if gamma is not None
                         else ops.group_layer_norm_fwd(x, channels, groups, eps))
        return {'y': y, 'mean': mean, 'rstd': rstd}
    y, rstd = (ops.group_rms_norm_fwd_affine(x, channels, groups, gamma, eps) if gamma is not None
               else ops.group_rms_norm_fwd(x, channels, groups, eps))
    return {'y': y, 'rstd': rstd}


def backward(norm, dy, x_or_y, groups, stats, gamma, beta, memory_efficient):
    channels = x_or_y.size(-1)
    if norm == 'layer':
        if gamma is not None:
            dx, dgamma, dbeta = ops.group_layer_norm_bwd_affine(dy, x_or_y, channels, groups, stats['mean'],
                                                                stats['rstd'], gamma, beta, memory_efficient)
            return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}
        return {'dx': ops.group_layer_norm_bwd(dy, x_or_y, channels, groups, stats['mean'], stats['rstd'],
                                               memory_efficient)}
    if gamma is not None:
        dx, dgamma = ops.group_rms_norm_bwd_affine(dy, x_or_y, channels, groups, stats['rstd'], gamma,
                                                   memory_efficient)
        return {'dx': dx, 'dgamma': dgamma}
    return {'dx': ops.group_rms_norm_bwd(dy, x_or_y, channels, groups, stats['rstd'], memory_efficient)}


def check(actual, expected, dtype, kind='randn'):
    for name, value in actual.items():
        atol, rtol = tolerance(dtype, name, kind)
        torch.testing.assert_close(value.double().reshape(expected[name].shape), expected[name], atol=atol,
                                   rtol=rtol, msg=lambda m: f'{name}: {m}')


def run_case(norm, rows, groups, width, dtype, affine, eps=1e-5, kind='randn', layout='contiguous'):
    """Checks the forward call and both backward modes against float64; returns every output for reuse."""
    x, dy, gamma, beta = inputs(rows, groups, width, dtype, kind)
    if layout == 'strided':  # rows of the input twice as far apart as their channels
        x = torch.empty(1, rows, 2 * x.size(-1), device='cuda', dtype=dtype)[..., :x.size(-1)].copy_(x)
    elif layout == 'misaligned':  # every tensor contiguous, one element past an aligned address
        x, dy, gamma, beta = (misaligned(t) for t in (x, dy, gamma, beta))
    gamma = gamma if affine else None
    beta = beta if affine and norm == 'layer' else None
    expected = reference(norm, x, dy, gamma, beta, groups, eps)
    out = forward(norm, x, groups, gamma, beta, eps)
    check(out, expected, dtype, kind)
    # Both backward modes differentiate the float64 forward state, rounded to the types the ops store.
    stats = {'rstd': expected['rstd'].float(), 'mean': expected['mean'].float() if norm == 'layer' else None}
    y = expected['y'].to(dtype)
    if layout == 'misaligned':
        stats = {k: v if v is None else misaligned(v) for k, v in stats.items()}
        y = misaligned(y)
    rstd = expected['rstd'].reshape(rows, groups, 1)
    results = [out]
    for memory_efficient in (False, True):
        result = backward(norm, dy, y if memory_efficient else x, groups, stats, gamma, beta, memory_efficient)
        if memory_efficient:
            grads = gradients(norm, recovered(y, gamma, beta, groups), rstd, dy, gamma, beta, groups)
        else:
            grads = {k: v for k, v in expected.items() if k in ('dx', 'dgamma', 'dbeta')}
        check(result, grads, dtype, kind)
        results.append(result)
    return x, dy, gamma, beta, stats, y, results


@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('dtype', DTYPES, ids=str)
@pytest.mark.parametrize('rows,groups,width', SHAPES)
@pytest.mark.parametrize('norm', ['rms', 'layer'])
def test_matches_float64(norm, rows, groups, width, dtype, affine):
    run_case(norm, rows, groups, width, dtype, affine)


@pytest.mark.parametrize('rows,groups,width,norm,affine,eps', MODEL_SHAPES)
def test_model_shapes(rows, groups, width, norm, affine, eps):
    run_case(norm, rows, groups, width, torch.bfloat16, affine, eps)


@pytest.mark.parametrize('layout', ['strided', 'misaligned'])
@pytest.mark.parametrize('affine', [False, True])
@pytest.mark.parametrize('dtype', DTYPES, ids=str)
@pytest.mark.parametrize('rows,groups,width', [(2048, 64, 40), (2048, 64, 64), (2048, 32, 80), (1024, 16, 256),
                                               (2048, 4, 1024), (128, 4, 2048), (128, 1, 8192), (128, 1, 65536),
                                               (64, 5, 353)])
@pytest.mark.parametrize('norm', ['rms', 'layer'])
def test_memory_layouts(norm, rows, groups, width, dtype, affine, layout):
    run_case(norm, rows, groups, width, dtype, affine, layout=layout)


@pytest.mark.parametrize('kind', ['offset', 'outlier'])
@pytest.mark.parametrize('dtype', DTYPES, ids=str)
@pytest.mark.parametrize('rows,groups,width', [(2048, 64, 64), (128, 8, 512), (128, 1, 8192), (2048, 1, 65536)])
@pytest.mark.parametrize('norm', ['rms', 'layer'])
def test_inputs_that_cancel(norm, rows, groups, width, dtype, kind):
    run_case(norm, rows, groups, width, dtype, affine=True, kind=kind)


@pytest.mark.parametrize('rows,groups,width', [(128, 1, 65536), (2048, 1, 65536), (128, 4, 2048), (2048, 64, 64)])
def test_float32_sum_of_squares_overflow(rows, groups, width):
    """Finite float32 inputs whose squares overflow while their centred variance does not."""
    run_case('layer', rows, groups, width, torch.float32, affine=True, kind='huge')


@pytest.mark.parametrize('dtype', DTYPES, ids=str)
@pytest.mark.parametrize('rows,groups,width', [(2048, 64, 64), (2048, 4, 2048), (8192, 64, 128), (2048, 1, 65536)])
@pytest.mark.parametrize('norm', ['rms', 'layer'])
def test_deterministic(norm, rows, groups, width, dtype):
    """Every output of every call is bitwise identical when the call is repeated, also under
    torch.use_deterministic_algorithms."""
    x, dy, gamma, beta, stats, y, first = run_case(norm, rows, groups, width, dtype, affine=True)
    beta = beta if norm == 'layer' else None
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        for deterministic in (False, True, True):
            torch.use_deterministic_algorithms(deterministic)
            again = [forward(norm, x, groups, gamma, beta, 1e-5)] + [
                backward(norm, dy, y if memory_efficient else x, groups, stats, gamma, beta, memory_efficient)
                for memory_efficient in (False, True)]
            for previous, current in zip(first, again):
                for name in previous:
                    assert torch.equal(previous[name], current[name]), name
    finally:
        torch.use_deterministic_algorithms(enabled)


def test_float64_inputs():
    """Float64 calls keep working; they are served by the portable kernels."""
    x = torch.randn(1, 64, 384, device='cuda', dtype=torch.float64)
    y, mean, rstd = ops.group_layer_norm_fwd(x, 384, 4, 1e-5)
    expected = reference('layer', x, x, None, None, 4, 1e-5)
    torch.testing.assert_close(y, expected['y'], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('channels,groups', [(64, 3), (64, 0)])
def test_rejects_groups_that_do_not_divide_the_channels(channels, groups):
    x = torch.randn(1, 4, channels, device='cuda')
    with pytest.raises(RuntimeError):
        ops.group_rms_norm_fwd(x, channels, groups, 1e-5)
