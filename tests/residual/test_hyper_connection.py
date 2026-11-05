"""CPU tests of the hyper-connection residuals: coefficients, stream read / write, the read -> forward
pairing inside a sublayer, and the config checks.

    python -m pytest -q tests/residual/test_hyper_connection.py
"""

import pytest
import torch

from xllm.modules.residual import build_residual, num_params_in_residual, num_residual_streams, expand_streams, sum_streams
from xllm.modules.residual.base import BaseResidual
from xllm.modules.residual.hyper_connection import (
    HYPER_CONNECTIONS,
    ManifoldHyperConnection,
    OrthogonalHyperConnection,
    quaternion_transport,
    sinkhorn_knopp,
    hc_read,
    hc_write,
)

B, L, N, D = 2, 3, 4, 16


def reference_read(x, a):
    """Loops over tokens and streams; x [B, L, N, D] FP64, a [B, L, N]."""
    out = torch.zeros(*x.shape[:2], x.shape[-1], dtype=x.dtype)
    for bi in range(x.shape[0]):
        for t in range(x.shape[1]):
            for n in range(x.shape[2]):
                out[bi, t] += a[bi, t, n] * x[bi, t, n]
    return out


def reference_write(x, R, b, y):
    out = torch.zeros_like(x)
    for bi in range(x.shape[0]):
        for t in range(x.shape[1]):
            for n in range(x.shape[2]):
                for m in range(x.shape[2]):
                    Rnm = (1.0 if n == m else 0.0) if R is None else R[bi, t, n, m]
                    out[bi, t, n] += Rnm * x[bi, t, m]
                out[bi, t, n] += b[bi, t, n] * y[bi, t]
    return out


def test_quaternion_identity_at_zero_logits():
    R = quaternion_transport(torch.zeros(B, L, 8))
    assert torch.equal(R, torch.eye(4).expand(B, L, 4, 4))


def test_quaternion_is_special_orthogonal():
    logits = torch.randn(B, L, 8, dtype=torch.float64)
    R = quaternion_transport(logits)
    eye = torch.eye(4, dtype=torch.float64).expand(B, L, 4, 4)
    torch.testing.assert_close(R @ R.transpose(-1, -2), eye)
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(B, L, dtype=torch.float64))


def test_quaternion_gradcheck():
    logits = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda l: quaternion_transport(l), (logits,))


def test_sinkhorn_converges_to_doubly_stochastic():
    logits = torch.randn(B, L, N, N, dtype=torch.float64)
    R = sinkhorn_knopp(logits, 50)
    torch.testing.assert_close(R.sum(-1), torch.ones(B, L, N, dtype=torch.float64))
    torch.testing.assert_close(R.sum(-2), torch.ones(B, L, N, dtype=torch.float64), atol=1e-6, rtol=0)


def test_mhc_transport_at_the_configured_regime():
    """20 iterations around the 4 I start: rows exact, columns approximately."""
    torch.manual_seed(0)
    logits = 4.0 * torch.eye(N, dtype=torch.float64) + 0.5 * torch.randn(4096, N, N, dtype=torch.float64)
    R = sinkhorn_knopp(logits, ManifoldHyperConnection.SINKHORN_ITERATIONS)
    torch.testing.assert_close(R.sum(-1), torch.ones(4096, N, dtype=torch.float64))
    assert (R.sum(-2) - 1).abs().max() < 5e-2


def test_mhc_bias_is_zero_centred():
    """Weight decay pulls the biases to 0, which must be the near-identity start (4 I is a constant)."""
    hc = ManifoldHyperConnection(0, D)
    assert torch.equal(hc.bias, torch.zeros_like(hc.bias))
    with torch.no_grad():
        hc.scale.zero_()
    _, R, _ = hc.coefficients(torch.randn(B, L, N, D))
    assert (R.diagonal(dim1=-2, dim2=-1) > 0.9).all()


def test_ohc_coefficients_are_fp32_in_an_fp64_model():
    torch.manual_seed(0)
    hc = OrthogonalHyperConnection(0, D).double()
    x = torch.randn(B, L, N, D, dtype=torch.float64)
    a, R, b = hc.coefficients(x)
    assert a.dtype == R.dtype == b.dtype == torch.float32
    torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(N).expand_as(R))
    h, state = hc.read(x)
    assert hc(h, x, state=state).dtype == torch.float64


def test_only_parameters_are_checkpointed():
    for residual_func in HYPER_CONNECTIONS:
        hc = build_residual(residual_func, 0, D, None, eps=1e-6)
        assert set(hc.state_dict()) == {'weight', 'bias', 'scale'}, residual_func


MODEL_CONF = dict(arch='transformer', num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2, head_dim=16,
                  ffn_hidden_dim=128, vocab_size=256, causal_attn_backend='flash', fused_block=False)


@pytest.mark.parametrize("residual_func", sorted(HYPER_CONNECTIONS))
def test_model_conf_accepts_every_hyper_connection(residual_func):
    from xllm.config import ModelConf
    ModelConf(**MODEL_CONF, residual_func=residual_func)


def test_config_names_match_the_classes():
    from xllm.config import HYPER_CONNECTION_FUNCS, FUSED_BLOCK_RESIDUALS
    assert set(HYPER_CONNECTION_FUNCS) == set(HYPER_CONNECTIONS)
    for name, hc in HYPER_CONNECTIONS.items():
        assert hc.NAME == name
        # a variant has a fused block exactly when it owns coefficient kernels (coef_fwd / coef_bwd overridden)
        assert hc.has_kernels() == ('coef_bwd' in vars(hc)) == (name in FUSED_BLOCK_RESIDUALS), name


def test_model_conf_fused_block_allowlist_and_unknown_names():
    from xllm.config import ModelConf
    ModelConf(**{**MODEL_CONF, 'residual_func': 'ohc', 'fused_block': True})  # ihc / ohc have a fused block
    for residual_func in ('mhc', 'ortho'):  # not invertible / not implemented by the fused block
        with pytest.raises(AssertionError, match="fused blocks implement"):
            ModelConf(**{**MODEL_CONF, 'residual_func': residual_func, 'fused_block': True})
    with pytest.raises(AssertionError):
        ModelConf(**{**MODEL_CONF, 'residual_func': 'hc'})
    # residual_ckpt_interval: a setting of the hyper-connection fused block
    ModelConf(**{**MODEL_CONF, 'residual_func': 'ohc', 'fused_block': True, 'residual_ckpt_interval': 4})
    for bad in (dict(residual_func='ohc', residual_ckpt_interval=4),  # the module path keeps every layer's streams
                dict(residual_func='base', fused_block=True, residual_ckpt_interval=4),
                dict(residual_func='ohc', fused_block=True, residual_ckpt_interval=-1)):
        with pytest.raises(AssertionError, match="residual_ckpt_interval"):
            ModelConf(**{**MODEL_CONF, **bad})


def test_read_write_match_reference():
    x = torch.randn(B, L, N, D, dtype=torch.float64)
    y = torch.randn(B, L, D, dtype=torch.float64)
    a = torch.rand(B, L, N, dtype=torch.float64)
    b = torch.rand(B, L, N, dtype=torch.float64)
    R = torch.randn(B, L, N, N, dtype=torch.float64)
    torch.testing.assert_close(hc_read(x, a), reference_read(x, a))
    torch.testing.assert_close(hc_write(x, R, b, y), reference_write(x, R, b, y))
    torch.testing.assert_close(hc_write(x, None, b, y), reference_write(x, None, b, y))


def test_read_write_dtype_follows_streams():
    x = torch.randn(B, L, N, D, dtype=torch.bfloat16)
    a = torch.rand(B, L, N)
    assert hc_read(x, a).dtype == torch.bfloat16
    assert hc_write(x, None, a, hc_read(x, a)).dtype == torch.bfloat16


def test_streams_expand_and_sum():
    x0 = torch.randn(B, L, D)
    assert expand_streams(x0, 1) is x0 and sum_streams(x0, 1) is x0
    x = expand_streams(x0, N)
    assert x.shape == (B, L, N, D) and x.is_contiguous() and torch.equal(x[:, :, 2], x0)
    torch.testing.assert_close(sum_streams(x, N), N * x0)
    assert num_residual_streams('ohc') == N and num_residual_streams('base') == 1


@pytest.mark.parametrize("residual_func", sorted(HYPER_CONNECTIONS))
def test_controller_init_is_near_plain_residual(residual_func):
    torch.manual_seed(0)
    hc = build_residual(residual_func, 1, D, None, eps=1e-6)
    x = torch.randn(B, L, N, D)
    a, R, b = hc.coefficients(x)
    assert a.shape == (B, L, N) and b.shape == (B, L, N) and a.dtype == torch.float32
    assert (a - 0.5).abs().max() < 0.05  # sigmoid(0.01 * O(1))
    assert (b - 1.0).abs().max() < 0.1
    if residual_func == 'ihc':
        assert R is None
    else:
        eye = torch.eye(N).expand(B, L, N, N)
        assert R.shape == eye.shape and (R - eye).abs().max() < 0.1
    assert sum(p.numel() for p in hc.parameters()) == num_params_in_residual(residual_func, D, None, D)


@pytest.mark.parametrize("residual_func", ['base', 'ortho'])
def test_plain_residuals_read_the_state_itself(residual_func):
    res = build_residual(residual_func, 1, D, num_heads=4, eps=1e-6)
    x = torch.randn(B, L, D)
    h, state = res.read(x)
    assert h is x and state is None
    assert isinstance(build_residual('base', 1, D, None), BaseResidual)


@pytest.mark.parametrize("residual_func", sorted(HYPER_CONNECTIONS))
def test_sublayer_read_then_write(residual_func):
    """The sublayer pattern: h, rc = read(X); y = branch(h); X' = residual(y, X, state=rc) equals the explicit
    math and every parameter and the streams get gradients; the write needs the state read returned."""
    torch.manual_seed(0)
    hc = build_residual(residual_func, 1, D, None, eps=1e-6)
    branch = torch.nn.Linear(D, D)
    x = torch.randn(B, L, N, D, requires_grad=True)
    a, R, b = hc.coefficients(x)
    expected = hc_write(x, R, b, branch(hc_read(x, a)))
    h, state = hc.read(x)
    assert state[1].shape == (B, L, N) and (state[0] is None) == (residual_func == 'ihc')
    out = hc(branch(h), x, c=None, state=state)
    torch.testing.assert_close(out, expected)
    out.float().square().sum().backward()
    for name, p in hc.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(AssertionError, match="state"):
        hc(branch(h), x)
