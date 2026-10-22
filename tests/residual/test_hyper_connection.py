"""CPU tests of the hyper-connection controllers and the stream read / write.

    python -m pytest -q tests/residual/test_hyper_connection.py
"""

import math

import pytest
import torch

from xllm.modules.residual.hyper_connection import (
    HYPER_CONNECTIONS,
    IdentityHyperConnection,
    ManifoldHyperConnection,
    OrthogonalHyperConnection,
    build_hyper_connection,
    num_params_in_hyper_connection,
    quaternion_transport,
    sinkhorn_knopp,
    hc_read,
    hc_write,
)

B, L, N, D = 2, 3, 4, 16


def reference_read(x, a):
    """Loops over tokens, groups and streams; x [B,L,N,D] FP64, a [B,L,G,N]."""
    G = a.shape[-2]
    d = x.shape[-1] // G
    out = torch.zeros(*x.shape[:2], x.shape[-1], dtype=x.dtype)
    for bi in range(x.shape[0]):
        for t in range(x.shape[1]):
            for g in range(G):
                for n in range(x.shape[2]):
                    out[bi, t, g * d:(g + 1) * d] += a[bi, t, g, n] * x[bi, t, n, g * d:(g + 1) * d]
    return out


def reference_write(x, R, b, y):
    G = b.shape[-2]
    d = x.shape[-1] // G
    out = torch.zeros_like(x)
    for bi in range(x.shape[0]):
        for t in range(x.shape[1]):
            for g in range(G):
                sl = slice(g * d, (g + 1) * d)
                for n in range(x.shape[2]):
                    for m in range(x.shape[2]):
                        Rnm = (1.0 if n == m else 0.0) if R is None else R[bi, t, g, n, m]
                        out[bi, t, n, sl] += Rnm * x[bi, t, m, sl]
                    out[bi, t, n, sl] += b[bi, t, g, n] * y[bi, t, sl]
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
    R = sinkhorn_knopp(logits, 20)
    torch.testing.assert_close(R.sum(-1), torch.ones(4096, N, dtype=torch.float64))
    assert (R.sum(-2) - 1).abs().max() < 5e-2


def test_mhc_bias_is_zero_centred():
    """Weight decay pulls the biases to 0, which must be the near-identity start (4 I is a constant)."""
    hc = ManifoldHyperConnection(D, N)
    assert torch.equal(hc.bias, torch.zeros_like(hc.bias))
    with torch.no_grad():
        hc.scale.zero_()
    _, R, _ = hc(torch.randn(B, L, N, D))
    assert (R.diagonal(dim1=-2, dim2=-1) > 0.9).all()


def test_ohc_coefficients_are_fp32_in_an_fp64_model():
    torch.manual_seed(0)
    for groups in (1, 4):
        hc = OrthogonalHyperConnection(D, N, groups=groups).double()  # buffers too
        x = torch.randn(B, L, N, D, dtype=torch.float64)
        a, R, b = hc(x)
        assert a.dtype == R.dtype == b.dtype == torch.float32
        torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(N).expand_as(R))
        assert hc_write(x, R, b, hc_read(x, a)).dtype == torch.float64


def test_ohc_constants_are_not_checkpointed():
    hc = OrthogonalHyperConnection(D, N, groups=4)
    assert set(hc.state_dict()) == {'weight', 'bias', 'scale'}


MODEL_CONF = dict(arch='transformer', num_layers=2, model_dim=64, num_heads=4, num_kv_heads=2, head_dim=16,
                  ffn_hidden_dim=128, vocab_size=256, causal_attn_backend='flash', fused_block=False)


@pytest.mark.parametrize("residual_func", sorted(HYPER_CONNECTIONS))
def test_model_conf_accepts_every_hyper_connection(residual_func):
    from xllm.config import ModelConf
    ModelConf(**MODEL_CONF, residual_func=residual_func)


@pytest.mark.parametrize("kwargs,message", [
    (dict(residual_func='ohc', residual_streams=3), "residual_streams must be 4"),
    (dict(residual_func='mhc', residual_streams=5), "2 to 4 residual_streams"),
    (dict(residual_func='mhc', residual_groups=2), "residual_groups is an ohc option"),
    (dict(residual_func='ihc', fused_block=True), "no fused block yet"),
    (dict(residual_func='ohc', residual_groups=4, residual_freq_base=0.5), "residual_freq_base must be >= 1"),
    (dict(residual_func='mhc', residual_sinkhorn_iterations=0), "residual_sinkhorn_iterations must be >= 1"),
    (dict(residual_func='ohc', residual_groups=3), "residual_groups must divide model_dim"),
])
def test_model_conf_rejects(kwargs, message):
    from xllm.config import ModelConf
    ModelConf(**{**MODEL_CONF, 'residual_func': kwargs['residual_func']})
    with pytest.raises(AssertionError, match=message):
        ModelConf(**{**MODEL_CONF, **kwargs})


@pytest.mark.parametrize("groups", [1, 4])
def test_read_write_match_reference(groups):
    x = torch.randn(B, L, N, D, dtype=torch.float64)
    y = torch.randn(B, L, D, dtype=torch.float64)
    a = torch.rand(B, L, groups, N, dtype=torch.float64)
    b = torch.rand(B, L, groups, N, dtype=torch.float64)
    R = torch.randn(B, L, groups, N, N, dtype=torch.float64)
    a_, b_, R_ = (a[:, :, 0], b[:, :, 0], R[:, :, 0]) if groups == 1 else (a, b, R)
    torch.testing.assert_close(hc_read(x, a_), reference_read(x, a))
    torch.testing.assert_close(hc_write(x, R_, b_, y), reference_write(x, R, b, y))
    torch.testing.assert_close(hc_write(x, None, b_, y), reference_write(x, None, b, y))


def test_read_write_dtype_follows_streams():
    x = torch.randn(B, L, N, D, dtype=torch.bfloat16)
    a = torch.rand(B, L, N)
    assert hc_read(x, a).dtype == torch.bfloat16
    assert hc_write(x, None, a, hc_read(x, a)).dtype == torch.bfloat16


@pytest.mark.parametrize("residual_func,groups", [("ihc", 1), ("mhc", 1), ("ohc", 1), ("ohc", 4)])
def test_controller_init_is_near_plain_residual(residual_func, groups):
    torch.manual_seed(0)
    hc = build_hyper_connection(residual_func, D, N, groups=groups)
    x = torch.randn(B, L, N, D)
    a, R, b = hc(x)
    shape = (B, L, N) if groups == 1 else (B, L, groups, N)
    assert a.shape == shape and b.shape == shape and a.dtype == torch.float32
    assert (a - 0.5).abs().max() < 0.05  # sigmoid(0.01 * O(1))
    assert (b - 1.0).abs().max() < 0.1
    if residual_func == 'ihc':
        assert R is None
    else:
        eye = torch.eye(N).expand(*shape, N)
        assert R.shape == eye.shape
        assert (R - eye).abs().max() < 0.1
    assert sum(p.numel() for p in hc.parameters()) == num_params_in_hyper_connection(residual_func, D, N, groups)


def test_ohc_groups_with_equal_rates_match_ungrouped():
    torch.manual_seed(0)
    single = OrthogonalHyperConnection(D, N)
    grouped = OrthogonalHyperConnection(D, N, groups=4, freq_base=1.0)
    with torch.no_grad():
        grouped.weight.copy_(single.weight)
        single.bias.normal_()
        grouped.bias.copy_(single.bias.expand(4, -1))
        single.scale.fill_(0.5)
        grouped.scale.fill_(0.5)
    x = torch.randn(B, L, N, D)
    y = torch.randn(B, L, D)
    a1, R1, b1 = single(x)
    a4, R4, b4 = grouped(x)
    torch.testing.assert_close(a4, a1.unsqueeze(-2).expand(-1, -1, 4, -1))
    torch.testing.assert_close(R4, R1.unsqueeze(-3).expand(-1, -1, 4, -1, -1))
    torch.testing.assert_close(hc_read(x, a4), hc_read(x, a1))
    torch.testing.assert_close(hc_write(x, R4, b4, y), hc_write(x, R1, b1, y))


def test_ohc_group_rates():
    hc = OrthogonalHyperConnection(D, N, groups=4, freq_base=2.0)
    torch.testing.assert_close(hc.rates, torch.tensor([2.0 ** (-g / 3) for g in range(4)]))
    assert hc.rotation_scale.requires_grad is False  # frozen alpha


@pytest.mark.parametrize("residual_func", ["ihc", "mhc", "ohc"])
def test_all_controller_parameters_get_gradients(residual_func):
    torch.manual_seed(0)
    hc = build_hyper_connection(residual_func, D, N)
    branch = torch.nn.Linear(D, D)
    x = torch.randn(B, L, N, D, requires_grad=True)
    a, R, b = hc(x)
    out = hc_write(x, R, b, branch(hc_read(x, a)))
    out.float().square().sum().backward()
    for name, p in hc.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert x.grad is not None and torch.isfinite(x.grad).all()
