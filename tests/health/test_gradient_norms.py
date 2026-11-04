import copy

import pytest

from tests.health.helpers import MATRICES, TOKENS, logged_metrics, shard, tiny_transformer

pytestmark = pytest.mark.usefixtures("single_process_group")


def _metric(layer_idx, path):
    return f"gradient_norms/layer_{layer_idx:03d}/{path}/l2"


def _gradient_norms(model, loss_scale, gradient_scale):
    metrics = logged_metrics(
        model, loss_scale=loss_scale, gradient_scale=gradient_scale, enable_gradient_norms=True,
    )
    return {name: value for name, value in metrics.items() if name.startswith("gradient_norms/")}


def _full_matrix_gradient_norms(model):
    """Reference: the same loss and backward on an unsharded copy, then every full matrix gradient's norm."""
    logits, _, _ = model.eval()(TOKENS, multi_segments=False)
    logits.float().square().mean().backward()
    return {
        _metric(i, path): layer.get_parameter(fqn).grad.norm().item()
        for i, layer in enumerate(model.layers) for path, fqn in MATRICES.items()
    }


# The loss is multiplied by loss_scale before backward, and record_matrix_norms divides gradient_scale out,
# so the logged norm is the plain gradient norm times loss_scale / gradient_scale.
SCALES = {
    "unscaled": (1.0, 1.0),
    "loss_scaled_and_unscaled": (4.0, 4.0),
    "loss_scaled_not_unscaled": (4.0, 1.0),
    "unscaled_by_a_fraction": (1.0, 0.5),
}


@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("scales", SCALES)
def test_gradient_norms_match_full_matrix_gradients(backend, num_layers, scales):
    loss_scale, gradient_scale = SCALES[scales]
    model = tiny_transformer(num_layers, ddp_backend=backend)
    expected = _full_matrix_gradient_norms(copy.deepcopy(model))

    logged = _gradient_norms(shard(model, backend), loss_scale, gradient_scale)

    assert logged == pytest.approx(
        {name: norm * loss_scale / gradient_scale for name, norm in expected.items()}, rel=1e-5,
    )


@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_zero_loss_logs_zero_gradient_norms(backend, num_layers):
    model = tiny_transformer(num_layers, ddp_backend=backend)

    logged = _gradient_norms(shard(model, backend), loss_scale=0.0, gradient_scale=1.0)

    assert logged == {_metric(i, path): 0.0 for i in range(num_layers) for path in MATRICES}
