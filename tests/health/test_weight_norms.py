import pytest
import torch

from tests.health.helpers import MATRICES, logged_metrics, shard, tiny_transformer

pytestmark = pytest.mark.usefixtures("single_process_group")


def _metric(layer_idx, path):
    return f"weight_norms/layer_{layer_idx:03d}/{path}/l2"


def _weight_norms(model):
    metrics = logged_metrics(model, enable_weight_norms=True)
    return {name: value for name, value in metrics.items() if name.startswith("weight_norms/")}


# A constant c over an R x C matrix has l2 = |c| * sqrt(R * C). Attention matrices are 8 x 8 (64 entries);
# MLP matrices are 16 x 8 or 8 x 16 (128 entries). Each case: (fill, expected l2) per matrix.
HAND_COMPUTED_CASES = {
    "ones": {path: (1.0, 8.0 if path.startswith("attention") else 11.3137085) for path in MATRICES},
    "distinct": {
        "attention/q_proj": (0.5, 4.0),
        "attention/k_proj": (-1.0, 8.0),
        "attention/v_proj": (2.0, 16.0),
        "attention/o_proj": (0.25, 2.0),
        "mlp/gate_proj": (1.0, 11.3137085),
        "mlp/up_proj": (-0.5, 5.6568542),
        "mlp/down_proj": (0.125, 1.4142136),
    },
    "zeros": {path: (0.0, 0.0) for path in MATRICES},
}


@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("case", HAND_COMPUTED_CASES)
def test_weight_norms_match_hand_computed_values(backend, num_layers, case):
    model = tiny_transformer(num_layers, ddp_backend=backend)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            for path, (fill, _) in HAND_COMPUTED_CASES[case].items():
                layer.get_parameter(MATRICES[path]).fill_(fill * (i + 1))  # layer i is (i + 1) x

    logged = _weight_norms(shard(model, backend))

    assert logged == pytest.approx({
        _metric(i, path): norm * (i + 1)
        for i in range(num_layers) for path, (_, norm) in HAND_COMPUTED_CASES[case].items()
    }, abs=1e-5)


@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
@pytest.mark.parametrize("num_layers", [1, 3])
def test_weight_norms_match_full_matrices(backend, num_layers):
    model = tiny_transformer(num_layers, ddp_backend=backend)
    expected = {
        _metric(i, path): layer.get_parameter(fqn).norm().item()
        for i, layer in enumerate(model.layers) for path, fqn in MATRICES.items()
    }

    logged = _weight_norms(shard(model, backend))

    assert logged == pytest.approx(expected, rel=1e-5)
