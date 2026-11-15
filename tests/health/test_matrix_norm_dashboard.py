"""Routed-expert and MoVA-value norms: the jsonl gets every norm, TensorBoard / W&B get statistics over them."""
import json
from types import SimpleNamespace

import pytest
import torch

import xllm.metrics as metrics_module
from tests.health.helpers import TOKENS, shard, tiny_transformer
from tests.health.test_routed_matrix_norms import CONFIGS, routed_group  # noqa: F401  (fixture)
from xllm.health_metrics import HealthMetricCollector
from xllm.metrics import MetricLogger


def logged_payloads(model):
    """Run one sampled step with weight and gradient norms; return (metrics, dashboard_metrics) given to the logger."""
    calls = []
    logger = SimpleNamespace(log=lambda metrics, step, dashboard_metrics=None: calls.append((metrics, dashboard_metrics)))
    collector = HealthMetricCollector(
        enabled=True, freq=1, num_layers=len(model.layers), enable_weight_norms=True, enable_gradient_norms=True,
    )
    collector.begin_step(1)
    output, _, _ = model.eval()(TOKENS, multi_segments=False)
    collector.end_forward()
    output.float().square().mean().backward()
    collector.record_matrix_norms(model)
    collector.flush(logger, 1)
    (payloads,) = calls
    return payloads


@pytest.mark.usefixtures("routed_group")
@pytest.mark.parametrize("backend", ["fsdp1", "fsdp2"])
# 2 routed layers x 2 families x (3 expert projections [+ 1 value projection])
@pytest.mark.parametrize("config,num_groups", [("moe_shared", 12), ("mova_shared_gate", 16)])
def test_dashboards_get_expert_and_value_statistics(backend, config, num_groups):
    model = shard(tiny_transformer(3, ddp_backend=backend, **CONFIGS[config]), backend)

    metrics, dashboard = logged_payloads(model)

    packed = {name for name in metrics if "/experts/expert_" in name or "/values/value_" in name}
    groups = {}
    for name in packed:
        family, layer, *group, _, projection, _ = name.split("/")
        groups.setdefault("/".join([family, layer, *group, projection]), []).append(metrics[name])
    assert len(groups) == num_groups
    for prefix, norms in groups.items():
        norms = torch.tensor(norms, dtype=torch.float64)
        for stat, value in (("mean", norms.mean()), ("min", norms.min()), ("max", norms.max()), ("std", norms.std())):
            assert dashboard[f"{prefix}/l2_{stat}"] == pytest.approx(value.item(), rel=1e-6)
    # Everything else, including each shared expert, reaches the dashboards unchanged.
    unpacked = {name: value for name, value in metrics.items() if name not in packed}
    assert {name: value for name, value in dashboard.items() if name in metrics} == unpacked
    assert len(dashboard) == len(unpacked) + 4 * num_groups


@pytest.mark.usefixtures("single_process_group")
def test_dense_models_send_one_payload():
    metrics, dashboard = logged_payloads(shard(tiny_transformer(2), "fsdp2"))

    assert dashboard is None
    assert any(name.startswith("weight_norms/") for name in metrics)


def test_metric_logger_sends_dashboard_metrics_to_tensorboard_and_wandb(tmp_path, monkeypatch):
    mlogger = MetricLogger(tmp_path, tag="health", enable=True, log_tb=True)
    tensorboard, wandb = [], []
    monkeypatch.setattr(
        mlogger.tb_writer.file_writer, "add_summary",
        lambda summary, global_step: tensorboard.extend(value.tag for value in summary.value),
    )
    monkeypatch.setattr(metrics_module, "wandb", SimpleNamespace(log=lambda m, step: wandb.append(m)), raising=False)
    mlogger.log_wandb = True

    mlogger.log({"a/expert_000": 1.0, "a/expert_001": 3.0, "b": 5.0}, 7, dashboard_metrics={"a/l2_mean": 2.0, "b": 5.0})
    mlogger.close()

    assert tensorboard == ["health/a/l2_mean", "health/b"]
    assert wandb == [{"health/a/l2_mean": 2.0, "health/b": 5.0}]
    line = json.loads((tmp_path / "metrics.health.jsonl").read_text())
    assert {k: v for k, v in line.items() if k not in ("global_step", "created_at")} == {
        "a/expert_000": 1.0, "a/expert_001": 3.0, "b": 5.0,
    }
