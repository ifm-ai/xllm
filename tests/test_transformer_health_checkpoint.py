from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import xllm.health_metrics as health_metrics
import xllm.models.transformer as transformer


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_unfused_logit_stats_preserve_checkpoint_backward(monkeypatch, sampled, masked):
    output_cls = transformer.TransformerOutputLayer
    output = output_cls.__new__(output_cls)
    nn.Module.__init__(output)
    # Training always uses the fused output; exercise the unfused evaluation path.
    output.eval()
    output.final_norm = nn.Identity()
    output.final_norm.gather_input = True
    output.output = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        output.output.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, -1.0, 0.0],
        ]))

    def cross_entropy(logits, target):
        return F.cross_entropy(
            logits.flatten(0, 1), target.flatten(), reduction="none"
        ).view_as(target)

    monkeypatch.setattr(transformer, "vocab_parallel_cross_entropy", cross_entropy)
    x = torch.tensor([[[1.0, -2.0, 3.0], [4.0, 0.0, -1.0], [2.0, 1.0, 0.0]]])
    target = torch.tensor([[0, 1, -100]])
    mask = torch.tensor([[True, False, True]]) if masked else None
    reference_x = x.clone().requires_grad_()
    reference_loss = output(reference_x, target, mask)
    reference_gradients = torch.autograd.grad(
        reference_loss.sum(), (reference_x, output.output.weight)
    )

    collector = health_metrics.HealthMetricCollector(
        enabled=True, freq=2, num_layers=1, enable_logits_stats=True,
    )
    step = 2 if sampled else 1
    collector.begin_step(step)
    x.requires_grad_()
    try:
        loss = checkpoint(output, x, target, mask, use_reentrant=False)
        collector.record_activation(0, x)
    finally:
        collector.end_forward()
    gradients = torch.autograd.grad(loss.sum(), (x, output.output.weight))
    torch.testing.assert_close(loss, reference_loss)
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference)

    logged_metrics = {}
    logger = SimpleNamespace(log=lambda metrics, step: logged_metrics.update(metrics))
    collector.flush(logger, step)
    if sampled:
        valid_tokens = target.ne(-100)
        if mask is not None:
            valid_tokens &= mask
        expected = output.output(x).detach()[valid_tokens]
        assert logged_metrics["logits_stats/mean"] == pytest.approx(expected.mean().item())
        assert logged_metrics["logits_stats/min"] == expected.min().item()
        assert logged_metrics["logits_stats/max"] == expected.max().item()
    else:
        assert logged_metrics == {}
