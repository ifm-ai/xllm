"""The fused Gekko block on a sampled health-logging step (xllm.health_metrics.HealthMetricCollector): such a step
runs the reference q / k preparation, output gate, adaptive working memory and cross-entropy paths, because the hooks
need the pre-RoPE q / k, the gate activation, the memory update ratio and the logits that the fused kernels never
materialise. The step's loss and gradients must match the fused kernels' step, no fused kernel may run, and the
collector must flush the full set of Gekko and logits statistics. 1 GPU, small model.
Run with the extension on the path, e.g. PYTHONPATH=<xext> pytest tests/test_health_fused_gekko.py."""
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    import xllm_extension  # noqa: F401
    _HAVE_EXT = True
except ImportError:
    _HAVE_EXT = False

pytestmark = pytest.mark.skipif(not _HAVE_EXT or not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                                reason="needs a Blackwell GPU and xllm_extension")


def _init():
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        from xllm.distributed import init_torch_distributed, initialize_model_parallel
        init_torch_distributed(600)
        initialize_model_parallel(1, 1, 600)


def _forbid_fused_kernels(monkeypatch):
    """The kernels a health-logging step must not run (imported lazily by the block, so patching the modules works)."""
    import xllm.modules.fused_ops.cute.qk_prep as qk
    import xllm.modules.fused_ops.cute.conv_qk_prep as cqk
    import xllm.modules.fused_ops.cute.attn_gate as gate
    import xllm.modules.fused_ops.cute.adaptive_working_memory as awm
    import xllm.modules.fused_ops.cute.cross_entropy as ce
    import xllm.modules.fused_ops.mxfp8 as mx
    for mod, name in ((qk, "qk_prep_fwd"), (qk, "qk_prep_bwd"), (cqk, "conv_qk_prep_fwd"), (gate, "attn_gate_fwd"),
                      (awm, "adaptive_working_memory_fwd"), (ce, "ce_logits"), (mx, "mx_gate_quantize")):
        def forbidden(*a, _name=name, **k):
            raise AssertionError(f"{_name} ran on a health-logging step")
        monkeypatch.setattr(mod, name, forbidden)


@pytest.mark.parametrize("fp8, overrides", [
    (False, {}),
    (False, {"recompute_conv_qk": False}),  # the q / k prep outputs are kept for the backward
    (True, {}),  # MXFP8 GEMMs: the gate output is quantised from the reference gate
])
def test_health_step_matches_fused_step(monkeypatch, fp8, overrides):
    _init()
    from tests import test_fp8_gekko_block as base
    from xllm.health_metrics import HealthMetricCollector

    g = torch.Generator(device="cuda").manual_seed(1)
    tokens = torch.randint(3, base._Tok.vocab_size, (1, 513), device="cuda", generator=g)
    tokens[0, 200] = base._Tok.eos_id  # a segment boundary
    x, y = tokens[:, :-1], tokens[:, 1:]
    loss_fused, grads_fused = base._step(base._build(fp8, **overrides), x, y)

    _forbid_fused_kernels(monkeypatch)
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=2, enable_gekko_stats=True,
                                     enable_logits_stats=True)
    collector.begin_step(1)
    model = base._build(fp8, **overrides)
    tok_loss, _, _ = model(tokens=x, targets=y, token_mask=None, multi_segments=True, deterministic=False)
    collector.end_forward()
    loss = tok_loss.sum() / tok_loss.numel()
    loss.backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    logged = {}
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.update(metrics)), 1)

    for prefix in ("gekko_stats/layer_000/", "gekko_stats/layer_001/", "q_stats/layer_001/", "k_stats/layer_001/", "logits_stats/",
                   "activation_stats/layer_001"):
        assert any(k.startswith(prefix) for k in logged), (prefix, sorted(logged))
    assert all(torch.isfinite(torch.as_tensor(float(v))) for v in logged.values()), logged

    # the same function through the reference path: bf16 rounding differences only (fp8: the fused step's gate
    # output is quantised from the fused gate kernel instead)
    assert abs(loss.item() - loss_fused) / loss_fused < 2e-2, (loss_fused, loss.item())
    assert grads.keys() == grads_fused.keys()
    for name in grads_fused:
        a, b = grads_fused[name].flatten(), grads[name].flatten()
        if a.norm() == 0:
            assert b.norm() == 0, name
            continue
        cos = torch.dot(a, b) / (a.norm() * b.norm())
        rel = (a - b).norm() / a.norm()
        assert cos > 0.99 and rel < 0.15, (name, cos.item(), rel.item())


def test_cuda_graph_block_refuses_a_health_step(monkeypatch):
    """A replayed graph would skip the hooks: the block asserts instead of logging stale statistics."""
    _init()
    from tests import test_fp8_gekko_block as base
    from xllm.health_metrics import HealthMetricCollector

    g = torch.Generator(device="cuda").manual_seed(1)
    tokens = torch.randint(3, base._Tok.vocab_size, (1, 513), device="cuda", generator=g)
    x, y = tokens[:, :-1], tokens[:, 1:]
    model = base._build(False, cuda_graph_block=True, dropout=0.0, attention_dropout=0.0, hidden_dropout=0.0)
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=2, enable_gekko_stats=True)
    collector.begin_step(1)
    with pytest.raises(AssertionError, match="cuda_graph_block cannot record health metrics"):
        model(tokens=x, targets=y, token_mask=None, multi_segments=True, deterministic=False)
    collector.end_forward()
