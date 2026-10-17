import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch._dynamo  # noqa: F401  (loaded before the stubs so it never sees the fake triton)
import torch.distributed as dist
import torch.nn.functional as F

# GPU-only packages the model imports at module load; stub them only when they are not installed.
# This lives here (not in conftest) so ranks spawned by multiprocessing can import xllm too.
_MISSING = {pkg for pkg in ("xllm_extension", "triton", "flash_attn_2_cuda", "fla")
            if importlib.util.find_spec(pkg) is None}
for name in (
    "xllm_extension", "xllm_extension.ops", "triton", "triton.language", "flash_attn_2_cuda",
    "fla", "fla.ops", "fla.ops.utils", "fla.modules", "fla.modules.conv", "fla.modules.conv.triton",
):
    if name.split(".")[0] in _MISSING:
        sys.modules.setdefault(name, MagicMock())

from xllm.config import ModelConf  # noqa: E402
from xllm.distributed import FullyShardedDataParallel  # noqa: E402
from xllm.health_metrics import HealthMetricCollector  # noqa: E402
from xllm.models.transformer import Transformer  # noqa: E402

TOKENS = torch.tensor([[3, 7, 5, 9], [4, 4, 8, 6]])
TARGETS = torch.tensor([[3, -100, 5, 9], [4, 4, -100, 6]])          # -100 is the ignore index
MASK = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.bool)  # token_mask

# Every matrix whose norm is logged for a Transformer layer: metric path -> parameter inside the layer.
MATRICES = {
    "attention/q_proj": "attention.wq.weight",
    "attention/k_proj": "attention.wk.weight",
    "attention/v_proj": "attention.wv.weight",
    "attention/o_proj": "attention.wo.weight",
    "mlp/gate_proj": "nffn.fc1.weight",
    "mlp/up_proj": "nffn.fc3.weight",
    "mlp/down_proj": "nffn.fc2.weight",
}


def cpu_rms_norm_forward(self, x):
    # GroupRMSNorm runs a CUDA kernel; torch's rms_norm keeps the forward on CPU. Parameters are untouched.
    return F.rms_norm(x, x.shape[-1:], self.weight + 1.0 if self.elementwise_affine else None, self.eps)


def tiny_transformer(num_layers, **overrides):
    """A real Transformer small enough for CPU: model_dim 8, 2 heads of 4, SwiGLU FFN of 16."""
    cfg = dict(
        num_layers=num_layers, model_dim=8, num_heads=2, ffn_hidden_dim=16,
        apply_rmsnorm=True, swiglu=True, vocab_size=11, chunk_size=4,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return Transformer(ModelConf(**cfg), SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))


def shard(model, backend):
    """Wrap like build_model: each decoder layer, the embedding, the output layer, then the whole model."""
    if backend == "fsdp1":
        from torch.distributed.fsdp.wrap import enable_wrap, wrap
        config = dict(wrapper_cls=FullyShardedDataParallel, device_id=torch.device("cpu"))
    else:
        from torch.distributed.device_mesh import init_device_mesh
        from xllm.distributed.wrap import enable_wrap, wrap
        config = dict(mesh=init_device_mesh("cpu", (dist.get_world_size(),)))
    with enable_wrap(**config):
        for i, layer in enumerate(model.layers):
            model.layers[i] = wrap(layer)
        model.embed, model.output = wrap(model.embed), wrap(model.output)
        return wrap(model)


def logged_metrics(model, tokens=TOKENS, targets=None, token_mask=None, train=False,
                   loss_scale=None, gradient_scale=1.0, **flags):
    """Run one sampled training step with a real collector and return everything flush logs.

    Forward with the collector active; if loss_scale is given, backward on loss_scale * mean(output^2);
    then matrix norms when enabled; then flush.
    """
    collector = HealthMetricCollector(enabled=True, freq=1, num_layers=len(model.layers), **flags)
    collector.begin_step(1)
    try:
        with torch.set_grad_enabled(loss_scale is not None):
            output, _, _ = model.train(train)(tokens, multi_segments=False, targets=targets, token_mask=token_mask)
    finally:
        collector.end_forward()
    if loss_scale is not None:
        (output.float().square().mean() * loss_scale).backward()
    if flags.get("enable_weight_norms") or flags.get("enable_gradient_norms"):
        collector.record_matrix_norms(model, gradient_scale=gradient_scale)
    logged = {}
    collector.flush(SimpleNamespace(log=lambda metrics, step: logged.update(metrics)), 1)
    return logged


def stats(metrics, prefix, names=("mean", "min", "max")):
    return [metrics[f"{prefix}/{name}"] for name in names]


def make_layers_identity(model):
    # Zeroing each sublayer's output projection leaves only the residual, so every layer returns its input.
    with torch.no_grad():
        for layer in model.layers:
            layer.attention.wo.weight.zero_()
            layer.nffn.fc2.weight.zero_()


def set_embeddings(model, rows):
    with torch.no_grad():
        for token, row in rows.items():
            model.embed.weight[token] = torch.tensor(row)
