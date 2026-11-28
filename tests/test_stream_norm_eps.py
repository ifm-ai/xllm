"""model.embed_multiplier automatically scales residual-stream norm eps by embed_multiplier^2.

A model whose branch outputs are scaled by c (each block's wo, fc2 and routed experts' weight2 × c), and the same
model with embed_multiplier m = 1/c: the second model's residual stream is the first's × m, every read of the stream
is a norm, and with each stream norm's eps automatically scaled by m² every norm computes the same output while
in-branch norms keep rmsnorm_eps. With c a power of 2 the loss, every gradient and the inference logits match bit for
bit across Gekko, Transformer, Hybrid, MoE, MoVA, OrthoResidual, RMSNorm and LayerNorm, fused and unfused.

    PYTHONPATH=. pytest -q tests/test_stream_norm_eps.py                                   # one GPU (or CPU config tests)
    PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -q tests/test_stream_norm_eps.py    # tensor parallel 2
    STREAM_EPS_PARALLEL=cp PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -q tests/test_stream_norm_eps.py
                                                                                           # context parallel 2
"""
import math
import os
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from xllm.config import ModelConf

C = 0.5  # the depth factor; m = 1/C
M = 1 / C
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
CP = WORLD if os.environ.get("STREAM_EPS_PARALLEL") == "cp" else 1  # else the processes are tensor parallel
CHUNK = 1024
SEQ = 2 * CHUNK * CP  # two Gekko chunks per context-parallel rank

GEKKO = ModelConf(  # the 30M recipe's Gekko at d768, 2 layers
    arch="gekko", model_dim=768, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=256, v_head_dim=256,
    rope_head_dim=128, rope_base=500000, ffn_hidden_dim=2048, swiglu=True, qknorm=True, attn_gate_func="softplus",
    causal_conv_width=4, causal_conv_weight_normalization=True, causal_conv_backend="triton", sca_backend="xattn",
    apply_rmsnorm=True, layernorm_num_groups=2, timenorm_num_groups=16, timenorm_beta1=0.999, timenorm_beta2=0.9999,
    init_mode="he", vocab_size=1024, chunk_size=CHUNK, ddp_backend="fsdp2",
)
TRANSFORMER = ModelConf(
    arch="transformer", model_dim=512, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=128, ffn_hidden_dim=1024,
    swiglu=True, qknorm=True, apply_rmsnorm=True, layernorm_num_groups=2, causal_attn_backend="flash",
    init_mode="he", vocab_size=1024, chunk_size=CHUNK, ddp_backend="fsdp2",
)
HYBRID = replace(
    GEKKO, arch="hybrid", full_attn_layer_list="1", causal_attn_backend="flash",
)
TRANSFORMER_MOE = replace(
    TRANSFORMER, num_dense_layers=1, num_experts=4, num_activated_experts=2, num_shared_experts=1,
    expert_inter_dim=256, moe_expert_backend="torch", moe_permutation_backend="triton",
)
TRANSFORMER_MOVA = replace(
    TRANSFORMER_MOE, num_values=4, num_activated_values=2, mova_backend="torch",
)
BRANCH_OUT = (
    "gda.wo.weight",
    "attention.wo.weight",
    "mova.wo.weight",
    "nffn.fc2.weight",
    "moe.fc2.weight",
    "moe.experts.weight2",
)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need a GPU")


@pytest.fixture(scope="module", autouse=True)
def tensor_parallel():
    """Under torchrun, every test runs with tensor (default) or context parallel (STREAM_EPS_PARALLEL=cp) = the
    number of processes."""
    if WORLD > 1:
        from xllm.distributed import initialize_model_parallel
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(WORLD // CP, CP, timeout=600)
    yield


def build(cfg, branch_scale=1.0, unscale_stream_eps=False):
    from xllm.models.gekko import Gekko
    from xllm.models.hybrid import HybridGekko
    from xllm.models.transformer import Transformer
    cls = {"gekko": Gekko, "hybrid": HybridGekko, "transformer": Transformer}[cfg.arch]
    torch.manual_seed(0)
    model = cls(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    if branch_scale != 1.0:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith(BRANCH_OUT):
                    p.mul_(branch_scale)
    if unscale_stream_eps:
        scale = cfg.stream_eps_scale
        for mod in model.modules():
            if hasattr(mod, "norm_eps"):
                mod.norm_eps /= scale
                if hasattr(mod, "norm"):
                    mod.norm.eps /= scale
                if hasattr(mod, "final_norm"):
                    mod.final_norm.eps /= scale
            if hasattr(mod, "timenorm"):
                mod.timenorm.eps /= scale
            if hasattr(mod, "final_norm") and not hasattr(mod, "norm_eps"):
                mod.final_norm.eps /= scale
            if hasattr(mod, "residual") and hasattr(mod.residual, "norm"):
                mod.residual.norm.eps /= scale
    return model.cuda().to(torch.bfloat16)


def data(cfg):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(3, cfg.vocab_size, (1, SEQ), generator=g).cuda()
    targets = torch.randint(3, cfg.vocab_size, (1, SEQ), generator=g).cuda()
    return tokens, targets


def loss_and_grads(cfg, branch_scale=1.0, unscale_stream_eps=False):
    model = build(cfg, branch_scale, unscale_stream_eps=unscale_stream_eps).train()
    tokens, targets = data(cfg)
    lb = "dot" if cfg.num_experts > 0 else None
    loss, aux_loss, _ = model(tokens=tokens, targets=targets, multi_segments=False, moe_router_load_balancing_type=lb)
    loss = loss.float().sum() + (aux_loss.float() if aux_loss is not None else 0.0)
    loss.backward()
    # dL/d(c·W) = dL/dW / c: compare the branch model's output-projection gradients times c
    grads = {n: p.grad * (branch_scale if n.endswith(BRANCH_OUT) else 1.0) for n, p in model.named_parameters()}
    return loss, grads


def logits(cfg, branch_scale=1.0, unscale_stream_eps=False):
    """Inference: eval mode with the KV / state cache, the path generation and eval use."""
    from xllm.models.build import init_cache
    model = build(cfg, branch_scale, unscale_stream_eps=unscale_stream_eps).eval()
    tokens, _ = data(cfg)
    with torch.no_grad():
        out, _, _ = model(tokens=tokens, multi_segments=False, cache=init_cache(model))
    return out


NORMS = [dict(apply_rmsnorm=True), dict(apply_rmsnorm=False)]
ARCHS = [GEKKO, TRANSFORMER, HYBRID, TRANSFORMER_MOE, TRANSFORMER_MOVA]
ARCH_IDS = ["gekko", "transformer", "hybrid", "moe", "mova"]


@gpu
@pytest.mark.parametrize("norm", NORMS, ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("base", ARCHS, ids=ARCH_IDS)
def test_stream_vs_inbranch_norm_eps(base, norm):
    """Stream-reading norms carry base_eps * embed_multiplier^2; in-branch norms keep rmsnorm_eps."""
    cfg = replace(base, embed_multiplier=M, **norm)
    scale = M * M
    want_norm = (cfg.rmsnorm_eps if cfg.apply_rmsnorm else cfg.layernorm_eps) * scale
    want_time = cfg.timenorm_eps * scale
    model = build(cfg)

    for layer in model.layers:
        if hasattr(layer, "gda"):
            assert math.isclose(layer.gda.timenorm.eps, want_time, rel_tol=1e-12)
            assert math.isclose(layer.gda.query_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
            assert math.isclose(layer.gda.key_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
            assert math.isclose(layer.gda.attn_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
            assert math.isclose(layer.gda.adaptive_working_memory.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
        if hasattr(layer, "attention"):
            assert math.isclose(layer.attention.norm.eps, want_norm, rel_tol=1e-12)
            assert math.isclose(layer.attention.norm_eps, want_norm, rel_tol=1e-12)
            assert math.isclose(layer.attention.query_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
            assert math.isclose(layer.attention.key_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
        if hasattr(layer, "mova"):
            assert math.isclose(layer.mova.norm.eps, want_norm, rel_tol=1e-12)
            assert math.isclose(layer.mova.norm_eps, want_norm, rel_tol=1e-12)
            assert math.isclose(layer.mova.query_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
            assert math.isclose(layer.mova.key_norm.eps, cfg.rmsnorm_eps, rel_tol=1e-12)
        if hasattr(layer, "nffn"):
            assert math.isclose(layer.nffn.norm.eps, want_norm, rel_tol=1e-12)
            assert math.isclose(layer.nffn.norm_eps, want_norm, rel_tol=1e-12)
        if hasattr(layer, "moe"):
            assert math.isclose(layer.moe.norm.eps, want_norm, rel_tol=1e-12)
            assert math.isclose(layer.moe.norm_eps, want_norm, rel_tol=1e-12)

    want_final = want_time if cfg.arch in ("gekko", "hybrid") else want_norm
    assert math.isclose(model.output.final_norm.eps, want_final, rel_tol=1e-12)


@gpu
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("norm", NORMS, ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("base", ARCHS, ids=ARCH_IDS)
def test_training_matches_branch_scale(base, norm, fused):
    base = replace(base, fused_block=fused, **norm)
    want_loss, want = loss_and_grads(base, branch_scale=C)
    got_loss, got = loss_and_grads(replace(base, embed_multiplier=M))
    assert torch.equal(got_loss, want_loss), (got_loss.item(), want_loss.item())
    differ = [n for n in want if not torch.equal(got[n], want[n])]
    assert differ == [], differ

    # undoing the automatic stream-norm eps scaling breaks the equivalence
    plain_loss, plain = loss_and_grads(replace(base, embed_multiplier=M), unscale_stream_eps=True)
    assert not torch.equal(plain_loss, want_loss) or any(not torch.equal(plain[n], want[n]) for n in want)


@gpu
@pytest.mark.parametrize("norm", NORMS, ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("base", [GEKKO, TRANSFORMER], ids=["gekko", "transformer"])
def test_ortho_residual_matches_branch_scale(base, norm):
    base = replace(base, fused_block=False, residual_func="ortho", residual_heads=4, **norm)
    want_loss, want = loss_and_grads(base, branch_scale=C)
    got_loss, got = loss_and_grads(replace(base, embed_multiplier=M))
    assert torch.equal(got_loss, want_loss), (got_loss.item(), want_loss.item())
    differ = [n for n in want if not torch.equal(got[n], want[n])]
    assert differ == [], differ


@gpu
@pytest.mark.skipif(CP > 1, reason="inference does not support context parallel")
@pytest.mark.parametrize("norm", NORMS, ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("base", ARCHS, ids=ARCH_IDS)
def test_inference_matches_branch_scale(base, norm):
    base = replace(base, **norm)
    want = logits(base, branch_scale=C)
    assert torch.equal(logits(replace(base, embed_multiplier=M)), want)
    assert not torch.equal(logits(replace(base, embed_multiplier=M), unscale_stream_eps=True), want)


def test_no_stream_norm_eps_config_field():
    assert "stream_norm_eps" not in {f.name for f in fields(ModelConf)}


def test_stream_eps_scale_property(caplog):
    with caplog.at_level("WARNING"):
        cfg = ModelConf(arch="gekko", apply_rmsnorm=True, embed_multiplier=2.5)
    assert cfg.stream_eps_scale == 6.25
    assert caplog.text == ""


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), 1e200])
def test_embed_multiplier_rejects_invalid(bad):
    with pytest.raises(AssertionError, match="embed_multiplier"):
        ModelConf(arch="transformer", embed_multiplier=bad)
