"""The depth factor on the embedding equals the depth factor on the branches, with model.stream_norm_eps. GPU.

A model whose branch outputs are scaled by c (each block's wo and fc2 weights × c), and the same model with
embed_multiplier m = 1/c, stream_norm_eps = (rmsnorm or layernorm eps)·m² and (Gekko) timenorm_eps·m²: the second
model's residual stream is the first's × m, every read of the stream is a norm, and with the eps scaled by m² each
norm computes the same output. With c a power of 2 the loss, every gradient and the inference logits match bit for
bit, for RMSNorm and LayerNorm, fused and unfused. Without the eps scaling they don't.

    PYTHONPATH=. pytest -q tests/test_stream_norm_eps.py                                   # one GPU
    PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -q tests/test_stream_norm_eps.py    # tensor parallel 2
    STREAM_EPS_PARALLEL=cp PYTHONPATH=. torchrun --nproc_per_node=2 -m pytest -q tests/test_stream_norm_eps.py
                                                                                           # context parallel 2
"""
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from xllm.config import ModelConf

C = 0.5  # the depth factor; m = 1/C
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
    swiglu=True, qknorm=True, apply_rmsnorm=True, causal_attn_backend="flash", init_mode="he", vocab_size=1024,
    ddp_backend="fsdp2",
)
BRANCH_OUT = ("gda.wo.weight", "attention.wo.weight", "nffn.fc2.weight")
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


def exact(cfg):
    """cfg with the depth factor on the embedding and the stream norms' eps scaled to match."""
    m = 1 / C
    eps = cfg.rmsnorm_eps if cfg.apply_rmsnorm else cfg.layernorm_eps
    return replace(cfg, embed_multiplier=m, stream_norm_eps=eps * m * m,
                   timenorm_eps=cfg.timenorm_eps * m * m)  # timenorm_eps: Gekko only


def build(cfg, branch_scale):
    from xllm.models.gekko import Gekko
    from xllm.models.transformer import Transformer
    torch.manual_seed(0)
    model = (Gekko if cfg.arch == "gekko" else Transformer)(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(BRANCH_OUT):
                p.mul_(branch_scale)
    return model.cuda().to(torch.bfloat16)


def data(cfg):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(3, cfg.vocab_size, (1, SEQ), generator=g).cuda()
    targets = torch.randint(3, cfg.vocab_size, (1, SEQ), generator=g).cuda()
    return tokens, targets


def loss_and_grads(cfg, branch_scale=1.0):
    model = build(cfg, branch_scale).train()
    tokens, targets = data(cfg)
    loss, _, _ = model(tokens=tokens, targets=targets, multi_segments=False)
    loss = loss.float().sum()
    loss.backward()
    # dL/d(c·W) = dL/dW / c: compare the branch model's wo / fc2 gradients times c
    grads = {n: p.grad * (branch_scale if n.endswith(BRANCH_OUT) else 1.0) for n, p in model.named_parameters()}
    return loss, grads


def logits(cfg, branch_scale=1.0):
    """Inference: eval mode with the KV / state cache, the path generation and eval use."""
    from xllm.models.build import init_cache
    model = build(cfg, branch_scale).eval()
    tokens, _ = data(cfg)
    with torch.no_grad():
        out, _, _ = model(tokens=tokens, multi_segments=False, cache=init_cache(model))
    return out


NORMS = [dict(apply_rmsnorm=True), dict(apply_rmsnorm=False)]
ARCHS = [GEKKO, TRANSFORMER]


@gpu
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("norm", NORMS, ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("base", ARCHS, ids=["gekko", "transformer"])
def test_training_matches_branch_scale(base, norm, fused):
    base = replace(base, fused_block=fused, **norm)
    want_loss, want = loss_and_grads(base, branch_scale=C)
    got_loss, got = loss_and_grads(exact(base))
    assert torch.equal(got_loss, want_loss), (got_loss.item(), want_loss.item())
    differ = [n for n in want if not torch.equal(got[n], want[n])]
    assert differ == [], differ

    # the same model with the eps left as they are is a slightly different model
    plain_loss, plain = loss_and_grads(replace(base, embed_multiplier=1 / C))
    assert not torch.equal(plain_loss, want_loss) or any(not torch.equal(plain[n], want[n]) for n in want)


@gpu
@pytest.mark.skipif(CP > 1, reason="inference does not support context parallel")
@pytest.mark.parametrize("norm", NORMS, ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("base", ARCHS, ids=["gekko", "transformer"])
def test_inference_matches_branch_scale(base, norm):
    base = replace(base, **norm)
    want = logits(base, branch_scale=C)
    assert torch.equal(logits(exact(base)), want)
    assert not torch.equal(logits(replace(base, embed_multiplier=1 / C)), want)


@pytest.mark.parametrize("moe", [dict(num_experts=4, num_activated_experts=2),
                                 dict(num_experts=4, num_activated_experts=2, num_values=4, num_activated_values=2)],
                         ids=["moe", "mova"])
def test_stream_norm_eps_rejects_moe(moe):
    with pytest.raises(AssertionError, match="MoE / MoVA"):
        ModelConf(arch="transformer", stream_norm_eps=1e-6, **moe)


def test_stream_norm_eps_rejects_ortho():
    with pytest.raises(AssertionError, match="ortho"):
        ModelConf(arch="transformer", stream_norm_eps=1e-6, residual_func="ortho", residual_heads=4)


# the warnings: embed_multiplier is only a reparameterization when every eps that reads the stream carries m^2
M = 2.0
MULT = dict(apply_rmsnorm=True, embed_multiplier=M)
SCALED = ModelConf().rmsnorm_eps * M * M


def test_warns_without_stream_norm_eps(caplog):
    with caplog.at_level("WARNING"):
        ModelConf(arch="transformer", **MULT)
    assert "without model.stream_norm_eps" in caplog.text


def test_warns_when_stream_norm_eps_does_not_match_embed_multiplier(caplog):
    with caplog.at_level("WARNING"):
        ModelConf(arch="transformer", stream_norm_eps=SCALED * 3, **MULT)
    assert "embed_multiplier^2" in caplog.text


def test_no_warning_for_transformer_when_scaled(caplog):
    with caplog.at_level("WARNING"):
        ModelConf(arch="transformer", stream_norm_eps=SCALED, **MULT)
    assert caplog.text == ""


def test_warns_when_gekko_timenorm_eps_left_at_default(caplog):
    """The gap this guards: the stream norms are scaled but Gekko's timestep norms, which also read the stream,
    are not. The old `elif` could not warn here, because stream_norm_eps was set."""
    with caplog.at_level("WARNING"):
        ModelConf(arch="gekko", stream_norm_eps=SCALED, **MULT)
    assert "timenorm_eps" in caplog.text


def test_no_warning_for_gekko_when_timenorm_eps_scaled_too(caplog):
    with caplog.at_level("WARNING"):
        ModelConf(arch="gekko", stream_norm_eps=SCALED, timenorm_eps=ModelConf().timenorm_eps * M * M, **MULT)
    assert caplog.text == ""
