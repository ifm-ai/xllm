"""Tests for Complete(d)P width/depth scaling (mup config).

Run on one GPU: pytest -q tests/test_completedp.py
"""
import copy
import json
import math
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.optim import AdamW

from xllm.config import (
    DataLoaderConfig,
    ModelConf,
    MuPConf,
    OptimConf,
    SlurmConf,
    TokenizerConf,
    TrainerConf,
    ValidConf,
    validate_mup_width_family,
)
from xllm.distributed import initialize_model_parallel
from xllm.models import build_model
from xllm.modules.residual import BaseResidual
from xllm.monitor import Timings
from xllm.optim import build_optimizer, get_group_hparams, optimizer_layout, param_roles, set_group_hparams
from xllm.utils import set_sharded_optimizer_state

# truncated normal at +-3 std has std ~0.9866 * std
TRUNC_STD = 0.98658

LR, EPS, WD = 1e-2, 1e-8, 0.1
STEPS = 100


@pytest.fixture(scope="module", autouse=True)
def distributed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(port))
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", rank=0, world_size=1)
        initialize_model_parallel(1, 1, 600)
    yield


class _Tokenizer:
    vocab_size = 512
    bos_id = 1
    eos_id = 2
    pad_id = 0


def model_conf(model_dim=64, num_layers=2, **kwargs) -> ModelConf:
    head_dim = 16
    defaults = dict(
        arch="transformer",
        ddp_backend="fsdp2",
        init_mode="gaussian",
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=model_dim // head_dim,
        head_dim=head_dim,
        ffn_hidden_dim=2 * model_dim,
        chunk_size=16,
        vocab_size=_Tokenizer.vocab_size,
        apply_rmsnorm=True,
        qknorm=True,
        rope_head_dim=head_dim,
    )
    defaults.update(kwargs)
    return ModelConf(**defaults)


def mup_conf(**kwargs) -> MuPConf:
    defaults = dict(
        base_model_dim=64,
        base_init_embed_std=1.0,
        base_init_hidden_std=0.04,
        base_init_unembed_std=0.04,
    )
    defaults.update(kwargs)
    return MuPConf(**defaults)


def optim_conf(**kwargs) -> OptimConf:
    defaults = dict(lr=LR, epsilon=EPS, weight_decay=WD, warmup=10, scheduler="cosine")
    defaults.update(kwargs)
    return OptimConf(**defaults)


def build(model_cfg: ModelConf, seed: int = 0):
    torch.manual_seed(seed)
    return build_model(
        model_cfg, dtype="fp32", fully_sharded_size=None, fp32_reduce_scatter=True,
        reshard_after_forward=True, forward_prefetch=False, tokenizer=_Tokenizer(),
    )


def resolved(model_cfg: ModelConf, mup: MuPConf):
    mup.resolve(model_cfg)
    return model_cfg, mup


def train_steps(model, optimizer, scheduler, num_steps, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(num_steps):
        tokens = torch.randint(3, _Tokenizer.vocab_size, (2, 33), device="cuda", generator=generator)
        loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
        loss.mean().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def local(p):
    return p.to_local() if hasattr(p, "to_local") else p


# ---------------------------------------------------------------------------
# config


def test_resolve_width():
    model_cfg, mup = resolved(model_conf(model_dim=256), mup_conf())
    assert mup.width_multiplier == 4.0
    assert mup.depth_multiplier == 1.0
    assert model_cfg.init_embed_std == 1.0
    assert model_cfg.init_std == pytest.approx(0.04 / 2)
    assert model_cfg.init_logits_std == pytest.approx(0.04 / 4)
    assert model_cfg.residual_multiplier == 1.0
    assert not model_cfg.zero_query
    assert not model_cfg.zero_readout


def test_resolve_depth():
    model_cfg, mup = resolved(model_conf(num_layers=8), mup_conf(base_num_layers=2, alpha=0.5))
    assert mup.depth_multiplier == 4.0
    assert model_cfg.residual_multiplier == pytest.approx(0.5)
    model_cfg, _ = resolved(model_conf(num_layers=8), mup_conf(base_num_layers=2, alpha=1.0))
    assert model_cfg.residual_multiplier == pytest.approx(0.25)


@pytest.mark.parametrize("model_kwargs, mup_kwargs", [
    ({}, {"base_num_layers": 2}),
    ({}, {"alpha": 1.0}),
    ({}, {"base_num_layers": 2, "alpha": 0.4}),
    ({}, {"base_num_layers": 2, "alpha": 1.1}),
    ({}, {"base_model_dim": 0}),
    ({}, {"base_init_hidden_std": -1.0}),
    ({"arch": "gekko"}, {}),
    ({"ddp_backend": "fsdp1"}, {}),
    ({"init_mode": "he"}, {}),
    ({"scale_emb": True}, {}),
    ({"residual_func": "ortho", "residual_heads": 4}, {}),
    ({"num_experts": 4, "num_activated_experts": 1, "expert_inter_dim": 32}, {}),
    ({"fused_block": True, "causal_attn_backend": "flash"}, {"base_num_layers": 2, "alpha": 1.0}),
    ({"qknorm": True}, {"zero_query": True}),
])
def test_resolve_rejects_unsupported(model_kwargs, mup_kwargs):
    with pytest.raises(ValueError):
        mup_conf(**mup_kwargs).resolve(model_conf(**model_kwargs))


@pytest.mark.parametrize("name", ["init_embed_std", "init_std", "init_logits_std"])
def test_resolve_rejects_conflicting_init_std(name):
    with pytest.raises(ValueError, match=name):
        mup_conf().resolve(model_conf(model_dim=256, **{name: 0.5}))


def test_resolve_accepts_matching_init_std():
    # a config saved by a mup run holds the resolved stds; resolving it again must work
    model_cfg, _ = resolved(model_conf(model_dim=256), mup_conf())
    again, _ = resolved(copy.deepcopy(model_cfg), mup_conf())
    assert (again.init_embed_std, again.init_std, again.init_logits_std) == \
        (model_cfg.init_embed_std, model_cfg.init_std, model_cfg.init_logits_std)


def test_width_only_allows_fused_block_and_dropout():
    model_cfg, _ = resolved(model_conf(fused_block=True, causal_attn_backend="flash", dropout=0.1), mup_conf())
    assert model_cfg.residual_multiplier == 1.0


def test_depth_scaling_allows_dropout():
    model_kwargs = dict(num_layers=8, dropout=0.1, hidden_dropout=0.1, attention_dropout=0.1)
    model_cfg, _ = resolved(model_conf(**model_kwargs), mup_conf(base_num_layers=2, alpha=1.0))
    assert model_cfg.residual_multiplier == pytest.approx(0.25)


def trainer_conf_from_args(args):
    base = TrainerConf(
        tokenizer=TokenizerConf(), slurm=SlurmConf(), model=ModelConf(), optim=OptimConf(),
        valid=ValidConf(), dataloader=DataLoaderConfig(),
    )
    arg_dict = vars(TrainerConf.to_cli().parse_args(args))
    arg_dict.pop("cfg")
    return TrainerConf.from_cli(arg_dict=arg_dict, default_instance=base)


def test_cli_without_mup_is_unchanged():
    cfg = trainer_conf_from_args(["--model.init_mode", "gaussian"])
    assert cfg.mup is None
    assert cfg.model.init_std is None
    assert cfg.model.residual_multiplier == 1.0


def test_cli_with_mup():
    cfg = trainer_conf_from_args([
        "--model.init_mode", "gaussian", "--model.model_dim", "1024", "--model.num_layers", "16",
        "--mup.base_model_dim", "256", "--mup.base_init_embed_std", "1.0",
        "--mup.base_init_hidden_std", "0.02", "--mup.base_init_unembed_std", "0.02",
        "--mup.base_num_layers", "4", "--mup.alpha", "1.0",
        "--mup.zero_query", "true", "--mup.zero_readout", "true",
    ])
    assert cfg.mup.width_multiplier == 4.0
    assert cfg.mup.depth_multiplier == 4.0
    assert cfg.model.init_std == pytest.approx(0.01)
    assert cfg.model.init_logits_std == pytest.approx(0.005)
    assert cfg.model.residual_multiplier == pytest.approx(0.25)
    assert cfg.model.zero_query
    assert cfg.model.zero_readout


def test_width_family():
    base = model_conf(model_dim=64, num_kv_heads=2)
    validate_mup_width_family(base, model_conf(model_dim=256, num_kv_heads=8))
    with pytest.raises(ValueError):
        validate_mup_width_family(base, model_conf(model_dim=256, num_kv_heads=2))
    with pytest.raises(ValueError):
        validate_mup_width_family(base, model_conf(model_dim=256, num_kv_heads=8, head_dim=32, num_heads=8))
    with pytest.raises(ValueError):
        validate_mup_width_family(base, model_conf(model_dim=256, num_kv_heads=8, ffn_hidden_dim=256))
    with pytest.raises(ValueError):
        validate_mup_width_family(base, model_conf(model_dim=256, num_kv_heads=8, vocab_size=1024))


# ---------------------------------------------------------------------------
# model


def test_param_roles_cover_every_parameter_once():
    model_cfg, _ = resolved(model_conf(num_layers=3), mup_conf())
    model = build(model_cfg)
    roles = param_roles(model)
    counts = {role: len(params) for role, params in roles.items()}
    # per layer: wq, wk, wv, wo, fc1, fc2 / attention norm, ffn norm, query norm (key norm has no weight)
    assert counts == {"embed": 1, "hidden_matrix": 18, "hidden_vector": 9, "final_norm": 1, "readout": 1}
    assert roles["readout"][0] is model.output.output.weight
    assert all(p.ndim == 2 for p in roles["hidden_matrix"])
    assert all(p.ndim == 1 for p in roles["hidden_vector"])
    ids = [id(p) for params in roles.values() for p in params]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())


def test_init_std_scaling():
    model_cfg, _ = resolved(model_conf(model_dim=256), mup_conf())
    model = build(model_cfg)
    roles = param_roles(model)
    def std(params):
        return torch.cat([local(p).detach().flatten() for p in params]).std().item()
    assert std(roles["embed"]) == pytest.approx(1.0 * TRUNC_STD, rel=0.03)
    assert std(roles["hidden_matrix"]) == pytest.approx(0.04 / 2 * TRUNC_STD, rel=0.03)
    assert std(roles["readout"]) == pytest.approx(0.04 / 4 * TRUNC_STD, rel=0.03)
    # norm gains are parameterized as 1 + weight, so a unit gain is a zero weight
    for p in roles["hidden_vector"] + roles["final_norm"]:
        assert torch.all(local(p) == 0)


def test_zero_query():
    model_cfg, _ = resolved(model_conf(qknorm=False), mup_conf(zero_query=True))
    model = build(model_cfg)
    assert torch.any(local(model.output.output.weight) != 0)
    for layer in model.layers:
        assert torch.all(local(layer.attention.wq.weight) == 0)
        assert torch.any(local(layer.attention.wk.weight) != 0)


@pytest.mark.parametrize("qknorm", [False, True])
def test_zero_readout(qknorm):
    model_cfg, _ = resolved(model_conf(qknorm=qknorm), mup_conf(zero_readout=True))
    model = build(model_cfg)
    assert torch.all(local(model.output.output.weight) == 0)
    for layer in model.layers:
        assert torch.any(local(layer.attention.wq.weight) != 0)


def test_residual_multiplier():
    h, residual = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    assert torch.equal(BaseResidual(0, 8)(h, residual), h + residual)
    assert torch.allclose(BaseResidual(0, 8, 0.25)(h, residual), 0.25 * h + residual)

    model_cfg, _ = resolved(model_conf(num_layers=8), mup_conf(base_num_layers=2, alpha=0.5))
    model = build(model_cfg)
    for layer in model.layers:
        assert layer.attention.residual.multiplier == pytest.approx(0.5)
        assert layer.nffn.residual.multiplier == pytest.approx(0.5)

    # a zero multiplier removes both branches, so every block is the identity
    model = build(model_conf(residual_multiplier=0.0))
    x = torch.randn(2, 32, 64, device="cuda")
    freqs = model.rope.get_freqs_cis(0, 32, x.device)
    for layer in model.layers:
        out, _, _ = layer(x, freqs)
        assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# optimizer


def groups_by_name(optimizer):
    return {group["name"]: group for group in optimizer.param_groups}


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_mup_param_groups_match_table(alpha):
    # m_N = 4, m_L = 2
    model_cfg, mup = resolved(model_conf(model_dim=256, num_layers=4), mup_conf(base_num_layers=2, alpha=alpha))
    model = build(model_cfg)
    optimizer, _ = build_optimizer(model, optim_conf(), STEPS, "fp32", mup=mup)
    groups = groups_by_name(optimizer)
    assert list(groups) == ["embed", "hidden_matrix", "hidden_vector", "final_norm", "readout"]
    m_n, m_l = 4.0, 2.0
    expected = {
        "embed": (LR, EPS / m_n, WD),
        "hidden_matrix": (LR / m_n * m_l ** (alpha - 1), EPS / m_n * m_l ** -alpha, WD * m_n),
        "hidden_vector": (LR * m_l ** (alpha - 1), EPS / m_n * m_l ** -alpha, WD),
        "final_norm": (LR, EPS, WD),
        "readout": (LR / m_n, EPS, WD * m_n),
    }
    for name, (lr, eps, wd) in expected.items():
        group = groups[name]
        assert group["initial_lr"] == pytest.approx(lr, rel=1e-12), name
        assert group["eps"] == pytest.approx(eps, rel=1e-12), name
        assert group["weight_decay"] == pytest.approx(wd, rel=1e-12), name


def test_mup_no_decay_embed_readout():
    model_cfg, mup = resolved(model_conf(model_dim=256), mup_conf())
    model = build(model_cfg)
    optimizer, _ = build_optimizer(model, optim_conf(no_decay_embed_readout=True), STEPS, "fp32", mup=mup)
    groups = groups_by_name(optimizer)
    assert groups["embed"]["weight_decay"] == 0.0
    assert groups["readout"]["weight_decay"] == 0.0
    assert groups["hidden_matrix"]["weight_decay"] == pytest.approx(WD * 4)
    assert groups["hidden_vector"]["weight_decay"] == WD
    assert groups["final_norm"]["weight_decay"] == WD


def test_tau_epoch():
    tau = 0.2
    model_cfg, mup = resolved(model_conf(model_dim=256, num_layers=4), mup_conf(base_num_layers=2, alpha=1.0))
    model = build(model_cfg)
    optimizer, _ = build_optimizer(model, optim_conf(tau_epoch=tau, weight_decay=123.0), STEPS, "fp32", mup=mup)
    base_wd = 1.0 / (LR * tau * STEPS)
    groups = groups_by_name(optimizer)
    assert groups["embed"]["weight_decay"] == pytest.approx(base_wd)
    assert groups["hidden_matrix"]["weight_decay"] == pytest.approx(base_wd * 4)
    # at alpha = 1, lr * weight_decay * steps = 1 / tau for every role
    for name, group in groups.items():
        assert group["initial_lr"] * group["weight_decay"] * STEPS == pytest.approx(1.0 / tau), name

    # without mup, the single group uses the tau_epoch weight decay
    optimizer, _ = build_optimizer(build(model_conf()), optim_conf(tau_epoch=tau), STEPS, "fp32")
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(base_wd)


def test_without_mup_optimizer_is_unchanged():
    model = build(model_conf())
    cfg = optim_conf()
    optimizer, _ = build_optimizer(model, cfg, STEPS, "fp32")
    reference = AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, eps=cfg.epsilon)
    assert len(optimizer.param_groups) == 1
    group, ref_group = optimizer.param_groups[0], reference.param_groups[0]
    assert [id(p) for p in group["params"]] == [id(p) for p in ref_group["params"]]
    assert group["initial_lr"] == cfg.lr
    assert {k: v for k, v in group.items() if k not in ("params", "initial_lr", "lr")} == \
        {k: v for k, v in ref_group.items() if k not in ("params", "lr")}


def test_no_decay_without_mup():
    model = build(model_conf())
    optimizer, _ = build_optimizer(model, optim_conf(no_decay_embed_readout=True), STEPS, "fp32")
    groups = groups_by_name(optimizer)
    no_decay = {id(p) for p in groups["no_decay"]["params"]}
    assert no_decay == {id(model.embed.weight), id(model.output.output.weight)}
    assert groups["no_decay"]["weight_decay"] == 0.0
    assert groups["decay"]["weight_decay"] == WD
    assert len(groups["decay"]["params"]) + 2 == len(list(model.parameters()))


@pytest.mark.parametrize("clip,scale,expected", [
    (1.0, 1.0, 1.0),
    (1.0, 4.0, 4.0),
    (0.0, 4.0, math.inf),
    (-1.0, 4.0, math.inf),
])
def test_clip_max_norm(clip, scale, expected):
    import train

    assert train.get_clip_max_norm(clip, scale) == expected


def test_infinite_clip_threshold_leaves_gradients_unchanged():
    model = build(model_conf())
    tokens = torch.randint(3, _Tokenizer.vocab_size, (2, 33), device="cuda")
    loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
    loss.mean().backward()
    grads = [local(p.grad).clone() for p in model.parameters()]
    expected_norm = torch.cat([g.flatten() for g in grads]).norm().item()
    norm = model.clip_grad_norm_(math.inf)
    assert norm == pytest.approx(expected_norm, rel=1e-5)
    assert all(torch.equal(local(p.grad), g) for p, g in zip(model.parameters(), grads))


# ---------------------------------------------------------------------------
# checkpoint resume


def test_resume_keeps_per_role_learning_rates():
    import train

    model_cfg, mup = resolved(model_conf(model_dim=128, num_layers=4), mup_conf(base_num_layers=2, alpha=0.5))
    cfg = optim_conf()

    # uninterrupted: 10 + 10 steps
    model = build(model_cfg, seed=1)
    optimizer, scheduler = build_optimizer(model, cfg, STEPS, "fp32", mup=mup)
    train_steps(model, optimizer, scheduler, 10, seed=2)
    model_state = {k: local(v).clone() for k, v in model.state_dict().items()}
    optim_state = copy.deepcopy(optimizer.state_dict())
    training_state = {
        "scheduler": copy.deepcopy(scheduler.state_dict()), "step": 10, "scale": 1.0, "scale_updates": 0,
        "clip_cumulative": 0, "data_state": None, "timings": {},
    }
    train_steps(model, optimizer, scheduler, 10, seed=3)
    expected_lrs = [group["lr"] for group in optimizer.param_groups]
    expected = {k: local(v).clone() for k, v in model.state_dict().items()}

    # resumed: rebuild, reload the step-10 states as train.py does, then 10 steps
    resumed = build(model_cfg, seed=4)
    with torch.no_grad():
        for k, v in resumed.state_dict().items():
            local(v).copy_(model_state[k])
    optimizer, scheduler = build_optimizer(resumed, cfg, STEPS, "fp32", mup=mup)
    configured = get_group_hparams(optimizer)
    set_sharded_optimizer_state(resumed, optimizer, optim_state, dcp=False)
    set_group_hparams(optimizer, configured)
    data_iterator = SimpleNamespace(set_state=lambda state: None)
    train.set_optim_state(scheduler, training_state, SimpleNamespace(), data_iterator, Timings())
    train_steps(resumed, optimizer, scheduler, 10, seed=3)

    assert [group["lr"] for group in optimizer.param_groups] == expected_lrs
    for k, v in resumed.state_dict().items():
        assert torch.equal(local(v), expected[k]), k


def test_resume_uses_configured_group_hparams():
    model_cfg, mup = resolved(model_conf(model_dim=128), mup_conf())
    model = build(model_cfg)
    optimizer, scheduler = build_optimizer(model, optim_conf(), STEPS, "fp32", mup=mup)
    train_steps(model, optimizer, scheduler, 2, seed=0)
    optim_state = copy.deepcopy(optimizer.state_dict())

    optimizer, _ = build_optimizer(model, optim_conf(epsilon=1e-10, weight_decay=0.2), STEPS, "fp32", mup=mup)
    configured = get_group_hparams(optimizer)
    set_sharded_optimizer_state(model, optimizer, optim_state, dcp=False)
    assert groups_by_name(optimizer)["hidden_matrix"]["weight_decay"] == pytest.approx(0.1 * 2)
    set_group_hparams(optimizer, configured)
    groups = groups_by_name(optimizer)
    assert groups["hidden_matrix"]["weight_decay"] == pytest.approx(0.2 * 2)
    assert groups["embed"]["eps"] == pytest.approx(1e-10 / 2)


MUP_ARGS = [
    "--model.init_mode", "gaussian", "--model.model_dim", "256",
    "--mup.base_model_dim", "64", "--mup.base_init_embed_std", "1.0",
    "--mup.base_init_hidden_std", "0.02", "--mup.base_init_unembed_std", "0.02",
]
NO_DECAY_ARGS = ["--model.init_mode", "gaussian", "--optim.no_decay_embed_readout", "true"]


def test_optimizer_layout():
    assert optimizer_layout(trainer_conf_from_args(["--model.init_mode", "gaussian"]).to_dict()) == "default"
    assert optimizer_layout(trainer_conf_from_args(NO_DECAY_ARGS).to_dict()) == "no_decay_embed_readout"
    assert optimizer_layout(trainer_conf_from_args(MUP_ARGS + ["--optim.no_decay_embed_readout", "true"]).to_dict()) == "mup"
    # config.json written before mup existed
    assert optimizer_layout({"optim": {"lr": 1e-3}}) == "default"


@pytest.mark.parametrize("saved_args, current_args, ok", [
    ([], [], True),
    (MUP_ARGS, MUP_ARGS, True),
    ([], MUP_ARGS, False),
    (MUP_ARGS, [], False),
    (MUP_ARGS, NO_DECAY_ARGS, False),
    (NO_DECAY_ARGS, [], False),
])
def test_check_optimizer_layout(tmp_path, saved_args, current_args, ok):
    import train

    saved = trainer_conf_from_args(["--model.init_mode", "gaussian"] + saved_args)
    (tmp_path / "config.json").write_text(saved.to_json())
    current = trainer_conf_from_args(["--model.init_mode", "gaussian"] + current_args)
    if ok:
        train.check_optimizer_layout(current, tmp_path)
    else:
        with pytest.raises(ValueError, match="param group layout"):
            train.check_optimizer_layout(current, tmp_path)


def test_check_optimizer_layout_without_config_json(tmp_path):
    import train

    train.check_optimizer_layout(trainer_conf_from_args(MUP_ARGS), tmp_path)
