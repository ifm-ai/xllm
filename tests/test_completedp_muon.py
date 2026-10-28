"""Complete(d)P with Muon and the depth scale c folded into wo / fc2 (MuPConf, xllm/optim/mup.py). CPU.

    pytest tests/test_completedp_muon.py

Expected values are written out by hand, not recomputed with the formulas under test.
"""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from xllm.config import (
    DataLoaderConfig, ModelConf, MuPConf, OptimConf, SlurmConf, TokenizerConf, TrainerConf, ValidConf,
)


def mup_conf(**kw):
    """m_N = 512 / 256 = 2, m_L = 16 / 4 = 4 with model_conf(); alpha 0.5 gives c = 4^-0.5 = 0.5."""
    return MuPConf(**dict(base_model_dim=256, base_num_layers=4, base_init_hidden_std=0.02,
                          base_init_embed_std=0.5, base_init_unembed_std=0.04, alpha=0.5) | kw)


def model_conf(**kw):
    return ModelConf(**dict(model_dim=512, num_layers=16, num_heads=8, ffn_hidden_dim=1024,
                            init_mode="gaussian") | kw)


def test_resolve_sets_multipliers_and_init_stds():
    model, mup = model_conf(), mup_conf()
    mup.resolve(model, OptimConf(optimizer="muon"))
    assert (mup.width_multiplier, mup.depth_multiplier, mup.branch_scale) == (2.0, 4.0, 0.5)
    assert math.isclose(model.init_std, 0.01414213562373095, rel_tol=1e-12)  # 0.02 / sqrt(2)
    assert math.isclose(model.init_out_std, 0.007071067811865475, rel_tol=1e-12)  # x c
    assert model.init_embed_std == 0.5
    assert model.init_logits_std == 0.02  # 0.04 / 2

    model, mup = model_conf(), mup_conf(alpha=1.0)
    mup.resolve(model, OptimConf(optimizer="muon"))
    assert mup.branch_scale == 0.25  # L0 / L


def test_resolve_is_idempotent_but_rejects_conflicting_init():
    model, mup = model_conf(), mup_conf()
    mup.resolve(model, OptimConf(optimizer="muon"))
    mup.resolve(model, OptimConf(optimizer="muon"))  # a reloaded config.json holds the resolved values
    with pytest.raises(ValueError, match="model.init_std"):
        mup_conf().resolve(model_conf(init_std=0.02), OptimConf(optimizer="muon"))


@pytest.mark.parametrize("model_kw, optim_kw, mup_kw, match", [
    ({"init_mode": "he"}, {}, {}, "init_mode"),
    ({"arch": "gekko"}, {}, {}, "arch"),
    ({"num_experts": 4, "num_activated_experts": 1, "expert_inter_dim": 64}, {}, {}, "num_experts"),
    ({"residual_func": "ortho", "residual_heads": 8}, {}, {}, "residual_func"),
    ({"scale_emb": True}, {}, {}, "scale_emb"),
    ({}, {"optimizer": "adamw"}, {}, "optimizer"),
    ({}, {"muon_adjust_lr": "rms_norm"}, {}, "muon_adjust_lr"),
    ({}, {"muon_block_scale": "matrix"}, {}, "muon_block_scale"),
    ({}, {}, {"alpha": 0.4}, "alpha"),
    ({}, {}, {"base_num_layers": 0}, "base_num_layers"),
])
def test_resolve_rejects_unsupported(model_kw, optim_kw, mup_kw, match):
    with pytest.raises(ValueError, match=match):
        mup_conf(**mup_kw).resolve(model_conf(**model_kw), OptimConf(**dict(optimizer="muon") | optim_kw))


def trainer_conf(**kw):
    return TrainerConf(slurm=SlurmConf(), model=model_conf(), optim=OptimConf(optimizer="muon"), valid=ValidConf(),
                       tokenizer=TokenizerConf(), dataloader=DataLoaderConfig(), dcp_for_optimizer=True, **kw)


def test_trainer_conf_resolves_from_cli_and_config_json():
    args = ["--slurm.partition", "main", "--optim.optimizer", "muon", "--dcp_for_optimizer", "true", "--model.model_dim", "512",
            "--model.num_layers", "16", "--model.num_heads", "8", "--model.init_mode", "gaussian",
            "--mup.base_model_dim", "256", "--mup.base_num_layers", "4", "--mup.alpha", "0.5",
            "--mup.base_init_hidden_std", "0.02", "--mup.base_init_embed_std", "0.5",
            "--mup.base_init_unembed_std", "0.04"]
    cfg = TrainerConf.from_cli(vars(TrainerConf.to_cli().parse_args(args)))
    assert cfg.mup.branch_scale == 0.5
    assert math.isclose(cfg.model.init_out_std, 0.007071067811865475, rel_tol=1e-12)
    # train.py writes config.json with to_dict(); reloading resolves again to the same values
    reloaded = TrainerConf.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert reloaded.mup == cfg.mup and reloaded.model == cfg.model
    assert trainer_conf().mup is None and trainer_conf().model.init_out_std is None


def test_out_projection_init_is_c_times_plain_init():
    """The real Transformer: wo / fc2 are exactly c x the init without mup's c (c a power of two), the rest equal."""
    from xllm.models.transformer import Transformer
    cfg = model_conf(model_dim=128, num_layers=4, num_heads=4, ffn_hidden_dim=256, vocab_size=64,
                     apply_rmsnorm=True, qknorm=True, swiglu=True)
    mup = mup_conf(base_model_dim=64, base_num_layers=1, alpha=1.0)
    mup.resolve(cfg, OptimConf(optimizer="muon"))
    assert mup.branch_scale == 0.25
    plain = copy.deepcopy(cfg)
    plain.init_out_std = None
    tokenizer = SimpleNamespace(bos_id=1, eos_id=2, pad_id=0)
    torch.manual_seed(0)
    folded = Transformer(cfg, tokenizer)
    torch.manual_seed(0)
    reference = Transformer(plain, tokenizer)
    n_out = 0
    for (name, a), (_, b) in zip(reference.named_parameters(), folded.named_parameters()):
        if name.endswith(("attention.wo.weight", "nffn.fc2.weight")):
            n_out += 1
            assert torch.equal(b, a * 0.25), name
        else:
            assert torch.equal(b, a), name
    assert n_out == 2 * 4


# ---- optimizer: Muon on CPU (single rank, as in tests/test_muon_missing_grad.py) ----

import torch.nn.functional as F
from torch import nn

import xllm.optim.muon as muon
from xllm.optim import mup as mup_lib
from xllm.optim import restore_optimizer_config
from xllm.optim.muon_spec import register_muon


class LocalMesh:
    def get_group(self):
        return None

    def size(self):
        return 1

    def get_local_rank(self):
        return 0


@pytest.fixture
def cpu_transport(monkeypatch):
    # the real gather / scatter and update code on CPU: only the CUDA allocation, the compiled kernel
    # wrapper and the single-rank transport are replaced
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(muon, "newton_schulz5", muon.newton_schulz5._torchdynamo_orig_callable)
    return lambda opt: monkeypatch.setattr(opt, "_exchange", lambda send, shapes: lambda: send)


# m_N = 2, m_L = 4, alpha = 0.5: c = 0.5 and m_L^(alpha - 1) = 0.5
CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, epsilon=1e-8, weight_decay=0.1, muon_weight_decay=0.05)


def resolved_mup(**kw):
    mup = mup_conf(**kw)
    mup.resolve(model_conf(), OptimConf(optimizer="muon"))
    return mup


def build_muon_cpu(model, cfg, mup, cpu_transport):
    """build_muon without FSDP2: the same split, groups and update scales."""
    muon_params, specs, _, adamw_params = muon.split_muon_params(model)
    muon_group = update_scales = None
    if mup is not None:
        muon_group, adamw_params, update_scales = mup_lib.param_groups(model, cfg, mup, muon_params)
    opt = muon.Muon(muon_params, specs, adamw_params, LocalMesh(), cfg, muon_group=muon_group,
                    update_scales=update_scales)
    cpu_transport(opt)
    return opt


def transformer(num_layers=4):
    from xllm.models.transformer import Transformer
    cfg = model_conf(model_dim=64, num_layers=num_layers, num_heads=4, ffn_hidden_dim=128, vocab_size=32,
                     apply_rmsnorm=True, qknorm=True, swiglu=True, init_std=0.02)
    torch.manual_seed(0)
    return Transformer(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))


def test_param_groups_on_transformer(cpu_transport):
    model = transformer()
    opt = build_muon_cpu(model, CFG, resolved_mup(), cpu_transport)
    groups = {g["role"]: g for g in opt.param_groups}
    # role: (lr, eps, weight_decay, tensors)
    want = {
        "muon": (0.01, 1e-8, 0.05, 4 * 7),  # wq wk wv wo fc1 fc2 fc3
        "embed": (1e-3, 5e-9, 0.1, 1),
        "hidden_vector": (5e-4, 2.5e-9, 0.1, 4 * 3),  # attention norm, query_norm, FFN norm
        "final_norm": (1e-3, 1e-8, 0.1, 1),
        "readout": (5e-4, 1e-8, 0.2, 1),
    }
    assert list(groups) == list(want)
    for role, (lr, eps, wd, n) in want.items():
        g = groups[role]
        assert math.isclose(g["lr"], lr, rel_tol=1e-12), role
        assert math.isclose(g["eps"], eps, rel_tol=1e-12), role
        assert math.isclose(g["weight_decay"], wd, rel_tol=1e-12), role
        assert len(g["params"]) == n, role
        assert g["algorithm"] == ("muon" if role == "muon" else "adamw")
    assert groups["muon"]["momentum"] == CFG.muon_momentum and groups["embed"]["betas"] == (CFG.beta1, CFG.beta2)
    assert groups["embed"]["params"][0] is model.embed.weight
    assert groups["readout"]["params"][0] is model.output.output.weight
    out = {id(layer.attention.wo.weight) for layer in model.layers} | {id(layer.nffn.fc2.weight) for layer in model.layers}
    assert [s for p, s in zip(groups["muon"]["params"], opt.update_scales) if id(p) in out] == [0.5] * 8
    assert [s for p, s in zip(groups["muon"]["params"], opt.update_scales) if id(p) not in out] == [1.0] * 20


def test_param_groups_reject_unassigned_matrices(cpu_transport):
    model = transformer(num_layers=1)
    model.layers[0].attention.extra = nn.Parameter(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="layers.0.attention.extra"):
        build_muon_cpu(model, CFG, resolved_mup(), cpu_transport)
    model = transformer(num_layers=1)
    del vars(model.layers[0].attention.wo)["muon_specs"]
    with pytest.raises(ValueError, match="wo and FFN fc2 must be Muon"):
        build_muon_cpu(model, CFG, resolved_mup(), cpu_transport)


def test_mup_at_base_size_matches_plain_muon(cpu_transport):
    """m_N = m_L = 1: every factor is 1, so mup's groups step exactly like the two plain groups."""
    plain, scaled = transformer(), transformer()
    mup = mup_conf(base_model_dim=512, base_num_layers=16)
    mup.resolve(model_conf(), OptimConf(optimizer="muon"))
    assert (mup.width_multiplier, mup.depth_multiplier, mup.branch_scale) == (1.0, 1.0, 1.0)
    opts = [build_muon_cpu(plain, CFG, None, cpu_transport), build_muon_cpu(scaled, CFG, mup, cpu_transport)]
    assert [len(o.param_groups) for o in opts] == [2, 5]
    torch.manual_seed(1)
    for _ in range(3):
        grads = [torch.randn_like(p) for p in plain.parameters()]
        for model, opt in zip((plain, scaled), opts):
            for p, g in zip(model.parameters(), grads):
                p.grad = g.clone()
            opt.step()
    for (name, a), b in zip(plain.named_parameters(), scaled.parameters()):
        assert torch.equal(a, b), name


def test_restore_applies_role_values(cpu_transport):
    mup = resolved_mup()
    opt = build_muon_cpu(transformer(num_layers=1), CFG, mup, cpu_transport)
    for group in opt.param_groups:  # what an optimizer state reload from another config would leave
        group.update(lr=1.0, initial_lr=1.0, eps=1.0, weight_decay=1.0)
    cfg = OptimConf(**dict(vars(CFG), muon_lr=0.04, lr=2e-3))
    restore_optimizer_config(opt, cfg, mup)
    groups = {g["role"]: g for g in opt.param_groups}
    for role, (initial_lr, eps, wd) in {"muon": (0.02, 1e-8, 0.05), "embed": (2e-3, 5e-9, 0.1),
                                        "hidden_vector": (1e-3, 2.5e-9, 0.1), "final_norm": (2e-3, 1e-8, 0.1),
                                        "readout": (1e-3, 1e-8, 0.2)}.items():
        g = groups[role]
        assert math.isclose(g["initial_lr"], initial_lr, rel_tol=1e-12), role
        assert math.isclose(g["eps"], eps, rel_tol=1e-12), role
        assert math.isclose(g["weight_decay"], wd, rel_tol=1e-12), role
        assert g["lr"] == 1.0  # the current lr comes from the scheduler (train.py)


# ---- the fold: training with c in wo / fc2 equals training with a residual multiplier c ----

class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))  # gain 1 + weight, as GroupRMSNorm

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],)) * (1 + self.weight)


class Attention(nn.Module):
    def __init__(self, dim, heads, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.norm, self.query_norm = Norm(dim), Norm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        register_muon(self.wq, layout="(h d) m", fan_out="d", fan_in="m", d=dim // heads)
        register_muon(self.wo)

    def forward(self, x):
        h = self.query_norm(self.wq(self.norm(x)))
        h = torch.cumsum(h, dim=1) / torch.arange(1, x.shape[1] + 1)[:, None]  # causal mixing over tokens
        return x + self.multiplier * self.wo(torch.tanh(h))


class FFN(nn.Module):
    def __init__(self, dim, hidden, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.norm = Norm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        register_muon(self.fc1)
        register_muon(self.fc2)

    def forward(self, x):
        return x + self.multiplier * self.fc2(F.silu(self.fc1(self.norm(x))))


class Toy(nn.Module):
    """The parameter layout of xllm's Transformer (embed, layers.*.attention / nffn, output.final_norm /
    output) with a residual multiplier on the branches, runnable on CPU."""

    def __init__(self, multiplier=1.0, vocab=48, dim=32, hidden=64, heads=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Module()
            block.attention, block.nffn = Attention(dim, heads, multiplier), FFN(dim, hidden, multiplier)
            self.layers.append(block)
        self.output = nn.Module()
        self.output.final_norm, self.output.output = Norm(dim), nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens, targets):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer.nffn(layer.attention(x))
        logits = self.output.output(self.output.final_norm(x))
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


FOLD_CFG = OptimConf(optimizer="muon", lr=1e-2, muon_lr=0.05, epsilon=1e-8, weight_decay=0.1, muon_weight_decay=0.2)


def fold_pair(c):
    """multiplier: residual multiplier c, plain init. folded: no multiplier, wo / fc2 x c."""
    torch.manual_seed(0)
    multiplier = Toy(multiplier=c)
    folded = Toy()
    folded.load_state_dict(multiplier.state_dict())
    with torch.no_grad():
        for p in mup_lib.out_projections(folded):
            p.mul_(c)
    return multiplier, folded


def train(model, opt, steps, clip=None):
    """clip(model) runs between backward and the optimizer step."""
    torch.manual_seed(1)
    losses = []
    for _ in range(steps):
        tokens = torch.randint(0, 48, (4, 16))
        loss = model(tokens, (5 * tokens + 1) % 48)  # learnable targets
        loss.backward()
        if clip is not None:
            clip(model)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses


def fold_error(multiplier, folded, c):
    """Max relative difference between the folded weights and the multiplier run's (wo / fc2 x c)."""
    out = {id(p) for p in mup_lib.out_projections(folded)}
    err = 0.0
    for a, b in zip(multiplier.parameters(), folded.parameters()):
        want = a * c if id(b) in out else a
        err = max(err, ((b - want).abs().max() / want.abs().max()).item())
    return err


def float64_newton_schulz(monkeypatch):
    """Muon's Newton-Schulz iteration and update buffers in float64 instead of bf16."""
    def newton_schulz(X, eps):
        X = X.double()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
        for a, b, c in muon.NS_COEFFS:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
        return X
    empty = torch.empty  # already moved to CPU by cpu_transport
    monkeypatch.setattr(torch, "empty", lambda *a, **k: empty(*a, **(k | {"dtype": torch.float64}
                                                                     if k.get("dtype") == torch.bfloat16 else k)))
    monkeypatch.setattr(muon, "newton_schulz5", newton_schulz)


def clipper(max_norm, grad_scales=None, norms=None):
    """The model's clip_grad_norm_ (xllm/models/xllm.py) as a train() hook; appends each norm to `norms`."""
    from xllm.models.xllm import XLLModel

    def clip(model):
        norm = XLLModel.clip_grad_norm_(model, max_norm, grad_scales=grad_scales)
        if norms is not None:
            norms.append(norm)
    return clip


# (alpha, clipping, Newton-Schulz in float64, tolerance)
FOLD_CASES = [
    (0.5, False, False, 0.0),
    (0.8, False, True, 1e-5),
    (0.5, True, True, 1e-5),
    (0.8, True, True, 1e-5),
]


@pytest.mark.parametrize("alpha, clipping, float64, tol", FOLD_CASES)
def test_folded_training_matches_residual_multiplier(cpu_transport, monkeypatch, alpha, clipping, float64, tol):
    """alpha 0.5: c = 0.5, a power of two, so Muon's bf16 Newton-Schulz gives bit-identical runs.
    alpha 0.8: c = 4^-0.8 = 0.33. bf16 rounds c * x differently from x, so bf16 Newton-Schulz differs at its own
    error (about 2% of the largest element); in float64 the runs agree up to rounding. So does clipping, whose
    norms are summed in another order (the multiplier run has no grad_scales: its gradients are the unfolded ones)."""
    mup = resolved_mup(alpha=alpha)  # m_N = 2, m_L = 4: every role scaled
    c = mup.branch_scale
    if float64:
        float64_newton_schulz(monkeypatch)
    multiplier, folded = fold_pair(c)
    opt_multiplier = build_muon_cpu(multiplier, FOLD_CFG, mup, cpu_transport)
    opt_multiplier.update_scales = [1.0] * len(opt_multiplier.update_scales)  # c sits in the forward instead
    opt_folded = build_muon_cpu(folded, FOLD_CFG, mup, cpu_transport)
    assert sorted(set(opt_folded.update_scales)) == sorted({c, 1.0})
    norms = []
    clips = [clipper(0.5, norms=norms), clipper(0.5, mup_lib.clip_grad_scales(folded, mup))] if clipping else [None] * 2
    losses = [train(multiplier, opt_multiplier, 8, clips[0]), train(folded, opt_folded, 8, clips[1])]
    assert max(abs(a - b) for a, b in zip(*losses)) <= tol * losses[0][0]
    assert losses[0][-1] < losses[0][0] - 0.5  # it trains
    assert not clipping or min(norms) > 0.5  # every step clips
    assert fold_error(multiplier, folded, c) <= tol

    # negative control: without the update scale, or the gradient scale when clipping, the folded run drifts away
    _, unscaled = fold_pair(c)
    opt_unscaled = build_muon_cpu(unscaled, FOLD_CFG, mup, cpu_transport)
    if clipping:
        train(unscaled, opt_unscaled, 8, clipper(0.5))
    else:
        opt_unscaled.update_scales = [1.0] * len(opt_unscaled.update_scales)
        train(unscaled, opt_unscaled, 8)
    assert fold_error(multiplier, unscaled, c) > 100 * max(tol, 1e-6)


def test_clip_norm_counts_scaled_gradients():
    from xllm.models.xllm import XLLModel
    model = Toy()
    torch.manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    out = {id(p) for p in mup_lib.out_projections(model)}
    grad_scales = {i: 0.3 for i in out}
    want = torch.cat([(p.grad * (0.3 if id(p) in out else 1.0)).flatten() for p in model.parameters()]).norm()
    grads = [p.grad.clone() for p in model.parameters()]
    norm = XLLModel.clip_grad_norm_(model, 1e9, grad_scales=grad_scales)
    assert math.isclose(norm, want.item(), rel_tol=1e-6)
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))  # under max_norm: unclipped
    XLLModel.clip_grad_norm_(model, norm / 4, grad_scales=grad_scales)
    for p, g in zip(model.parameters(), grads):  # every gradient by the same factor
        torch.testing.assert_close(p.grad, g / 4, rtol=1e-5, atol=0)
    plain = torch.cat([g.flatten() for g in grads]).norm().item()
    assert math.isclose(XLLModel.clip_grad_norm_(model, 1e9) * 4, plain, rel_tol=1e-6)  # without grad_scales
