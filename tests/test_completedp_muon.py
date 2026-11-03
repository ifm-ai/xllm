"""Complete(d)P with Muon (MuPConf, xllm/config.py): the preset of the roles, and the depth scale c folded into
residual_out (attention wo / FFN fc2) training exactly as a residual multiplier c. CPU.

    PYTHONPATH=. pytest -q tests/test_completedp_muon.py

Expected values are written out by hand, not recomputed with the formulas under test.
"""
import copy
import json
import math
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from xllm.config import (
    DataLoaderConfig, ModelConf, MuPConf, OptimConf, RoleConf, RolesConf, SlurmConf, TokenizerConf, TrainerConf,
    ValidConf,
)
import xllm.optim.muon as muon
from xllm.models.xllm import XLLModel
from xllm.optim import restore_optimizer_config
from xllm.optim import roles as roles_lib
from xllm.optim.muon_spec import register_muon
from tests.test_muon_missing_grad import cpu_transport  # noqa: F401 (fixture)
from tests.test_roles import build_muon_cpu


def mup_conf(**kw):
    """m_N = 512 / 256 = 2, m_L = 16 / 4 = 4 with model_conf(); alpha 0.5 gives c = 4^-0.5 = 0.5."""
    return MuPConf(**dict(base_model_dim=256, base_num_layers=4, base_init_hidden_std=0.02,
                          base_init_embed_std=0.5, base_init_unembed_std=0.04, alpha=0.5) | kw)


def model_conf(**kw):
    return ModelConf(**dict(model_dim=512, num_layers=16, num_heads=8, ffn_hidden_dim=1024,
                            init_mode="gaussian") | kw)


def resolved(model=None, **kw):
    """The resolved mup and its roles."""
    mup, roles = mup_conf(**kw), RolesConf()
    mup.resolve(model or model_conf(), OptimConf(optimizer="muon"), roles)
    return mup, roles


def trainer_conf(**kw):
    return TrainerConf(**(dict(slurm=SlurmConf(), model=model_conf(), optim=OptimConf(optimizer="muon"),
                               valid=ValidConf(), tokenizer=TokenizerConf(), dataloader=DataLoaderConfig(),
                               dcp_for_optimizer=True) | kw))


def assert_roles(roles, want):
    """want: role -> (lr, eps, weight_decay, init_std, clip_scale)."""
    for role, values in want.items():
        r = getattr(roles, role)
        for name, got, expected in zip(("lr", "eps", "weight_decay", "init_std", "clip_scale"),
                                       (r.lr, r.eps, r.weight_decay, r.init_std, r.clip_scale), values):
            ok = got is None if expected is None else got is not None and math.isclose(got, expected, rel_tol=1e-12)
            assert ok, (role, name, got, expected)


# ---- the preset ----

def test_resolve_sets_multipliers_and_roles():
    mup, roles = resolved()
    assert (mup.width_multiplier, mup.depth_multiplier, mup.branch_scale) == (2.0, 4.0, 0.5)
    assert_roles(roles, {
        "embed": (1.0, 0.5, 1.0, 0.5, 1.0),
        "hidden_matrix": (0.5, 1.0, 1.0, 0.01414213562373095, 1.0),  # 0.02 / sqrt(2)
        "residual_out": (0.25, 1.0, 2.0, 0.007071067811865475, 0.5),  # lr and init x c, weight decay x 1/c, clipping x c
        "hidden_vector": (0.5, 0.25, 1.0, None, 1.0),
        "final_norm": (1.0, 1.0, 1.0, None, 1.0),
        "readout": (0.5, 1.0, 2.0, 0.02, 1.0),  # 0.04 / 2
        "other": (1.0, 1.0, 1.0, None, 1.0),  # holds no parameter under mup
    })
    mup, roles = resolved(alpha=1.0)
    assert mup.branch_scale == 0.25  # L0 / L
    assert_roles(roles, {"hidden_matrix": (1.0, 1.0, 1.0, 0.01414213562373095, 1.0),
                         "residual_out": (0.25, 1.0, 4.0, 0.0035355339059327377, 0.25),
                         "hidden_vector": (1.0, 0.125, 1.0, None, 1.0)})


@pytest.mark.parametrize("model_kw, optim_kw, mup_kw, match", [
    ({"init_mode": "he"}, {}, {}, "init_mode"),
    ({"init_std": 0.02}, {}, {}, "model.init_std"),  # the roles' init_std replace the model's
    ({"init_logits_std": 0.02}, {}, {}, "model.init_logits_std"),
    ({"arch": "gekko"}, {}, {}, "arch"),
    ({"num_experts": 4, "num_activated_experts": 1, "expert_inter_dim": 64}, {}, {}, "num_experts"),
    ({"residual_func": "ortho", "residual_heads": 8}, {}, {}, "residual_func"),
    ({"scale_emb": True}, {}, {}, "scale_emb"),
    ({}, {"optimizer": "adamw"}, {}, "optimizer"),
    ({}, {"muon_adjust_lr": "rms_norm"}, {}, "muon_adjust_lr"),
    ({}, {"muon_block_scale": "matrix"}, {}, "muon_block_scale"),
    ({}, {}, {"alpha": 0.4}, "alpha"),
    ({}, {}, {"base_num_layers": 0}, "base_num_layers"),
    ({}, {}, {"base_head_dim": 32}, "mup keeps head_dim fixed across widths: model 64, mup.base_head_dim 32"),
])
def test_resolve_rejects_unsupported(model_kw, optim_kw, mup_kw, match):
    with pytest.raises(ValueError, match=match):
        mup_conf(**mup_kw).resolve(model_conf(**model_kw), OptimConf(**dict(optimizer="muon") | optim_kw), RolesConf())
    resolved(base_head_dim=64)  # 512 / 8


@pytest.mark.parametrize("roles, match", [
    (RolesConf(readout=RoleConf(weight_decay=2.0, lr=0.5)), None),  # the preset's values
    (RolesConf(readout=RoleConf(weight_decay=1.0)), "roles.readout.weight_decay=1.0 conflicts with mup, which sets it to 2.0"),
    (RolesConf(readout=RoleConf(lr=0.3)), "roles.readout.lr=0.3 conflicts with mup, which sets it to 0.5"),
    (RolesConf(hidden_vector=RoleConf(init_std=0.01)), "roles.hidden_vector.init_std=0.01 conflicts with mup, which sets it to None"),
    (RolesConf(other=RoleConf(lr=0.5)), "roles.other.lr=0.5 conflicts with mup, which sets it to 1.0"),
])
def test_roles_set_by_hand_must_equal_the_preset(roles, match):
    """An explicit value is the user's, even 1.0: it must equal the preset's, never be silently replaced."""
    if match is None:
        mup_conf().resolve(model_conf(), OptimConf(optimizer="muon"), roles)
        assert roles == resolved()[1]
    else:
        with pytest.raises(ValueError, match=re.escape(match)):
            mup_conf().resolve(model_conf(), OptimConf(optimizer="muon"), roles)


def test_trainer_conf_resolves_from_cli_and_config_json():
    args = ["--slurm.partition", "main", "--optim.optimizer", "muon", "--dcp_for_optimizer", "true", "--model.model_dim", "512",
            "--model.num_layers", "16", "--model.num_heads", "8", "--model.init_mode", "gaussian",
            "--mup.base_model_dim", "256", "--mup.base_num_layers", "4", "--mup.alpha", "0.5",
            "--mup.base_init_hidden_std", "0.02", "--mup.base_init_embed_std", "0.5",
            "--mup.base_init_unembed_std", "0.04"]
    cfg = TrainerConf.from_cli(vars(TrainerConf.to_cli().parse_args(args)))
    assert cfg.mup.branch_scale == 0.5 and cfg.roles == resolved()[1]
    saved = json.loads(json.dumps(cfg.to_dict()))  # train.py's config.json holds the resolved roles
    reloaded = TrainerConf.from_dict(copy.deepcopy(saved))
    assert reloaded.mup == cfg.mup and reloaded.model == cfg.model and reloaded.roles == cfg.roles
    del saved["roles"]["other"]  # the first version of this PR had no `other`
    assert TrainerConf.from_dict(copy.deepcopy(saved)).roles == cfg.roles
    saved["model"].update(model_dim=1024, num_heads=16)  # a run's config.json as the base of a wider run
    wider = TrainerConf.from_dict(saved)
    assert wider.mup.width_multiplier == 4.0
    assert_roles(wider.roles, {"embed": (1.0, 0.25, 1.0, 0.5, 1.0), "readout": (0.25, 1.0, 4.0, 0.01, 1.0)})
    assert trainer_conf().mup is None and trainer_conf().roles == RolesConf()


def test_trainer_conf_rejects_mup_with_tensor_parallel():
    """mup sets init stds, which a tensor-parallel shard would draw the same on every rank."""
    with pytest.raises(AssertionError, match=re.escape("roles' init_std (set by mup) need model_parallel_size 1")):
        trainer_conf(mup=mup_conf(), model_parallel_size=2)


def transformer(num_layers=4, roles=None, **kw):
    from xllm.models.transformer import Transformer
    cfg = model_conf(**dict(model_dim=64, num_layers=num_layers, num_heads=4, ffn_hidden_dim=128, vocab_size=32,
                            apply_rmsnorm=True, qknorm=True, swiglu=True) | kw)
    torch.manual_seed(0)
    with roles_lib.init_roles(roles or RolesConf()):
        return Transformer(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))


def test_mup_requires_a_role_on_every_parameter():
    model = transformer()
    roles_lib.require_roles(model)
    del vars(model.layers[1].nffn.fc3)["param_roles"]  # as an untagged module would leave it in `other`
    with pytest.raises(ValueError, match=re.escape("these have none: ['layers.1.nffn.fc3.weight']")):
        roles_lib.require_roles(model)


def test_residual_out_init_is_c_times_plain_init():
    """The real Transformer: wo / fc2 are exactly c x their init without c (c a power of two), the rest equal."""
    cfg = dict(model_dim=128, num_heads=4, ffn_hidden_dim=256, vocab_size=64)
    mup, roles = resolved(model_conf(num_layers=4, **cfg), base_model_dim=64, base_num_layers=1, alpha=1.0)
    assert mup.branch_scale == 0.25
    plain = copy.deepcopy(roles)
    plain.residual_out.init_std = plain.hidden_matrix.init_std
    folded, reference = transformer(roles=roles, **cfg), transformer(roles=plain, **cfg)
    n_out = 0
    for (name, a), (_, b) in zip(reference.named_parameters(), folded.named_parameters()):
        if name.endswith(("attention.wo.weight", "nffn.fc2.weight")):
            n_out += 1
            assert torch.equal(b, a * 0.25), name
        else:
            assert torch.equal(b, a), name
    assert n_out == 2 * 4


# ---- the preset in Muon (single rank on CPU) ----

# m_N = 2, m_L = 4, alpha = 0.5: c = 0.5 and m_L^(alpha - 1) = 0.5
CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, epsilon=1e-8, weight_decay=0.1, muon_weight_decay=0.05)


def groups_by_role(opt):
    return {g.get("role", g["algorithm"]): g for g in opt.param_groups}


def test_param_groups_on_transformer(cpu_transport):
    model = transformer()
    opt = build_muon_cpu(model, CFG, resolved()[1], cpu_transport)
    groups = groups_by_role(opt)
    # group: (lr, eps, weight_decay, tensors); the Muon group keeps the configured values, its matrices are scaled
    want = {
        "muon": (0.02, 1e-8, 0.05, 4 * 7),  # wq wk wv wo fc1 fc2 fc3
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
    # (lr, weight_decay) factors: hidden matrices x m_L^(alpha - 1); wo / fc2 lr x c more, decay lr * wd the same
    out = {id(layer.attention.wo.weight) for layer in model.layers} | {id(layer.nffn.fc2.weight) for layer in model.layers}
    scales = [(p, (lr, wd)) for p, lr, wd in zip(groups["muon"]["params"], opt.lr_scales, opt.wd_scales)]
    assert [s for p, s in scales if id(p) in out] == [(0.25, 2.0)] * 8
    assert [s for p, s in scales if id(p) not in out] == [(0.5, 1.0)] * 20


def test_mup_at_base_size_matches_plain_muon(cpu_transport):
    """m_N = m_L = 1: every factor is 1, so build_optimizer keeps the two plain groups; role groups would step the same."""
    plain, scaled = transformer(), transformer()
    mup, roles = resolved(base_model_dim=512, base_num_layers=16)
    assert (mup.width_multiplier, mup.depth_multiplier, mup.branch_scale) == (1.0, 1.0, 1.0)
    assert not roles_lib.grouped(roles)
    opts = [build_muon_cpu(plain, CFG, None, cpu_transport), build_muon_cpu(scaled, CFG, roles, cpu_transport)]
    assert [len(o.param_groups) for o in opts] == [2, 5]
    for seed in range(3):
        torch.manual_seed(seed)
        grads = [torch.randn_like(p) for p in plain.parameters()]
        for model, opt in zip((plain, scaled), opts):
            for p, g in zip(model.parameters(), grads):
                p.grad = g.clone()
            opt.step()
    for (name, a), b in zip(plain.named_parameters(), scaled.parameters()):
        assert torch.equal(a, b), name


def test_restore_applies_preset_values(cpu_transport):
    _, roles = resolved()
    opt = build_muon_cpu(transformer(num_layers=1), CFG, roles, cpu_transport)
    for group in opt.param_groups:  # what an optimizer state reloaded from another config leaves
        group.update(lr=1.0, initial_lr=1.0, eps=1.0, weight_decay=1.0)
    restore_optimizer_config(opt, OptimConf(**dict(vars(CFG), muon_lr=0.04, lr=2e-3)), roles)
    groups = groups_by_role(opt)
    for role, values in {"muon": (0.04, 1e-8, 0.05), "embed": (2e-3, 5e-9, 0.1), "hidden_vector": (1e-3, 2.5e-9, 0.1),
                         "final_norm": (2e-3, 1e-8, 0.1), "readout": (1e-3, 1e-8, 0.2)}.items():
        g = groups[role]
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip((g["initial_lr"], g["eps"], g["weight_decay"]), values)), role
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
        for module, role in ((self.norm, "hidden_vector"), (self.query_norm, "hidden_vector"),
                             (self.wq, "hidden_matrix"), (self.wo, "residual_out")):
            roles_lib.register_role(module, "weight", role)

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
        for module, role in ((self.norm, "hidden_vector"), (self.fc1, "hidden_matrix"), (self.fc2, "residual_out")):
            roles_lib.register_role(module, "weight", role)

    def forward(self, x):
        return x + self.multiplier * self.fc2(F.silu(self.fc1(self.norm(x))))


class Toy(nn.Module):
    """The parameter layout and roles of xllm's Transformer (embed, layers.*.attention / nffn, output.final_norm /
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
        for module, role in ((self.embed, "embed"), (self.output.final_norm, "final_norm"),
                             (self.output.output, "readout")):
            roles_lib.register_role(module, "weight", role)

    def forward(self, tokens, targets):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer.nffn(layer.attention(x))
        logits = self.output.output(self.output.final_norm(x))
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


FOLD_CFG = OptimConf(optimizer="muon", lr=1e-2, muon_lr=0.05, epsilon=1e-8, weight_decay=0.1, muon_weight_decay=0.2)


def residual_out(model):
    """wo / fc2 of every block: the weights that hold the folded depth scale c."""
    tags = roles_lib.param_roles(model)
    return [p for p in model.parameters() if tags[id(p)] == "residual_out"]


def unfolded(roles):
    """The roles of the multiplier run: c sits in the forward, so residual_out steps as hidden_matrix."""
    roles = copy.deepcopy(roles)
    roles.residual_out = copy.deepcopy(roles.hidden_matrix)
    return roles


def fold_pair(c):
    """multiplier: residual multiplier c, plain init. folded: no multiplier, wo / fc2 x c."""
    torch.manual_seed(0)
    multiplier = Toy(multiplier=c)
    folded = Toy()
    folded.load_state_dict(multiplier.state_dict())
    with torch.no_grad():
        for p in residual_out(folded):
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
    out = {id(p) for p in residual_out(folded)}
    err = 0.0
    for a, b in zip(multiplier.parameters(), folded.parameters()):
        want = a * c if id(b) in out else a
        err = max(err, ((b - want).abs().max() / want.abs().max()).item())
    return err


def clipper(max_norm, grad_scales=None, norms=None):
    """The model's clip_grad_norm_ (xllm/models/xllm.py) as a train() hook; appends each clipping norm to `norms`."""
    def clip(model):
        clip_norm, _ = XLLModel.clip_grad_norm_(model, max_norm, grad_scales=grad_scales)
        if norms is not None:
            norms.append(clip_norm)
    return clip


def exact_orthogonalization(monkeypatch):
    """Newton-Schulz replaced by the polar factor U V^T it approximates (SVD in fp64), exactly invariant to the scale
    of its input; Newton-Schulz is so only up to its rounding and its eps. The fold is a property of the scaling
    around it; Newton-Schulz's own accuracy is tested in tests/test_muon.py and tests/test_muon_ops.py."""
    def polar(X, eps):
        U, _, Vh = torch.linalg.svd(X.double(), full_matrices=False)
        return U @ Vh
    monkeypatch.setattr(muon, "newton_schulz", polar)


# (alpha, clipping, tolerance)
FOLD_CASES = [
    (0.5, False, 0.0),
    (1.0, False, 0.0),
    (0.5, True, 2e-3),
    (1.0, True, 2e-3),
]


@pytest.mark.parametrize("alpha, clipping, tol", FOLD_CASES)
def test_folded_training_matches_residual_multiplier(cpu_transport, monkeypatch, alpha, clipping, tol):
    """m_L = 4: c = 0.5 (alpha 0.5) and 0.25 (alpha 1), powers of two, so bf16 rounds c * x as it rounds x and the
    runs are bit-identical. With clipping they agree up to rounding: the clipping norm sums the gradients in another
    order (by clip scale; the multiplier run has none), so the clip factor can differ in its last bit, and a few bf16
    roundings of the update with it."""
    exact_orthogonalization(monkeypatch)
    mup, roles = resolved(alpha=alpha)  # m_N = 2, m_L = 4: every role scaled
    c = mup.branch_scale
    cfg = FOLD_CFG
    multiplier, folded = fold_pair(c)
    opt_multiplier = build_muon_cpu(multiplier, cfg, unfolded(roles), cpu_transport)
    opt_folded = build_muon_cpu(folded, cfg, roles, cpu_transport)
    depth_lr = roles.hidden_matrix.lr
    assert set(zip(opt_folded.lr_scales, opt_folded.wd_scales)) == {(depth_lr, 1.0), (depth_lr * c, 1 / c)}
    assert roles_lib.clip_grad_scales(multiplier, unfolded(roles)) is None
    norms = []
    clip_scales = roles_lib.clip_grad_scales(folded, roles)
    clips = [clipper(0.5, norms=norms), clipper(0.5, clip_scales)] if clipping else [None] * 2
    losses = [train(multiplier, opt_multiplier, 8, clips[0]), train(folded, opt_folded, 8, clips[1])]
    assert max(abs(a - b) for a, b in zip(*losses)) <= tol * losses[0][0]
    assert losses[0][-1] < losses[0][0] - 0.5  # it trains
    assert not clipping or min(norms) > 0.5  # every step clips
    assert fold_error(multiplier, folded, c) <= tol

    # negative control: without residual_out's lr / decay factors, or its clip_scale when clipping, the folded run
    # drifts away
    _, unscaled = fold_pair(c)
    if clipping:
        train(unscaled, build_muon_cpu(unscaled, cfg, roles, cpu_transport), 8, clipper(0.5))
    else:
        train(unscaled, build_muon_cpu(unscaled, cfg, unfolded(roles), cpu_transport), 8)
    assert fold_error(multiplier, unscaled, c) > 5 * max(tol, 1e-4)
