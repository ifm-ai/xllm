"""Per-role AdamW values with Muon (AdamWConf's role fields, xllm/optim/roles.py). CPU.

    PYTHONPATH=. pytest -q tests/test_roles.py

Expected values are written out by hand.
"""
import math
from types import SimpleNamespace

import pytest
import torch

import xllm.optim.muon as muon
from xllm.config import AdamWConf, AdamWRoleConf, ModelConf, MuonConf, OptimConf
from xllm.optim import restore_optimizer_config
from xllm.optim import roles as roles_lib
from tests.test_muon_missing_grad import LocalMesh, cpu_transport  # noqa: F401 (fixture)

MODEL = dict(model_dim=64, num_layers=2, num_heads=4, ffn_hidden_dim=128, vocab_size=32, init_mode="gaussian",
             swiglu=True)
# a recipe at one size: embedding, readout and block vectors without decay, the final norm at the plain values.
# The Muon matrices are optim.muon's, not a role.
PLAIN = AdamWConf(lr=2e-3, epsilon=1e-8, weight_decay=0.1)
CFG = OptimConf(optimizer="muon", muon=MuonConf(lr=8e-3, weight_decay=0.05),
                adamw=AdamWConf(lr=2e-3, epsilon=1e-8, weight_decay=0.1,
                                embed=AdamWRoleConf(lr=6.4e-2, weight_decay=0.0),
                                hidden_vector=AdamWRoleConf(lr=1.6e-2, epsilon=2.5e-9, weight_decay=0.0),
                                readout=AdamWRoleConf(lr=2.75e-3, weight_decay=0.0)))
CFG_PLAIN = OptimConf(optimizer="muon", muon=MuonConf(lr=8e-3, weight_decay=0.05), adamw=PLAIN)
WANT = {  # group: (lr, eps, weight_decay)
    "muon": (8e-3, 1e-8, 0.05),  # optim.muon's lr and weight decay; Newton-Schulz eps is adamw.epsilon
    "embed": (6.4e-2, 1e-8, 0.0),
    "hidden_vector": (1.6e-2, 2.5e-9, 0.0),
    "final_norm": (2e-3, 1e-8, 0.1),  # unset: the configured optim.adamw values
    "readout": (2.75e-3, 1e-8, 0.0),
}


def build(arch):
    torch.manual_seed(0)
    if arch == "gekko":
        from xllm.models.gekko import Gekko as cls
        cfg = ModelConf(**(MODEL | dict(arch="gekko", init_mode="he")))
    elif arch == "hybrid":
        from xllm.models.hybrid import HybridGekko as cls
        cfg = ModelConf(**(MODEL | dict(arch="hybrid", init_mode="he", full_attn_layer_list="1",
                                        apply_conv_in_causal_attn=True, apply_attn_gate=True, qknorm=True)))
    else:
        from xllm.models.transformer import Transformer as cls
        cfg = ModelConf(**MODEL)
    return cls(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))


class OneRankMesh(LocalMesh):
    ndim = 1


def sharded(arch):
    """build(arch) with each parameter on a one-rank mesh, where build_muon looks for FSDP2's (p.device_mesh)."""
    model, mesh = build(arch), OneRankMesh()
    for p in model.parameters():
        p.device_mesh = mesh
    return model


def groups(opt):
    return {g.get("role", g["algorithm"]): g for g in opt.param_groups}


def assert_values(opt, lr_key="lr"):
    g = groups(opt)
    assert set(g) == set(WANT)
    for name, (lr, eps, wd) in WANT.items():
        assert math.isclose(g[name][lr_key], lr, rel_tol=1e-12), (name, g[name][lr_key])
        assert math.isclose(g[name]["eps"], eps, rel_tol=1e-12), (name, g[name]["eps"])
        assert g[name]["weight_decay"] == wd, (name, g[name]["weight_decay"])


@pytest.mark.parametrize("arch", ["transformer", "gekko", "hybrid"])
def test_only_muon_matrices_have_no_role(arch):
    """Every parameter AdamW steps carries a role; Muon's matrices carry none, and take optim.muon's values."""
    from xllm.optim.muon_spec import split_muon_params
    model = build(arch)
    muon_params, _, _, _ = split_muon_params(model)
    muon_ids = {id(p) for p in muon_params}
    tags = roles_lib.param_roles(model)
    untagged = {n for n, p in model.named_parameters() if tags[id(p)] == "other"}
    assert untagged == {n for n, p in model.named_parameters() if id(p) in muon_ids}


def test_roles_are_the_adamw_config_fields():
    """ROLES and AdamWConf's role fields must not drift apart; Muon is not among them."""
    assert roles_lib.ROLES == tuple(AdamWConf().roles())
    assert "hidden_matrix" not in roles_lib.ROLES


def test_a_muon_weight_with_a_role_is_rejected(cpu_transport):
    """register_muon and register_role are written separately, so they can disagree: a tagged weight handed to
    Muon would silently step at optim.muon's values instead of its role's."""
    from xllm.optim.muon_spec import register_muon
    model = sharded("transformer")
    tags = roles_lib.param_roles(model)
    readout = next(m for m in model.modules()
                   if getattr(m, "weight", None) is not None and tags.get(id(m.weight)) == "readout")
    register_muon(readout, tp_dim=None)
    with pytest.raises(ValueError, match="must carry no role"):
        muon.build_muon(model, CFG)


@pytest.mark.parametrize("arch", ["transformer", "gekko", "hybrid"])
def test_optimizer_groups(cpu_transport, arch):
    """Muon steps at optim.muon's values, AdamW one group per role; unset values are the configured ones, and a
    resume applies them again whatever the checkpoint held."""
    model = sharded(arch)
    opt = muon.build_muon(model, CFG)
    cpu_transport(opt)
    assert_values(opt)
    for group in opt.param_groups:
        group.update(initial_lr=1.0, eps=1.0, weight_decay=1.0)
    restore_optimizer_config(opt, CFG)  # cfg carries the roles: nothing extra to pass
    assert_values(opt, lr_key="initial_lr")
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()


def test_without_roles_the_groups_are_unchanged(cpu_transport):
    opt = muon.build_muon(sharded("transformer"), CFG_PLAIN)
    assert [g["algorithm"] for g in opt.param_groups] == ["muon", "adamw"]
    assert "role" not in opt.param_groups[1]
    restore_optimizer_config(opt, CFG_PLAIN)
    assert opt.param_groups[0]["lr"] == 8e-3 and opt.param_groups[1]["lr"] == 2e-3


@pytest.mark.parametrize("role, match", [
    (dict(readout=AdamWRoleConf(lr=-1.0)), "optim.adamw.readout.lr must be finite and >= 0"),
    (dict(embed=AdamWRoleConf(epsilon=0.0)), "optim.adamw.embed.epsilon must be finite and > 0"),
    (dict(final_norm=AdamWRoleConf(weight_decay=float("nan"))), "optim.adamw.final_norm.weight_decay"),
])
def test_invalid_values(role, match):
    with pytest.raises(ValueError, match=match):
        AdamWConf(**role)


def test_roles_require_muon():
    with pytest.raises(AssertionError, match="optim.optimizer=muon"):
        OptimConf(optimizer="adamw", adamw=AdamWConf(embed=AdamWRoleConf(lr=1e-2)))
    OptimConf(optimizer="adamw", adamw=PLAIN)  # no roles set: fine
