"""Per-role optimizer hyperparameters with Muon (RolesConf, xllm/optim/roles.py). CPU.

    PYTHONPATH=. pytest -q tests/test_roles.py

Expected values are written out by hand.
"""
import math
from types import SimpleNamespace

import pytest
import torch

import xllm.optim.muon as muon
from xllm.config import (
    AdamWConf, DataLoaderConfig, ModelConf, MuonConf, OptimConf, RoleConf, RolesConf, SlurmConf, TokenizerConf, TrainerConf, ValidConf,
)
from xllm.optim import restore_optimizer_config
from xllm.optim import roles as roles_lib
from tests.test_muon_missing_grad import LocalMesh, cpu_transport  # noqa: F401 (fixture)

CFG = OptimConf(optimizer="muon", adamw=AdamWConf(lr=2e-3, epsilon=1e-8, weight_decay=0.1),
                muon=MuonConf(lr=0.01, weight_decay=0.05))
MODEL = dict(model_dim=64, num_layers=2, num_heads=4, ffn_hidden_dim=128, vocab_size=32, init_mode="gaussian",
             swiglu=True)
# a recipe at one size: embedding, readout and block vectors without decay, the final norm at the plain values,
# and a different lr for each matrix role (Complete(d)P's per-tensor groups)
ROLES = RolesConf(embed=RoleConf(lr=6.4e-2, weight_decay=0.0),
                  attn_qkv=RoleConf(lr=8e-3), attn_out=RoleConf(lr=7e-3),
                  ffn_in=RoleConf(lr=6e-3), ffn_out=RoleConf(lr=5e-3), hidden_matrix=RoleConf(lr=4e-3),
                  qk_norm=RoleConf(lr=1.2e-2, weight_decay=0.0),
                  hidden_vector=RoleConf(lr=1.6e-2, eps=2.5e-9, weight_decay=0.0),
                  readout=RoleConf(lr=2.75e-3, weight_decay=0.0))
WANT = {  # group: (lr, eps, weight_decay); the matrix roles take muon.weight_decay, Newton-Schulz eps is adamw.epsilon
    "attn_qkv": (8e-3, 1e-8, 0.05),
    "attn_out": (7e-3, 1e-8, 0.05),
    "ffn_in": (6e-3, 1e-8, 0.05),
    "ffn_out": (5e-3, 1e-8, 0.05),
    "hidden_matrix": (4e-3, 1e-8, 0.05),
    "qk_norm": (1.2e-2, 1e-8, 0.0),
    "embed": (6.4e-2, 1e-8, 0.0),
    "hidden_vector": (1.6e-2, 2.5e-9, 0.0),
    "final_norm": (2e-3, 1e-8, 0.1),
    "readout": (2.75e-3, 1e-8, 0.0),
}
MATRIX = {"attn_qkv", "attn_out", "ffn_in", "ffn_out"}  # every arch has these; hidden_matrix and qk_norm vary


def build(arch):
    torch.manual_seed(0)
    if arch == "gekko":
        from xllm.models.gekko import Gekko as cls
        cfg = ModelConf(**(MODEL | dict(arch="gekko", init_mode="he")))
    elif arch == "hybrid":
        from xllm.models.hybrid import HybridGekko as cls
        # attention last, as the stage-1 layout puts it
        cfg = ModelConf(**(MODEL | dict(arch="hybrid", init_mode="he", full_attn_layer_list="1")))
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
    """Every group is one of WANT's roles, at its values. Which roles have parameters depends on the arch, so the
    set is checked against MATRIX rather than against all of WANT."""
    g = groups(opt)
    assert set(g) <= set(WANT), sorted(set(g) - set(WANT))
    assert MATRIX <= set(g), sorted(MATRIX - set(g))
    for name, group in g.items():
        lr, eps, wd = WANT[name]
        assert math.isclose(group[lr_key], lr, rel_tol=1e-12), (name, group[lr_key])
        assert math.isclose(group["eps"], eps, rel_tol=1e-12), (name, group["eps"])
        assert group["weight_decay"] == wd, (name, group["weight_decay"])


@pytest.mark.parametrize("arch", ["transformer", "gekko"])
def test_every_parameter_has_a_role(arch):
    model = build(arch)
    tags = roles_lib.param_roles(model)
    assert [n for n, p in model.named_parameters() if tags[id(p)] == "other"] == []


@pytest.mark.parametrize("arch", ["transformer", "gekko"])
def test_optimizer_groups(cpu_transport, arch):
    """One Muon group per matrix role and one AdamW group per role; unset values are the configured ones, and a
    resume applies them again whatever the checkpoint held."""
    model = sharded(arch)
    opt = muon.build_muon(model, CFG, ROLES)
    cpu_transport(opt)
    assert_values(opt)
    for group in opt.param_groups:
        group.update(initial_lr=1.0, eps=1.0, weight_decay=1.0)
    restore_optimizer_config(opt, CFG, ROLES)
    assert_values(opt, lr_key="initial_lr")
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()


def test_restore_needs_the_roles(cpu_transport):
    """Resuming a per-role optimizer without its roles would reset every group to the plain values."""
    opt = muon.build_muon(sharded("transformer"), CFG, ROLES)
    with pytest.raises(ValueError, match="pass the roles"):
        restore_optimizer_config(opt, CFG)


def test_without_roles_the_groups_are_unchanged(cpu_transport):
    opt = muon.build_muon(sharded("transformer"), CFG, RolesConf())
    assert [g["algorithm"] for g in opt.param_groups] == ["muon", "adamw"]
    assert "role" not in opt.param_groups[1]


@pytest.mark.parametrize("roles, match", [
    (dict(readout=RoleConf(lr=-1.0)), "roles.readout.lr must be finite and >= 0"),
    (dict(embed=RoleConf(eps=0.0)), "roles.embed.eps must be finite and > 0"),
    (dict(final_norm=RoleConf(weight_decay=float("nan"))), "roles.final_norm.weight_decay"),
])
def test_invalid_values(roles, match):
    with pytest.raises(ValueError, match=match):
        RolesConf(**roles)


def test_rejects(cpu_transport):
    with pytest.raises(ValueError, match="roles.attn_qkv.eps"):
        muon.build_muon(sharded("transformer"), CFG, RolesConf(attn_qkv=RoleConf(eps=1e-9)))
    with pytest.raises(AssertionError, match="optim.optimizer=muon"):
        TrainerConf(slurm=SlurmConf(), model=ModelConf(**MODEL), optim=OptimConf(optimizer="adamw"),
                    valid=ValidConf(), tokenizer=TokenizerConf(), dataloader=DataLoaderConfig(), roles=ROLES)
