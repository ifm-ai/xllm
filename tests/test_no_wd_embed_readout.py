"""optim.no_weight_decay_embed_readout: the embedding and the readout step with weight_decay 0. CPU.

    PYTHONPATH=. pytest -q tests/test_no_wd_embed_readout.py

Both tensors are AdamW-stepped under either optimizer (neither is register_muon'd), so the flag has to work for
optimizer "adamw" and "muon" alike, and has to survive a resume.
"""
from types import SimpleNamespace

import pytest
import torch

from xllm.config import AdamWConf, ModelConf, MuonConf, OptimConf
from xllm.optim import restore_optimizer_config
from xllm.optim.optimizer import build_optimizer
from tests.test_muon_missing_grad import LocalMesh, cpu_transport  # noqa: F401 (fixture)

MODEL = dict(model_dim=64, num_layers=2, num_heads=4, ffn_hidden_dim=128, vocab_size=32, init_mode="gaussian",
             swiglu=True)
WD = 0.1
ARCHS = ["transformer", "gekko", "hybrid"]


def conf(optimizer, exempt):
    return OptimConf(optimizer=optimizer, no_weight_decay_embed_readout=exempt,
                     adamw=AdamWConf(lr=2e-3, epsilon=1e-8, weight_decay=WD),
                     muon=MuonConf(lr=0.01, weight_decay=0.05))


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
    """build(arch) with each parameter on a one-rank mesh, where build_muon looks for FSDP2's p.device_mesh."""
    model, mesh = build(arch), OneRankMesh()
    for p in model.parameters():
        p.device_mesh = mesh
    return model


def groups_by_algorithm(optimizer):
    """Muon labels its groups; the plain AdamW path has no reason to, so that path is checked by weight_decay."""
    return {g["algorithm"]: g for g in optimizer.param_groups}


@pytest.mark.parametrize("arch", ARCHS)
def test_adamw_path_exempts_them(arch):
    model = build(arch)
    optimizer, _ = build_optimizer(model, conf("adamw", True), total_steps=10, param_dtype="float32")
    want = {id(model.embed.weight), id(model.output.output.weight)}
    undecayed = [g for g in optimizer.param_groups if g["weight_decay"] == 0.0]
    decayed = [g for g in optimizer.param_groups if g["weight_decay"] == WD]
    assert len(undecayed) == 1 and len(decayed) == 1
    assert {id(p) for p in undecayed[0]["params"]} == want
    assert not want & {id(p) for p in decayed[0]["params"]}  # never in two groups
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))
    assert undecayed[0]["lr"] == decayed[0]["lr"]  # only the decay differs


def test_adamw_path_unchanged_when_off():
    model = build("transformer")
    optimizer, _ = build_optimizer(model, conf("adamw", False), total_steps=10, param_dtype="float32")
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == WD


@pytest.mark.parametrize("arch", ARCHS)
def test_muon_path_exempts_them(cpu_transport, arch):
    model = sharded(arch)
    cfg = conf("muon", True)
    optimizer, _ = build_optimizer(model, cfg, total_steps=10, param_dtype="float32")
    groups = groups_by_algorithm(optimizer)
    assert optimizer.param_groups[0]["algorithm"] == "muon"  # the muon group stays first
    assert groups["muon"]["weight_decay"] == 0.05
    assert groups["adamw"]["weight_decay"] == WD
    assert groups["adamw_no_wd"]["weight_decay"] == 0.0
    exempt = {id(p) for p in groups["adamw_no_wd"]["params"]}
    assert exempt == {id(model.embed.weight), id(model.output.output.weight)}
    assert not exempt & {id(p) for p in groups["adamw"]["params"]}
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))


def test_muon_path_unchanged_when_off(cpu_transport):
    model = sharded("transformer")
    optimizer, _ = build_optimizer(model, conf("muon", False), total_steps=10, param_dtype="float32")
    assert [g["algorithm"] for g in optimizer.param_groups] == ["muon", "adamw"]


def test_resume_reapplies_the_zero(cpu_transport):
    """restore_optimizer_config rewrites each group from the config; the exempt group must not pick up the decay."""
    model = sharded("transformer")
    cfg = conf("muon", True)
    optimizer, _ = build_optimizer(model, cfg, total_steps=10, param_dtype="float32")
    for group in optimizer.param_groups:  # a checkpoint that decayed everything
        group["weight_decay"] = 0.42
    restore_optimizer_config(optimizer, cfg)
    groups = groups_by_algorithm(optimizer)
    assert groups["adamw_no_wd"]["weight_decay"] == 0.0
    assert groups["adamw"]["weight_decay"] == WD

