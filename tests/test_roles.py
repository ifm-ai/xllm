"""Parameter roles (RolesConf, xllm/optim/roles.py) without mup. CPU.

    PYTHONPATH=. pytest -q tests/test_roles.py

Expected values are written out by hand, not recomputed with the code under test.
"""
import copy
import json
import math
import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.nn.utils import get_total_norm

import xllm.optim.muon as muon
from xllm.config import (
    MUON_ROLES, DataLoaderConfig, ModelConf, OptimConf, RoleConf, RolesConf, SlurmConf, TokenizerConf, TrainerConf,
    ValidConf,
)
from xllm.models.xllm import XLLModel
from xllm.optim import build_optimizer, check_param_groups, param_group_names, restore_optimizer_config
from xllm.optim import roles as roles_lib
from xllm.optim.muon_spec import muon_spec
from xllm.utils import clip_grad_norm_
from tests.test_muon_missing_grad import LocalMesh, cpu_transport  # noqa: F401 (fixture)

CFG = OptimConf(optimizer="muon", lr=1e-3, muon_lr=0.02, epsilon=1e-8, weight_decay=0.1, muon_weight_decay=0.05)
# LayerNorm gains and biases, query_norm, the attention gate wr, SwiGLU's fc3
MODEL = dict(model_dim=64, num_layers=2, num_heads=4, ffn_hidden_dim=128, vocab_size=32, init_mode="gaussian",
             qknorm=True, apply_attn_gate=True, swiglu=True)


def transformer(roles=None, **options):
    from xllm.models.transformer import Transformer
    cfg = ModelConf(**(MODEL | options))
    torch.manual_seed(0)
    if roles is None:
        return Transformer(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    with roles_lib.init_roles(roles):
        return Transformer(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))


def build_muon_cpu(model, cfg, roles, cpu_transport):
    """build_muon without FSDP2: the same split, role groups and scales."""
    muon_params, specs, _, adamw_params = muon.split_muon_params(model)
    lr_scales = wd_scales = None
    if roles is not None:
        adamw_params, lr_scales, wd_scales = roles_lib.param_groups(model, cfg, roles, muon_params)
    opt = muon.Muon(muon_params, specs, adamw_params, LocalMesh(), cfg, lr_scales=lr_scales, wd_scales=wd_scales)
    cpu_transport(opt)
    return opt


def random_grads(model, seed):
    torch.manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn_like(p)


def expected_role(name):
    if name.startswith("embed."):
        return "embed"
    if name.startswith("output.final_norm."):
        return "final_norm"
    if name == "output.output.weight":
        return "readout"
    if name.endswith(("attention.wo.weight", "nffn.fc2.weight")):
        return "residual_out"
    return "hidden_vector" if "norm." in name else "hidden_matrix"


# ---- which parameters have which role ----

@pytest.mark.parametrize("options", [
    {},  # LayerNorm gains and biases, query_norm, the attention gate wr, SwiGLU's fc3
    dict(apply_rmsnorm=True),
    dict(norm_affine=False, qknorm=False),  # no norm parameters in the blocks
    dict(attn_act_func="softdelta"),  # its gate wg
    dict(apply_attn_gate=False, swiglu=False),
    dict(num_kv_heads=2, head_dim=32),  # GQA, head_dim != model_dim / num_heads
    dict(fused_block=True, causal_attn_backend="flash"),
    dict(layerwise_ckpt=True),
])
def test_transformer_tags_every_parameter(options):
    model = transformer(**options)
    tags = roles_lib.param_roles(model)
    roles = {name: tags[id(p)] for name, p in model.named_parameters()}
    assert roles == {name: expected_role(name) for name in roles}  # none left to `other`
    muon_weights = {id(p) for p in muon.split_muon_params(model)[0]}
    assert {id(p) for p in model.parameters() if tags[id(p)] in MUON_ROLES} == muon_weights


def test_untagged_parameters_are_in_role_other(cpu_transport):
    """As gekko's and MoE's, which have no register_role: they step at the role `other`'s values."""
    model = transformer()
    wq = model.layers[0].attention.wq
    del vars(wq)["param_roles"]  # a Muon matrix without a role
    model.layers[0].attention.extra = nn.Parameter(torch.zeros(4))  # an AdamW parameter without a role
    extra = model.layers[0].attention.extra
    tags = roles_lib.param_roles(model)
    assert tags[id(wq.weight)] == tags[id(extra)] == "other"
    roles = RolesConf(other=RoleConf(lr=0.5, weight_decay=2.0, clip_scale=0.25))
    opt = build_muon_cpu(model, CFG, roles, cpu_transport)
    (group,) = [g for g in opt.param_groups if g.get("role") == "other"]
    assert (group["lr"], group["weight_decay"]) == (5e-4, 0.2) and group["params"] == [extra]
    scales = {id(p): s for p, *s in zip(opt.param_groups[0]["params"], opt.lr_scales, opt.wd_scales)}
    assert scales.pop(id(wq.weight)) == [0.5, 2.0]
    assert all(s == [1.0, 1.0] for s in scales.values())
    assert roles_lib.clip_grad_scales(model, roles) == {id(wq.weight): 0.25, id(extra): 0.25}


def test_param_groups_reject_mismatched_roles():
    model = transformer()
    del vars(model.layers[0].attention.wo)["muon_specs"]  # residual_out, but stepped by AdamW
    with pytest.raises(ValueError, match=r"tagged Muon weights \(register_muon\): \['layers.0.attention.wo"):
        roles_lib.param_groups(model, CFG, RolesConf(), muon.split_muon_params(model)[0])
    model = transformer()
    vars(model.layers[1].nffn.fc1)["param_roles"]["weight"] = "readout"  # a Muon weight in an AdamW role
    with pytest.raises(ValueError, match=r"tagged Muon weights \(register_muon\): \['layers.1.nffn.fc1.weight'\]"):
        roles_lib.param_groups(model, CFG, RolesConf(), muon.split_muon_params(model)[0])
    model = transformer()
    with pytest.raises(ValueError, match="Muon's eps is not scaled per role"):
        roles_lib.param_groups(model, CFG, RolesConf(hidden_matrix=RoleConf(eps=2.0)), muon.split_muon_params(model)[0])


def test_register_role_checks_and_skips():
    linear = nn.Linear(2, 2, bias=False)
    with pytest.raises(AssertionError, match="unknown role 'hidden'"):
        roles_lib.register_role(linear, "weight", "hidden")
    roles_lib.register_role(linear, "bias", "hidden_vector")  # None: no parameter, no tag
    assert "param_roles" not in vars(linear)
    roles_lib.register_role(linear, "weight", "hidden_matrix")
    with pytest.raises(AssertionError, match="Linear.weight already has role hidden_matrix"):
        roles_lib.register_role(linear, "weight", "readout")
    model = nn.Module()
    model.a, model.b = linear, nn.Module()
    model.b.weight = linear.weight  # shared
    roles_lib.register_role(model.b, "weight", "readout")
    with pytest.raises(ValueError, match="b.weight has two roles: hidden_matrix and readout"):
        roles_lib.param_roles(model)


# ---- unset roles, values, config ----

def test_unset_roles_change_nothing():
    plain, unset = transformer(), transformer(RolesConf())
    for (name, a), b in zip(plain.named_parameters(), unset.parameters()):
        assert torch.equal(a, b), name  # init_roles with unset roles initializes nothing
    assert roles_lib.clip_grad_scales(plain, RolesConf()) is None  # the plain clipping norm
    assert not roles_lib.grouped(RolesConf())  # build_optimizer: Muon's two plain groups


@pytest.mark.parametrize("roles, grouped", [
    (RolesConf(), False),
    (RolesConf(embed=RoleConf(init_std=0.02), residual_out=RoleConf(clip_scale=0.5)), False),  # nothing for the optimizer
    (RolesConf(readout=RoleConf(lr=1.0, eps=1.0, weight_decay=1.0)), False),  # set, but 1
    (RolesConf(readout=RoleConf(lr=0.5)), True),
    (RolesConf(embed=RoleConf(eps=0.5)), True),
    (RolesConf(readout=RoleConf(weight_decay=0.0)), True),
])
def test_role_groups_only_for_optimizer_factors(roles, grouped):
    """So that a resume without the init_std / clip_scale flags finds the param groups it saved."""
    assert roles_lib.grouped(roles) == grouped


@pytest.mark.parametrize("role, values, match", [
    ("readout", dict(lr=-0.5), "roles.readout.lr must be finite and >= 0, got -0.5"),
    ("hidden_matrix", dict(lr=float("inf")), "roles.hidden_matrix.lr must be finite and >= 0, got inf"),
    ("embed", dict(eps=0.0), "roles.embed.eps must be finite and > 0, got 0.0"),
    ("hidden_vector", dict(weight_decay=-1.0), "roles.hidden_vector.weight_decay must be finite and >= 0, got -1.0"),
    ("residual_out", dict(init_std=-0.01), "roles.residual_out.init_std must be finite and >= 0, got -0.01"),
    ("residual_out", dict(clip_scale=float("nan")), "roles.residual_out.clip_scale must be finite and > 0, got nan"),
    ("other", dict(init_std=0.02), "roles.other.init_std: the parameters without a role keep their module's init"),
])
def test_role_values_are_checked(role, values, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        RolesConf(**{role: RoleConf(**values)})
    assert RolesConf(residual_out=RoleConf(init_std=0.0, weight_decay=0.0, lr=0.0)).residual_out.init_std == 0.0


def trainer_conf(**kw):
    return TrainerConf(**(dict(slurm=SlurmConf(), model=ModelConf(model_dim=512, num_layers=16, num_heads=8),
                               optim=OptimConf(optimizer="muon"), valid=ValidConf(), tokenizer=TokenizerConf(),
                               dataloader=DataLoaderConfig(), dcp_for_optimizer=True) | kw))


@pytest.mark.parametrize("kw, match", [
    (dict(optim=OptimConf(optimizer="adamw"), dcp_for_optimizer=False, roles=RolesConf(readout=RoleConf(lr=0.5))),
     "roles (--roles.*, --mup) require optim.optimizer=muon"),
    (dict(roles=RolesConf(hidden_matrix=RoleConf(eps=0.5))), "roles.hidden_matrix.eps: Muon has no eps to scale"),
    (dict(roles=RolesConf(embed=RoleConf(init_std=0.02)), model_parallel_size=2),
     "roles' init_std (set by mup) need model_parallel_size 1"),
])
def test_trainer_conf_rejects_role_misuse(kw, match):
    """At config time, before the model or the optimizer is built."""
    with pytest.raises(AssertionError, match=re.escape(match)):
        trainer_conf(**kw)
    assert trainer_conf().roles == RolesConf()


def test_roles_from_cli_and_config_json():
    args = ["--slurm.partition", "main", "--optim.optimizer", "muon", "--dcp_for_optimizer", "true",
            "--roles.readout.lr", "0.5", "--roles.readout.weight_decay", "0", "--roles.embed.init_std", "0.02"]
    cfg = TrainerConf.from_cli(vars(TrainerConf.to_cli().parse_args(args)))
    assert cfg.roles == RolesConf(readout=RoleConf(lr=0.5, weight_decay=0.0), embed=RoleConf(init_std=0.02))
    saved = json.loads(json.dumps(cfg.to_dict()))  # train.py's config.json
    assert TrainerConf.from_dict(saved).roles == cfg.roles
    del saved["roles"]  # a config.json from before roles
    assert TrainerConf.from_dict(saved).roles == RolesConf()


# ---- init ----

def test_init_roles_initializes_each_role_at_its_std():
    stds = {"embed": 0.5, "hidden_matrix": 0.03, "residual_out": 0.0, "readout": 0.07}  # 0: zeros
    model = transformer(RolesConf(**{role: RoleConf(init_std=std) for role, std in stds.items()}))
    tags = roles_lib.param_roles(model)
    for name, p in model.named_parameters():
        std = stds.get(tags[id(p)])
        if std:  # truncated at 3 std: std x 0.9866, nothing beyond 3 std
            assert abs(p.std().item() / (0.9866 * std) - 1) < 0.05, name
            assert p.abs().max().item() <= 3 * std, name
        else:  # zeros, and the norm gains (an offset from 1) and biases keep their init
            assert torch.equal(p, torch.zeros_like(p)), name
    assert roles_lib._init_stds == {}  # only while the model is built


def test_init_roles_nests_and_clears():
    with roles_lib.init_roles(RolesConf(embed=RoleConf(init_std=0.1))):
        with roles_lib.init_roles(RolesConf(readout=RoleConf(init_std=0.2))):
            assert (roles_lib._init_stds["embed"], roles_lib._init_stds["readout"]) == (None, 0.2)
        assert (roles_lib._init_stds["embed"], roles_lib._init_stds["readout"]) == (0.1, None)
    with pytest.raises(RuntimeError):
        with roles_lib.init_roles(RolesConf(embed=RoleConf(init_std=0.1))):
            raise RuntimeError
    assert roles_lib._init_stds == {}


# ---- optimizer ----

def test_adamw_role_groups_step_as_torch_adamw(cpu_transport):
    """Each AdamW role group, with its own lr, eps and weight decay, steps as torch.optim.AdamW with those values."""
    roles = RolesConf(embed=RoleConf(lr=0.5, eps=1e7), hidden_vector=RoleConf(weight_decay=0.0),
                      final_norm=RoleConf(lr=2.0), readout=RoleConf(lr=0.25, eps=1e5, weight_decay=4.0))
    # (lr, eps, weight_decay): CFG's 1e-3, 1e-8, 0.1 times the factors
    want = {"embed": (5e-4, 0.1, 0.1), "hidden_vector": (1e-3, 1e-8, 0.0), "final_norm": (2e-3, 1e-8, 0.1),
            "readout": (2.5e-4, 1e-3, 0.4)}
    model = transformer(apply_rmsnorm=True)
    opt = build_muon_cpu(model, CFG, roles, cpu_transport)
    groups = {g["role"]: g["params"] for g in opt.param_groups[1:]}
    assert list(groups) == list(want)
    copies = {role: [p.detach().clone().requires_grad_() for p in ps] for role, ps in groups.items()}
    reference = [torch.optim.AdamW(copies[role], lr=lr, eps=eps, weight_decay=wd, betas=(CFG.beta1, CFG.beta2))
                 for role, (lr, eps, wd) in want.items()]
    control_params = [p.detach().clone().requires_grad_() for p in groups["embed"]]
    control = torch.optim.AdamW(control_params, lr=5e-4, eps=1e-8, weight_decay=0.1, betas=(CFG.beta1, CFG.beta2))
    for seed in range(3):
        random_grads(model, seed)
        for role, ps in groups.items():
            for p, q in zip(ps, copies[role]):
                q.grad = p.grad.clone()
        for p, q in zip(groups["embed"], control_params):
            q.grad = p.grad.clone()
        for o in (opt, control, *reference):
            o.step()
    for role, ps in groups.items():
        for p, q in zip(ps, copies[role]):
            torch.testing.assert_close(p, q, rtol=1e-6, atol=0, msg=role)
    # negative control: without the embed's eps factor
    assert max((p - q).abs().max() / p.abs().max() for p, q in zip(groups["embed"], control_params)) > 1e-4


def test_muon_matrix_scales_step_as_scaled_lr_and_decay(cpu_transport):
    """A matrix with lr / weight decay factors (l, w) steps as a Muon of its own whose lr and weight decay are
    x l and x w; three of the matrices share a Newton-Schulz buffer. Powers of two throughout, so every product in
    the update is exact and the runs are bit-identical."""
    cfg = OptimConf(optimizer="muon", muon_lr=0.0625, weight_decay=0.5)
    shapes = [(8, 4), (8, 4), (8, 4), (4, 8)]
    factors = [(0.5, 2.0), (2.0, 0.25), (1.0, 1.0), (0.25, 4.0)]
    torch.manual_seed(0)
    init = [torch.randn(shape) for shape in shapes]

    def scaled(lr_scales, wd_scales):
        weights = [nn.Parameter(w.clone()) for w in init]
        opt = muon.Muon(weights, [muon_spec()] * len(weights), [], LocalMesh(), cfg,
                        lr_scales=lr_scales, wd_scales=wd_scales)
        cpu_transport(opt)
        return weights, [opt]

    weights, opts = scaled([l for l, _ in factors], [w for _, w in factors])
    reversed_weights, reversed_opts = scaled([l for l, _ in factors[::-1]], [w for _, w in factors[::-1]])
    alone = [nn.Parameter(w.clone()) for w in init]
    for p, (l, w) in zip(alone, factors):
        opts.append(muon.Muon([p], [muon_spec()], [], LocalMesh(),
                              OptimConf(optimizer="muon", muon_lr=cfg.muon_lr * l, weight_decay=cfg.weight_decay * w)))
        cpu_transport(opts[-1])
    for _ in range(3):
        grads = [torch.randn(shape) for shape in shapes]
        for params in (weights, reversed_weights, alone):
            for p, g in zip(params, grads):
                p.grad = g.clone()
        for o in opts + reversed_opts:
            o.step()
    for i, (p, q) in enumerate(zip(weights, alone)):
        assert torch.equal(p, q), f"matrix {i}"
    # negative control: each matrix with another's factors
    assert max(((p - q).abs().max() / q.abs().max()).item() for p, q in zip(reversed_weights, alone)) > 1e-2


def test_restore_applies_role_values(cpu_transport):
    roles = RolesConf(embed=RoleConf(eps=0.5), readout=RoleConf(lr=0.5, weight_decay=2.0))
    opt = build_muon_cpu(transformer(num_layers=1), CFG, roles, cpu_transport)
    for group in opt.param_groups:  # what an optimizer state reloaded from another config leaves
        group.update(lr=1.0, initial_lr=1.0, eps=1.0, weight_decay=1.0)
    restore_optimizer_config(opt, OptimConf(**dict(vars(CFG), muon_lr=0.04, lr=2e-3)), roles)
    # (initial_lr, eps, weight_decay) by hand
    want = {"muon": (0.04, 1e-8, 0.05), "embed": (2e-3, 5e-9, 0.1), "hidden_vector": (2e-3, 1e-8, 0.1),
            "final_norm": (2e-3, 1e-8, 0.1), "readout": (1e-3, 1e-8, 0.2)}
    groups = {g.get("role", g["algorithm"]): g for g in opt.param_groups}
    assert list(groups) == list(want)
    for role, values in want.items():
        g = groups[role]
        assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip((g["initial_lr"], g["eps"], g["weight_decay"]), values)), role
        assert g["lr"] == 1.0  # the current lr comes from the scheduler (train.py)


def test_role_groups_resume(cpu_transport, tmp_path):
    """Role groups' optimizer state through DCP, and the check before loading that the groups are the saved ones."""
    roles = RolesConf(readout=RoleConf(lr=0.5, weight_decay=0.0), residual_out=RoleConf(lr=2.0))
    model = transformer(num_layers=1)
    opt = build_muon_cpu(model, CFG, roles, cpu_transport)
    random_grads(model, 0)
    opt.step()
    dcp.save(get_optimizer_state_dict(model, opt), checkpoint_id=tmp_path / "optimizer")
    training_state = {"param_groups": param_group_names(opt), "scheduler": {"base_lrs": [0.0] * len(opt.param_groups)}}
    assert training_state["param_groups"] == ["muon", "embed", "hidden_vector", "final_norm", "readout"]

    resumed = copy.deepcopy(model)
    new = build_muon_cpu(resumed, CFG, roles, cpu_transport)
    check_param_groups(new, training_state, "checkpoint")
    template = get_optimizer_state_dict(resumed, new)
    dcp.load(template, checkpoint_id=tmp_path / "optimizer")
    set_optimizer_state_dict(resumed, new, template)
    restore_optimizer_config(new, CFG, roles)
    assert [g.get("role") for g in new.param_groups] == [None, "embed", "hidden_vector", "final_norm", "readout"]
    for o, m in ((opt, model), (new, resumed)):  # the next step: as without the interruption
        random_grads(m, 1)
        o.step()
    for (name, a), b in zip(model.named_parameters(), resumed.parameters()):
        assert torch.equal(a, b), name

    plain = build_muon_cpu(copy.deepcopy(model), CFG, None, cpu_transport)
    with pytest.raises(ValueError, match=re.escape(
            "checkpoint has optimizer param groups ['muon', 'embed', 'hidden_vector', 'final_norm', 'readout'], "
            "this run ['muon', 'adamw']")):
        check_param_groups(plain, training_state, "checkpoint")
    with pytest.raises(ValueError, match=re.escape("checkpoint has optimizer param groups 2, this run 5")):
        check_param_groups(new, {"scheduler": {"base_lrs": [0.02, 1e-3]}}, "checkpoint")  # older: only the count


def test_adamw_param_groups():
    """optimizer=adamw, the default (torch's AdamW, whose group has no algorithm key): the names saved at a checkpoint
    and checked on resume, also against a checkpoint from before the names were saved."""
    opt, scheduler = build_optimizer(nn.Linear(2, 2), OptimConf(), 10, "bf16")
    assert type(opt) is torch.optim.AdamW
    training_state = {"param_groups": param_group_names(opt), "scheduler": scheduler.state_dict()}
    assert training_state["param_groups"] == ["adamw"]
    check_param_groups(opt, training_state, "checkpoint")
    check_param_groups(opt, {"scheduler": scheduler.state_dict()}, "checkpoint")  # older: only the count
    with pytest.raises(ValueError, match=re.escape(
            "checkpoint has optimizer param groups ['muon', 'adamw'], this run ['adamw']")):
        check_param_groups(opt, {"param_groups": ["muon", "adamw"], "scheduler": {}}, "checkpoint")


def test_group_lrs(cpu_transport):
    """The lrs train.py logs: adamw_lr, or one per role (a Muon role's: Muon's lr x its factor)."""
    model = transformer(num_layers=1)
    assert roles_lib.group_lrs(build_muon_cpu(model, CFG, None, cpu_transport), RolesConf()) == \
        {"muon": 0.02, "adamw": 1e-3}
    roles = RolesConf(residual_out=RoleConf(lr=0.5), readout=RoleConf(lr=0.25))
    assert roles_lib.group_lrs(build_muon_cpu(model, CFG, roles, cpu_transport), roles) == {
        "muon": 0.02, "embed": 1e-3, "hidden_vector": 1e-3, "final_norm": 1e-3, "readout": 2.5e-4,
        "hidden_matrix": 0.02, "residual_out": 0.01,
    }


# ---- clipping ----

def test_clip_norm_counts_scaled_gradients():
    model = transformer()
    random_grads(model, 0)
    grad_scales = roles_lib.clip_grad_scales(model, RolesConf(residual_out=RoleConf(clip_scale=0.3)))
    assert len(grad_scales) == 2 * 2  # wo and fc2 of each layer
    grads = [p.grad.clone() for p in model.parameters()]
    scaled = torch.cat([(g * grad_scales.get(id(p), 1.0)).flatten() for p, g in zip(model.parameters(), grads)])
    clip, grad = XLLModel.clip_grad_norm_(model, 1e9, grad_scales=grad_scales)
    assert math.isclose(clip, scaled.norm().item(), rel_tol=1e-6)
    assert math.isclose(grad, torch.cat([g.flatten() for g in grads]).norm().item(), rel_tol=1e-6)
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))  # under max_norm: unclipped
    XLLModel.clip_grad_norm_(model, clip / 4, grad_scales=grad_scales)
    for p, g in zip(model.parameters(), grads):  # every gradient by the same factor
        torch.testing.assert_close(p.grad, g / 4, rtol=1e-5, atol=0)

    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone()
    plain = get_total_norm(grads, 2.0, False, None).item()  # the norm without roles, as computed before them
    assert XLLModel.clip_grad_norm_(model, 1e9) == (plain, plain)
    fsdp1 = SimpleNamespace(clip_grad_norm_=lambda max_norm, norm_type: 3.0)  # FSDP1's: one norm
    assert clip_grad_norm_(fsdp1, 1.0) == (3.0, 3.0)
