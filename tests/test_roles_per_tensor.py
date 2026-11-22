"""The matrix roles: one Muon group per tensor type, as Complete(d)P groups them. CPU.

    PYTHONPATH=. pytest -q tests/test_roles_per_tensor.py

The split is only worth anything if each group's lr and weight decay really reach its own tensors, so the tests
here check the tagging, the flat index the groups carry, and that a step moves the groups by their own values.
"""
import pytest
import torch

import xllm.optim.muon as muon
from xllm.config import RoleConf, RolesConf
from xllm.optim import restore_optimizer_config
from xllm.optim.roles import MATRIX_ROLES, param_roles
from tests.test_muon_missing_grad import cpu_transport  # noqa: F401 (fixture)
from tests.test_roles import CFG, ROLES, build, sharded

ARCHS = ["transformer", "gekko", "hybrid"]
# the suffix of a parameter's name -> the role its tensor must have
BY_TENSOR = {
    "wq.weight": "attn_qkv", "wk.weight": "attn_qkv", "wv.weight": "attn_qkv",
    "wo.weight": "attn_out",
    "fc1.weight": "ffn_in", "fc3.weight": "ffn_in",
    "fc2.weight": "ffn_out",
    "wr.weight": "hidden_matrix", "forget_gate.weight": "hidden_matrix",
    "query_norm.weight": "qk_norm", "key_norm.weight": "qk_norm",
}


def roles_of(model):
    tags = param_roles(model)
    return {name: tags[id(p)] for name, p in model.named_parameters()}


def muon_groups(opt):
    return [g for g in opt.param_groups if g["algorithm"] == "muon"]


@pytest.mark.parametrize("arch", ARCHS)
def test_each_tensor_gets_its_own_role(arch):
    """The tagging, tensor by tensor. Anything BY_TENSOR names must carry exactly that role."""
    found = roles_of(build(arch))
    checked = 0
    for name, role in found.items():
        for suffix, want in BY_TENSOR.items():
            if name.endswith(suffix):
                assert role == want, f"{name}: {role}, want {want}"
                checked += 1
    assert checked, f"{arch}: BY_TENSOR matched nothing, so this test proved nothing"


@pytest.mark.parametrize("arch", ARCHS)
def test_no_parameter_is_left_untagged(arch):
    found = roles_of(build(arch))
    assert [name for name, role in found.items() if role == "other"] == []


@pytest.mark.parametrize("arch", ARCHS)
def test_the_muon_groups_partition_the_matrices(cpu_transport, arch):
    """Every Muon parameter is in exactly one Muon group, and each group's `index` is its slot in the flat list
    the per-shard state and the block plan are built over."""
    model = sharded(arch)
    opt = muon.build_muon(model, CFG, ROLES)
    groups = muon_groups(opt)
    assert len(groups) >= 4
    present = [g["role"] for g in groups]
    assert present == [r for r in MATRIX_ROLES if r in present]  # MATRIX_ROLES order, whichever are present
    assert opt.param_groups[0]["algorithm"] == "muon"  # the Muon groups come first
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(opt.muon_params)
    for group in groups:
        for p, i in zip(group["params"], group["index"]):
            assert opt.muon_params[i] is p


def test_a_group_steps_at_its_own_lr(cpu_transport):
    """The point of the split: attn_qkv at lr 0 leaves wq/wk/wv alone while ffn_in still moves."""
    model = sharded("transformer")
    roles = RolesConf(attn_qkv=RoleConf(lr=0.0), ffn_in=RoleConf(lr=1e-2))
    opt = muon.build_muon(model, CFG, roles)
    cpu_transport(opt)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    torch.manual_seed(1)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    after = dict(model.named_parameters())
    frozen = [n for n in after if n.endswith(("wq.weight", "wk.weight", "wv.weight"))]
    moved = [n for n in after if n.endswith(("fc1.weight", "fc2.weight"))]
    assert frozen and moved
    for name in frozen:  # lr 0, so the decay term is multiplied by 0 too
        assert torch.equal(after[name], before[name]), name
    for name in moved:
        assert not torch.equal(after[name], before[name]), name


def test_the_groups_keep_their_own_values_on_resume(cpu_transport):
    """restore_optimizer_config rewrites each group from its role, so the Muon groups must not collapse onto one."""
    model = sharded("transformer")
    opt = muon.build_muon(model, CFG, ROLES)
    cpu_transport(opt)
    want = {g["role"]: g["lr"] for g in muon_groups(opt)}
    assert len(set(want.values())) == len(want), want  # the roles really do differ, or this proves nothing
    for group in opt.param_groups:
        group.update(initial_lr=1.0, weight_decay=1.0)
    restore_optimizer_config(opt, CFG, ROLES)
    assert {g["role"]: g["initial_lr"] for g in muon_groups(opt)} == want


def test_state_dict_round_trip_keeps_the_momenta(cpu_transport):
    """m is flat over muon_params while the state is per group, so load_state_dict has to put it back by index."""
    model = sharded("transformer")
    opt = muon.build_muon(model, CFG, ROLES)
    cpu_transport(opt)
    torch.manual_seed(2)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    assert all(m is not None for m in opt.m)

    resumed = muon.build_muon(sharded("transformer"), CFG, ROLES)
    cpu_transport(resumed)
    assert all(m is None for m in resumed.m)
    resumed.load_state_dict(opt.state_dict())
    assert all(m is not None for m in resumed.m)
    for i, m in enumerate(opt.m):  # the right momentum in the right flat slot, not merely some momentum
        assert torch.equal(resumed.m[i], m), i


def test_a_checkpoint_without_the_index_still_loads(cpu_transport):
    """A checkpoint written before the Muon groups were split carries no `index`, and torch replaces the live
    groups with the checkpoint's on load. The index has to be rebuilt from the parameters, not read back."""
    model = sharded("transformer")
    opt = muon.build_muon(model, CFG, ROLES)
    cpu_transport(opt)
    torch.manual_seed(3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state_dict()
    for group in state["param_groups"]:  # as an older checkpoint would have it
        group.pop("index", None)

    resumed = muon.build_muon(sharded("transformer"), CFG, ROLES)
    cpu_transport(resumed)
    resumed.load_state_dict(state)
    for i, m in enumerate(opt.m):
        assert torch.equal(resumed.m[i], m), i
    for group in muon_groups(resumed):
        for p, i in zip(group["params"], group["index"]):
            assert resumed.muon_params[i] is p


def test_a_role_with_no_gradients_at_all(cpu_transport):
    """Splitting the matrices makes an empty live list reachable per group: before, every Muon matrix shared one
    group, so the lists were only empty when nothing in the whole model had a gradient."""
    model = sharded("transformer")
    opt = muon.build_muon(model, CFG, ROLES)
    cpu_transport(opt)
    torch.manual_seed(4)
    qkv = {id(p) for g in muon_groups(opt) if g["role"] == "attn_qkv" for p in g["params"]}
    assert qkv, "attn_qkv has no parameters, so this proves nothing"
    for p in model.parameters():
        p.grad = None if id(p) in qkv else torch.randn_like(p)
    opt.step()  # the attn_qkv group steps with empty lists
    for g in muon_groups(opt):
        if g["role"] == "attn_qkv":
            for p in g["params"]:
                assert "momentum" not in opt.state[p]  # no gradient, no state


@pytest.mark.parametrize("arch", ARCHS)
def test_the_convs_and_qk_norms_are_tagged_when_enabled(arch):
    """apply_conv_in_causal_attn and qknorm add tensors to the attention blocks; the hybrid-gekko configs set them,
    so a default-config test would miss anything left untagged there."""
    from types import SimpleNamespace
    from xllm.config import ModelConf
    from tests.test_roles import MODEL
    extra = dict(apply_conv_in_causal_attn=True, qknorm=True)
    if arch == "gekko":
        from xllm.models.gekko import Gekko as cls
        cfg = ModelConf(**(MODEL | dict(arch="gekko", init_mode="he") | extra))
    elif arch == "hybrid":
        from xllm.models.hybrid import HybridGekko as cls
        cfg = ModelConf(**(MODEL | dict(arch="hybrid", init_mode="he", full_attn_layer_list="1") | extra))
    else:
        from xllm.models.transformer import Transformer as cls
        cfg = ModelConf(**(MODEL | extra))
    model = cls(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    found = roles_of(model)
    assert [name for name, role in found.items() if role == "other"] == []
    convs = [name for name in found if "_conv." in name]
    assert convs, "no causal convs were built, so this proves nothing"
    assert all(found[name] == "hidden_vector" for name in convs), [(n, found[n]) for n in convs]


def test_the_lr_logging_finds_the_headline_group(cpu_transport):
    """train.py keys its per-group lr dict by role, falling back to the algorithm, and pops the headline Muon lr
    out of it. With roles every Muon group is named by its role, so "muon" is not a key any more."""
    for roles in (ROLES, RolesConf()):
        opt = muon.build_muon(sharded("transformer"), CFG, roles)
        groups = opt.param_groups
        # the expression in train.py, verbatim
        group_lrs = {g.get("role") or g["algorithm"]: float(g["lr"]) for g in groups}
        assert len(group_lrs) == len(groups), "two groups collide on one key, so a group's lr is lost"
        first_muon = next(g for g in groups if g["algorithm"] == "muon")
        curr_lr = group_lrs.pop(first_muon.get("role") or "muon")
        assert curr_lr == first_muon["lr"]


MOE = dict(num_experts=4, num_activated_experts=2, num_shared_experts=1, expert_inter_dim=96, moe_expert_backend="sequential",
           moe_permutation_backend="torch")


def test_moe_tensors_get_the_ffn_roles():
    """The routed experts and the shared expert are the FFN's matrices, so they take ffn_in / ffn_out like the
    dense FFN does; the router is a matrix AdamW steps, so it has a role of its own.

    Transformer only: gekko and hybrid ignore model.num_experts and keep the dense nffn."""
    from types import SimpleNamespace
    from xllm.config import ModelConf
    from tests.test_roles import MODEL
    from xllm.models.transformer import Transformer as cls
    cfg = ModelConf(**(MODEL | MOE))
    found = roles_of(cls(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0)))
    assert [name for name, role in found.items() if role == "other"] == []
    want = {"weight1": "ffn_in", "weight3": "ffn_in", "weight2": "ffn_out", "router.weight": "router"}
    checked = 0
    for name, role in found.items():
        for suffix, expected in want.items():
            if name.endswith(suffix):
                assert role == expected, f"{name}: {role}, want {expected}"
                checked += 1
    assert checked >= len(want), f"only matched {checked} of the MoE tensors"


def test_the_router_is_not_a_muon_role(cpu_transport):
    """A matrix role's lr is a Muon-scale value. The router is never register_muon'd, so it must not carry one,
    or it would step with AdamW at a Muon lr."""
    from xllm.optim.roles import MATRIX_ROLES
    assert "router" not in MATRIX_ROLES


def test_a_matrix_role_on_an_adamw_tensor_is_rejected(cpu_transport):
    """The guard that makes the MoVA mistake loud: a Muon role on a tensor Muon does not step."""
    import xllm.optim.roles as roles_lib
    model = sharded("transformer")
    tagged = next(m for m in model.modules() if "param_roles" in vars(m) and
                  any(r == "hidden_vector" for r in vars(m)["param_roles"].values()))
    name = next(n for n, r in vars(tagged)["param_roles"].items() if r == "hidden_vector")
    vars(tagged)["param_roles"][name] = "attn_qkv"  # an AdamW tensor wearing a Muon role
    with pytest.raises(ValueError, match="would step with AdamW at a Muon learning rate"):
        muon.build_muon(model, CFG, ROLES)
    assert roles_lib.MATRIX_ROLES  # the constant the guard reads
