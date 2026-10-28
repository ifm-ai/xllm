"""CPU optimizer API regressions; single-rank communication runs locally."""
import ast
import copy
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from xllm.config import OptimConf
import xllm.optim.muon as muon
from xllm.optim.muon_spec import muon_spec
from xllm.optim import build_optimizer, build_lr_scheduler, restore_optimizer_config


class LocalMesh:
    def get_group(self):
        return None

    def size(self):
        return 1

    def get_local_rank(self):
        return 0


@pytest.fixture
def cpu_transport(monkeypatch):
    # Exercise the actual gather/scatter and update code on CPU, replacing only
    # the CUDA allocation and single-rank transport.
    real_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)

    def exchange(send, recv):  # one rank: what I send is what I receive
        return lambda: [d.copy_(s) for ds, ss in zip(recv, send) for d, s in zip(ds, ss)]
    return lambda opt: monkeypatch.setattr(opt, "_exchange", exchange)


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_missing_grad_skips_state_and_weight_update(cpu_transport, nesterov, initialized):
    skipped = nn.Parameter(torch.ones(2, 3), requires_grad=initialized)
    active = nn.Parameter(torch.ones(2, 3))
    cfg = OptimConf(optimizer="muon", muon_lr=0.1, weight_decay=0.2, muon_nesterov=nesterov)
    opt = muon.Muon([skipped, active], [muon_spec()] * 2, [], LocalMesh(), cfg)
    cpu_transport(opt)

    if initialized:
        skipped.grad = torch.ones_like(skipped)
        active.grad = torch.ones_like(active)
        opt.step()
    before = skipped.detach().clone()
    old_momentum = opt.state[skipped]["momentum"].clone() if initialized else None
    active_before = active.detach().clone()
    skipped.grad = None
    active.grad = torch.ones_like(active)
    opt.step()

    assert torch.equal(skipped, before)
    assert not torch.equal(active, active_before)
    if initialized:
        assert torch.equal(opt.state[skipped]["momentum"], old_momentum)
        # A real zero gradient still decays momentum and updates the parameter.
        skipped.grad = torch.zeros_like(skipped)
        opt.step()
        torch.testing.assert_close(opt.state[skipped]["momentum"], old_momentum * cfg.muon_momentum)
        assert not torch.equal(skipped, before)
    else:
        assert skipped not in opt.state


def test_state_roundtrip_restores_algorithm_and_next_update(cpu_transport):
    cfg = OptimConf(optimizer="muon", muon_lr=0.02, muon_momentum=0.8,
                    muon_nesterov=False, muon_adjust_lr="rms_norm", beta1=0.7, beta2=0.9)
    p, a = nn.Parameter(torch.ones(4, 3)), nn.Parameter(torch.ones(3))
    spec = muon_spec("(h d) m", fan_out="d", fan_in="m", d=2)
    old = muon.Muon([p], [spec], [a], LocalMesh(), cfg)
    cpu_transport(old)
    p.grad, a.grad = torch.arange(12.).view(4, 3), torch.ones_like(a)
    old.step()
    saved = copy.deepcopy(old.state_dict())
    q, b = nn.Parameter(p.detach().clone()), nn.Parameter(a.detach().clone())
    new = muon.Muon([q], [spec], [b], LocalMesh(), OptimConf(optimizer="muon", muon_block_scale="matrix"))
    cpu_transport(new)
    new.load_state_dict(saved)
    for loaded, expected in zip(new.param_groups, saved["param_groups"]):
        assert {k: v for k, v in loaded.items() if k != "params"} == {k: v for k, v in expected.items() if k != "params"}
    p.grad = q.grad = torch.arange(12.).flip(0).view(4, 3)
    a.grad = b.grad = torch.full_like(a, 0.5)
    old.step()
    new.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)

    momentum = new.state[q]["momentum"].clone()
    moments = new.state[b]["exp_avg"].clone()
    cfg = OptimConf(optimizer="muon", muon_momentum=0.6, muon_nesterov=True,
                    muon_block_scale="matrix", muon_adjust_lr="spectral_norm",
                    beta1=0.5, beta2=0.8, muon_weight_decay=0.2, weight_decay=0.03, epsilon=1e-6)
    restore_optimizer_config(new, cfg)
    group, adam = new.param_groups
    assert (group["momentum"], group["nesterov"], group["block_scale"], group["adjust_lr"]) == (0.6, True, "matrix", "spectral_norm")
    assert (group["weight_decay"], adam["weight_decay"], adam["betas"], group["eps"]) == (0.2, 0.03, (0.5, 0.8), 1e-6)
    torch.testing.assert_close(new.state[q]["momentum"], momentum, rtol=0, atol=0)
    torch.testing.assert_close(new.state[b]["exp_avg"], moments, rtol=0, atol=0)
    # The next real step uses the explicit override, without resetting history.
    q.grad = torch.ones_like(q)
    new.step()
    torch.testing.assert_close(new.state[q]["momentum"], momentum * 0.6 + q.grad)


def test_closure_computes_gradients_and_returns_loss(cpu_transport):
    p = nn.Parameter(torch.ones(2, 3))
    opt = muon.Muon([p], [muon_spec()], [], LocalMesh(), OptimConf(optimizer="muon"))
    cpu_transport(opt)
    calls = []

    def closure():
        calls.append(1)
        assert torch.is_grad_enabled()
        loss = p.square().sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert calls == [1] and loss.item() == 6
    assert not torch.equal(p, torch.ones_like(p))


def test_muon_factory_requires_fsdp2_model():
    with pytest.raises(ValueError, match="wrapped with FSDP2"):
        build_optimizer(nn.Linear(3, 2), OptimConf(optimizer="muon"), 10, "bf16")


def test_registration_does_not_import_optimizer_runtime():
    result = subprocess.run([sys.executable, "-c", "import sys; from xllm.optim.muon_spec import register_muon; "
                             "assert 'xllm.optim.muon' not in sys.modules; "
                             "assert 'xllm.optim.optimizer' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_first_resumed_lr_uses_new_base_at_restored_position():
    # Load just this training helper: importing train.py also loads GPU/data dependencies.
    path = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "set_optim_state")
    namespace = dict(State=object, MultiSourceDataLoader=object, Timings=object,
                     OptimConf=OptimConf, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    p = nn.Parameter(torch.ones(1))
    cfg = OptimConf(lr=0.1, scheduler="constant", warmup=10)
    old_opt = torch.optim.AdamW([p], lr=cfg.lr)
    old_sched = build_lr_scheduler(old_opt, cfg, 100)
    for _ in range(3):
        old_opt.step()
        old_sched.step()
    saved_sched = copy.deepcopy(old_sched.state_dict())
    cfg.lr = 0.2
    new_opt = torch.optim.AdamW([p], lr=cfg.lr)
    new_sched = build_lr_scheduler(new_opt, cfg, 100)
    new_opt.load_state_dict(old_opt.state_dict())
    saved = dict(scheduler=saved_sched, step=3, scale=1, scale_updates=0, clip_cumulative=0, data_state={})
    namespace["set_optim_state"](new_sched, saved, SimpleNamespace(),
                                  SimpleNamespace(set_state=lambda state: None),
                                  SimpleNamespace(load_state_dict=lambda state: None), cfg)
    assert new_opt.param_groups[0]["lr"] == pytest.approx(old_sched.get_last_lr()[0] * 2)
    assert new_sched.last_epoch == 3 and new_sched._step_count == old_sched._step_count


def test_dcp_checkpoint_roundtrip(cpu_transport, tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

    cfg = OptimConf(optimizer="muon", muon_momentum=0.8, weight_decay=0.2)
    model = nn.Linear(3, 2)
    opt = muon.Muon([model.weight], [muon_spec()], [model.bias], LocalMesh(), cfg)
    cpu_transport(opt)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    opt.step()
    opt.zero_grad(set_to_none=True)
    saved = get_optimizer_state_dict(model, opt)
    dcp.save(saved, checkpoint_id=tmp_path / "optimizer")
    new = muon.Muon([model.weight], [muon_spec()], [model.bias], LocalMesh(),
                    OptimConf(optimizer="muon", muon_momentum=0.6, weight_decay=0.01))
    cpu_transport(new)
    template = get_optimizer_state_dict(model, new)
    dcp.load(template, checkpoint_id=tmp_path / "optimizer")
    set_optimizer_state_dict(model, new, template)
    assert new.param_groups[0]["momentum"] == 0.8
    assert [group["weight_decay"] for group in new.param_groups] == [0.2, 0.2]
    torch.testing.assert_close(new.state[model.weight]["momentum"], opt.state[model.weight]["momentum"])
    torch.testing.assert_close(new.state[model.bias]["exp_avg"], opt.state[model.bias]["exp_avg"])
