from copy import deepcopy

import pytest
import torch
from torch import nn

from xllm.config import OptimConf
from xllm.optim.dion_muon import parameter_groups, validate_optimizer_contract
from xllm.optim.lr_scheduler import build_lr_scheduler, load_lr_scheduler_state
from xllm.optim.optimizer import adamw_param_groups


def model_like():
    """Mirrors the real layout: ``embed``, ``layers.*``, ``output.{final_norm,output}``."""
    model = nn.Module()
    model.arch, model.num_experts, model.num_values = 'transformer', 0, 0
    model.embed = nn.Embedding(16, 8)
    block = nn.Module()
    block.attention = nn.Module()
    block.attention.wq = nn.Linear(8, 8, bias=False)
    block.norm = nn.LayerNorm(8)
    model.layers = nn.ModuleList([block])
    model.output = nn.Module()
    model.output.final_norm = nn.LayerNorm(8)
    model.output.output = nn.Linear(8, 16, bias=False)
    return model


FLAGS = [(False, False), (True, False), (False, True), (True, True)]


def expected_no_decay(model, emb, out):
    return ({id(model.embed.weight)} if emb else set()) | ({id(model.output.output.weight)} if out else set())


def test_flags_default_off():
    cfg = OptimConf()
    assert not cfg.disable_wd_emb and not cfg.disable_wd_out
    args = OptimConf.to_cli().parse_args(['--disable_wd_emb', 'true', '--disable_wd_out', 'true'])
    cfg = OptimConf.from_cli(vars(args))
    assert cfg.disable_wd_emb and cfg.disable_wd_out


@pytest.mark.parametrize('emb,out', FLAGS)
def test_adamw_groups(emb, out):
    model = model_like()
    cfg = OptimConf(disable_wd_emb=emb, disable_wd_out=out)
    optimizer = torch.optim.AdamW(adamw_param_groups(model, cfg), lr=cfg.lr, weight_decay=cfg.weight_decay)
    no_decay = expected_no_decay(model, emb, out)
    if not no_decay:
        assert len(optimizer.param_groups) == 1
    else:
        decay_group, no_decay_group = optimizer.param_groups
        assert decay_group['weight_decay'] == cfg.weight_decay
        assert no_decay_group['weight_decay'] == 0.0
        assert {id(p) for p in no_decay_group['params']} == no_decay
        # The final norm of the output layer keeps its decay.
        assert id(model.output.final_norm.weight) in {id(p) for p in decay_group['params']}
    assert sum(len(g['params']) for g in optimizer.param_groups) == len(list(model.parameters()))


@pytest.mark.parametrize('emb,out', FLAGS)
def test_muon_groups(emb, out):
    model = model_like()
    cfg = OptimConf(optimizer='dion_muon', disable_wd_emb=emb, disable_wd_out=out)
    groups, contract = parameter_groups(model, cfg)
    no_decay = expected_no_decay(model, emb, out)
    assert [g['group_id'] for g in groups] == ['muon', 'adamw'] + (['adamw_no_wd'] if no_decay else [])
    assert {id(p) for p in groups[0]['params']} == {id(model.layers[0].attention.wq.weight)}
    if no_decay:
        assert groups[2]['algorithm'] == 'adamw'
        assert groups[2]['weight_decay'] == 0.0
        assert groups[2]['lr'] == cfg.lr
        assert {id(p) for p in groups[2]['params']} == no_decay
    # Muon and AdamW groups leave weight_decay to the optimizer default.
    assert 'weight_decay' not in groups[0] and 'weight_decay' not in groups[1]
    assert len(contract) == len(groups)
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))


def test_muon_contract_rejects_flag_change_on_resume():
    model = model_like()
    _, before = parameter_groups(model, OptimConf(optimizer='dion_muon'))
    _, after = parameter_groups(model, OptimConf(optimizer='dion_muon', disable_wd_emb=True))
    assert before != after


def test_legacy_muon_contract_without_flags_still_resumes():
    current = {'optimizer': 'dion_muon', 'recipe': {
        'lr': 0.001, 'muon_use_gram_newton_schulz': False, 'muon_use_triton': False,
        'disable_wd_emb': False, 'disable_wd_out': False,
    }}
    legacy = {'optimizer_contract': {'optimizer': 'dion_muon', 'recipe': {'lr': 0.001}}}
    optimizer = type('O', (), {'_xllm_optimizer_contract': current})()
    validate_optimizer_contract(optimizer, legacy)
    current['recipe']['disable_wd_out'] = True
    with pytest.raises(ValueError, match='contract changed'):
        validate_optimizer_contract(optimizer, legacy)


def test_adamw_scheduler_resume_overrides_every_group_lr():
    """With two AdamW groups, resume must keep scheduling both (not just the first)."""
    model = model_like()
    cfg = OptimConf(lr=0.002, warmup=3, disable_wd_emb=True)

    def build(lr):
        optimizer = torch.optim.AdamW(adamw_param_groups(model, cfg), lr=lr)
        return optimizer, build_lr_scheduler(optimizer, cfg, 20)

    optimizer, scheduler = build(cfg.lr)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    saved = deepcopy(scheduler.state_dict())
    saved['base_lrs'] = [0.001, 0.001]  # stale LR from an older config
    resumed, resumed_scheduler = build(cfg.lr)
    load_lr_scheduler_state(resumed_scheduler, saved, cfg)
    assert resumed_scheduler.base_lrs == [cfg.lr, cfg.lr]
    for _ in range(5):
        optimizer.step()
        resumed.step()
        scheduler.step()
        resumed_scheduler.step()
        lrs = [g['lr'] for g in resumed.param_groups]
        assert lrs[0] == lrs[1] == scheduler.get_last_lr()[0]


def test_muon_scheduler_resume_accepts_three_groups_and_rejects_mismatch():
    cfg = OptimConf(optimizer='dion_muon', disable_wd_emb=True)
    params = [nn.Parameter(torch.ones(2)) for _ in range(3)]
    optimizer = torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(params, (cfg.muon_lr, cfg.lr, cfg.lr))])
    scheduler = build_lr_scheduler(optimizer, cfg, 20)
    state = deepcopy(scheduler.state_dict())
    load_lr_scheduler_state(scheduler, state, cfg)
    state['base_lrs'] = state['base_lrs'][:2]
    with pytest.raises(ValueError):
        load_lr_scheduler_state(scheduler, state, cfg)
