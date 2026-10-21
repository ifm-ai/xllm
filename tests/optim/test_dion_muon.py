from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xllm.config import (
    DataLoaderConfig,
    ModelConf,
    OptimConf,
    SlurmConf,
    TokenizerConf,
    TrainerConf,
    ValidConf,
)
from xllm.optim.dion_muon import (
    get_group_hparams,
    parameter_groups,
    set_group_hparams,
    validate_optimizer_contract,
)
from xllm.optim.lr_scheduler import build_lr_scheduler, load_lr_scheduler_state


def routing_model():
    model = nn.Module()
    model.arch, model.num_experts, model.num_values = 'gekko', 0, 0
    model.embed = nn.Embedding(16, 8)
    full = nn.Module()
    full.attention = nn.Module()
    full.attention.wq = nn.Linear(8, 8)
    full.attention.norm = nn.LayerNorm(8)
    gekko = nn.Module()
    gekko.gda = nn.Module()
    gekko.gda.wq = nn.Linear(8, 8)
    gekko.gda.q_conv = nn.Module()
    gekko.gda.q_conv.weight = nn.Parameter(torch.empty(8, 4))
    gekko.gda.query_norm = nn.LayerNorm(8)
    gekko.nffn = nn.Module()
    gekko.nffn.fc1 = nn.Linear(8, 16)
    model.layers = nn.ModuleList([full, gekko])
    model.output = nn.Module()
    model.output.norm = nn.LayerNorm(8)
    model.output.output = nn.Linear(8, 16, bias=False)
    return model


def group_names(model, groups):
    names = {id(p): n for n, p in model.named_parameters()}
    return {g['group_id']: {names[id(p)] for p in g['params']} for g in groups}


def test_hybrid_routing_keeps_convolution_norm_embedding_and_output_in_adamw():
    model = routing_model()
    groups, contract = parameter_groups(model, OptimConf(optimizer='dion_muon'))
    names = group_names(model, groups)
    assert names['muon'] == {
        'layers.0.attention.wq.weight',
        'layers.1.gda.wq.weight',
        'layers.1.nffn.fc1.weight',
    }
    assert names['adamw'] == {'embed.weight', 'layers.1.gda.q_conv.weight', 'output.output.weight'}
    assert all(name.endswith(('.bias', 'norm.weight')) for name in names['adamw_no_decay'])
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))
    assert [(g['group_id'], g['algorithm']) for g in contract] == [
        ('muon', 'muon'), ('adamw', 'adamw'), ('adamw_no_decay', 'adamw'),
    ]
    assert groups[0]['lr'] != groups[1]['lr']


def test_weight_decay_is_separate_for_muon_adamw_and_vectors():
    model = routing_model()
    groups, _ = parameter_groups(model, OptimConf(optimizer='dion_muon', weight_decay=0.1))
    assert [g['weight_decay'] for g in groups] == [0.1, 0.1, 0.0]
    cfg = OptimConf(optimizer='dion_muon', weight_decay=0.0, muon_weight_decay=0.05)
    groups, _ = parameter_groups(model, cfg)
    assert [g['weight_decay'] for g in groups] == [0.05, 0.0, 0.0]


def test_no_decay_embed_readout_moves_them_out_of_weight_decay():
    model = routing_model()
    cfg = OptimConf(optimizer='dion_muon', no_decay_embed_readout=True)
    groups, contract = parameter_groups(model, cfg)
    names = group_names(model, groups)
    assert names['adamw'] == {'layers.1.gda.q_conv.weight'}
    assert {'embed.weight', 'output.output.weight'} <= names['adamw_no_decay']
    assert contract != parameter_groups(model, OptimConf(optimizer='dion_muon'))[1]


def test_empty_groups_are_dropped():
    model = routing_model()
    model.layers[1].gda.q_conv.weight.requires_grad_(False)
    groups, contract = parameter_groups(model, OptimConf(optimizer='dion_muon', no_decay_embed_readout=True))
    assert [g['group_id'] for g in groups] == [c['group_id'] for c in contract] == ['muon', 'adamw_no_decay']


def test_frozen_parameters_are_excluded_and_membership_is_stable():
    model = routing_model()
    before = parameter_groups(model, OptimConf())[1]
    assert before == parameter_groups(model, OptimConf())[1]
    model.layers[1].gda.wq.weight.requires_grad_(False)
    groups, after = parameter_groups(model, OptimConf())
    assert all(id(model.layers[1].gda.wq.weight) != id(p) for g in groups for p in g['params'])
    assert before != after


def test_tied_parameters_fail_before_building_optimizer():
    model = routing_model()
    model.output.weight = model.embed.weight
    with pytest.raises(ValueError, match='tied parameter'):
        parameter_groups(model, OptimConf())


@pytest.mark.parametrize(
    'field,value',
    [
        ('muon_lr', float('nan')),
        ('lr', 0),
        ('muon_weight_decay', -0.1),
        ('muon_momentum', 1),
        ('muon_adjust_lr', 'unknown'),
        ('muon_orthogonalizer', 'unknown'),
    ],
)
def test_invalid_muon_recipe(field, value):
    with pytest.raises(ValueError):
        OptimConf(optimizer='dion_muon', **{field: value})


def trainer(**overrides):
    values = dict(
        slurm=SlurmConf(),
        model=ModelConf(ddp_backend='fsdp2'),
        optim=OptimConf(optimizer='dion_muon'),
        valid=ValidConf(),
        tokenizer=TokenizerConf(),
        dataloader=DataLoaderConfig(),
        dcp_for_optimizer=True,
    )
    values.update(overrides)
    return TrainerConf(**values)


def test_muon_admits_context_parallelism():
    assert trainer(context_parallel_size=2).context_parallel_size == 2


@pytest.mark.parametrize(
    'overrides',
    [
        {'model': ModelConf(ddp_backend='fsdp1')},
        {'model_parallel_size': 2},
        {'dcp_for_optimizer': False},
        {'async_checkpointing': True},
        {'model': ModelConf(ddp_backend='fsdp2', memory_efficient_norm=True)},
        {'seq_len': 8192, 'context_parallel_size': 8},
    ],
)
def test_muon_rejects_unqualified_execution(overrides):
    with pytest.raises(ValueError):
        trainer(**overrides)


def test_scheduler_resume_takes_group_lrs_from_config():
    cfg = OptimConf(optimizer='dion_muon', muon_lr=0.02, lr=0.001, warmup=3)
    parameters = [nn.Parameter(torch.ones(2, 2)), nn.Parameter(torch.ones(2))]

    def build(cfg, total_steps=20):
        optimizer = torch.optim.SGD(
            [
                {'params': [parameters[0]], 'lr': cfg.muon_lr},
                {'params': [parameters[1]], 'lr': cfg.lr},
            ]
        )
        return optimizer, build_lr_scheduler(optimizer, cfg, total_steps)

    optimizer, scheduler = build(cfg)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    scheduler_state = deepcopy(scheduler.state_dict())

    resumed, resumed_scheduler = build(cfg)
    load_lr_scheduler_state(resumed_scheduler, scheduler_state, cfg)
    for _ in range(5):
        optimizer.step()
        resumed.step()
        scheduler.step()
        resumed_scheduler.step()
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()

    # Lower both LRs and extend the run without touching the code.
    lowered = OptimConf(optimizer='dion_muon', muon_lr=0.01, lr=0.0005, warmup=3)
    resumed, resumed_scheduler = build(lowered, total_steps=40)
    load_lr_scheduler_state(resumed_scheduler, scheduler_state, lowered)
    assert resumed_scheduler.base_lrs == [0.01, 0.0005]
    assert resumed_scheduler.last_epoch == scheduler_state['last_epoch']
    assert scheduler_state['base_lrs'] == [cfg.muon_lr, cfg.lr]


def test_scheduler_resume_rejects_another_group_count():
    cfg = OptimConf(optimizer='dion_muon')
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(2))], lr=0.001)
    scheduler = build_lr_scheduler(optimizer, cfg, 20)
    saved = deepcopy(scheduler.state_dict())
    saved['base_lrs'] = [0.01, 0.001]
    with pytest.raises(ValueError, match='LR groups'):
        load_lr_scheduler_state(scheduler, saved, cfg)


def contract(**overrides):
    value = {
        'optimizer': 'dion_muon',
        'groups': [{'group_id': 'muon', 'algorithm': 'muon', 'membership_sha256': 'a'}],
        'mesh_shape': [2, 2],
        'mesh_dims': ['dp', 'cp'],
        'context_parallel_size': 2,
        'data_parallel_size': 2,
    }
    value.update(overrides)
    return value


def test_contract_rejects_layout_and_topology_drift():
    optimizer = SimpleNamespace(_xllm_optimizer_contract=contract())
    validate_optimizer_contract(optimizer, {'optimizer_contract': contract()})
    for overrides in (
        {'groups': []},
        {'mesh_shape': [4, 1]},
        {'context_parallel_size': 1},
        {'optimizer': 'other'},
    ):
        with pytest.raises(ValueError, match='contract changed'):
            validate_optimizer_contract(optimizer, {'optimizer_contract': contract(**overrides)})


def test_contract_ignores_hyperparameters_and_scheduler_from_older_checkpoints():
    optimizer = SimpleNamespace(_xllm_optimizer_contract=contract())
    older = contract(
        recipe={'lr': 0.001, 'muon_lr': 0.02, 'muon_use_triton': True},
        scheduler={'total_steps': 100},
        replica_sync=True,
    )
    validate_optimizer_contract(optimizer, {'optimizer_contract': older})


def test_contract_rejects_switching_between_adamw_and_muon():
    optimizer = SimpleNamespace(_xllm_optimizer_contract=contract())
    with pytest.raises(ValueError):
        validate_optimizer_contract(optimizer, {})
    with pytest.raises(ValueError):
        validate_optimizer_contract(SimpleNamespace(), {'optimizer_contract': contract()})
    validate_optimizer_contract(SimpleNamespace(), {})


def test_configured_group_hparams_override_reloaded_ones():
    weights = [nn.Parameter(torch.ones(2, 2)), nn.Parameter(torch.ones(2))]
    optimizer = torch.optim.SGD(
        [
            {'params': [weights[0]], 'group_id': 'muon', 'weight_decay': 0.1, 'mu': 0.95},
            {'params': [weights[1]], 'group_id': 'adamw', 'weight_decay': 0.1, 'epsilon': 1e-8},
        ],
        lr=0.001,
    )
    saved = deepcopy(optimizer.state_dict())
    optimizer.param_groups[0].update(weight_decay=0.05, mu=0.9)
    optimizer.param_groups[1].update(weight_decay=0.0)
    configured = get_group_hparams(optimizer)
    optimizer.load_state_dict(saved)
    assert optimizer.param_groups[0]['weight_decay'] == 0.1
    set_group_hparams(optimizer, configured)
    assert [g['weight_decay'] for g in optimizer.param_groups] == [0.05, 0.0]
    assert optimizer.param_groups[0]['mu'] == 0.9


def test_adamw_defaults_and_scheduler_lr_override_are_unchanged():
    cfg = OptimConf(lr=0.002)
    assert cfg.optimizer == 'adamw'
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(2))], lr=cfg.lr)
    scheduler = build_lr_scheduler(optimizer, cfg, 20)
    saved = deepcopy(scheduler.state_dict())
    saved['base_lrs'] = [0.001]
    load_lr_scheduler_state(scheduler, saved, cfg)
    assert scheduler.base_lrs == [cfg.lr]
    assert saved['base_lrs'] == [0.001]


def test_optional_muon_flags_default_off_and_accept_cli_flags():
    cfg = OptimConf.from_dict({'optimizer': 'dion_muon'})
    assert not cfg.muon_use_gram_newton_schulz and not cfg.muon_use_triton
    assert not cfg.muon_replica_sync and cfg.muon_weight_decay is None
    args = OptimConf.to_cli().parse_args([
        '--optimizer', 'dion_muon',
        '--muon_use_gram_newton_schulz', 'true', '--muon_use_triton', 'true',
        '--muon_replica_sync', 'true', '--muon_weight_decay', '0.05',
    ])
    cfg = OptimConf.from_cli(vars(args))
    assert cfg.muon_use_gram_newton_schulz and cfg.muon_use_triton
    assert cfg.muon_replica_sync and cfg.muon_weight_decay == 0.05
