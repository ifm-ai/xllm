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
from xllm.optim.dion_muon import parameter_groups, validate_optimizer_contract
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
    model.output = nn.Linear(8, 16)
    return model


def test_hybrid_routing_keeps_convolution_norm_embedding_and_output_in_adamw():
    model = routing_model()
    groups, contract = parameter_groups(model, OptimConf(optimizer='dion_muon'))
    muon_ids = {id(p) for p in groups[0]['params']}
    expected = {
        'layers.0.attention.wq.weight',
        'layers.1.gda.wq.weight',
        'layers.1.nffn.fc1.weight',
    }
    assert {n for n, p in model.named_parameters() if id(p) in muon_ids} == expected
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))
    assert [g['algorithm'] for g in contract] == ['muon', 'adamw']
    assert groups[0]['lr'] != groups[1]['lr']


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
    ],
)
def test_muon_rejects_unqualified_execution(overrides):
    with pytest.raises(ValueError):
        trainer(**overrides)


@pytest.mark.parametrize('interval', [-1, 1, True, 4.0])
def test_invalid_hybrid_interval(interval):
    with pytest.raises(ValueError):
        ModelConf(arch='gekko', full_attention_interval=interval)


def test_hybrid_is_opt_in_and_keeps_full_attention_head_geometry():
    assert ModelConf(arch='gekko').full_attention_interval == 0
    assert ModelConf(arch='gekko', full_attention_interval=4).full_attention_interval == 4
    with pytest.raises(ValueError):
        ModelConf(full_attention_interval=4)
    with pytest.raises(ValueError):
        ModelConf(arch='gekko', full_attention_interval=4, head_dim=128, v_head_dim=256)


def test_scheduler_resume_preserves_both_group_trajectories():
    cfg = OptimConf(optimizer='dion_muon', muon_lr=0.02, lr=0.001, warmup=3)
    parameters = [nn.Parameter(torch.ones(2, 2)), nn.Parameter(torch.ones(2))]

    def build():
        optimizer = torch.optim.SGD(
            [
                {'params': [parameters[0]], 'lr': cfg.muon_lr},
                {'params': [parameters[1]], 'lr': cfg.lr},
            ]
        )
        return optimizer, build_lr_scheduler(optimizer, cfg, 20)

    optimizer, scheduler = build()
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    optimizer_state, scheduler_state = (
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    resumed, resumed_scheduler = build()
    resumed.load_state_dict(optimizer_state)
    load_lr_scheduler_state(resumed_scheduler, scheduler_state, cfg)
    for _ in range(5):
        optimizer.step()
        resumed.step()
        scheduler.step()
        resumed_scheduler.step()
        assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert scheduler_state['base_lrs'] == [cfg.muon_lr, cfg.lr]


def test_contract_rejects_drift_and_keeps_legacy_adamw_resume():
    optimizer = SimpleNamespace(_xllm_optimizer_contract={'groups': ['muon', 'adamw']})
    validate_optimizer_contract(optimizer, {'optimizer_contract': {'groups': ['muon', 'adamw']}})
    with pytest.raises(ValueError):
        validate_optimizer_contract(optimizer, {})
    with pytest.raises(ValueError):
        validate_optimizer_contract(
            SimpleNamespace(), {'optimizer_contract': optimizer._xllm_optimizer_contract}
        )
    validate_optimizer_contract(SimpleNamespace(), {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native RoPE kernel requires CUDA')
def test_partial_fused_rope_preserves_normalization_inputs():
    from xllm.models.fused_blocks.utils import apply_rope, apply_ropes

    query = torch.randn(1, 16, 4, 128, device='cuda', dtype=torch.bfloat16)
    key = torch.randn(1, 16, 1, 128, device='cuda', dtype=torch.bfloat16)
    original_query, original_key = query.clone(), key.clone()
    phase = torch.randn(16, 32, device='cuda')
    frequencies = torch.polar(torch.ones_like(phase), phase)
    rotated_query, rotated_key = apply_ropes(query, key, frequencies, 128, 64, False)
    torch.testing.assert_close(query, original_query, rtol=0, atol=0)
    torch.testing.assert_close(key, original_key, rtol=0, atol=0)
    torch.testing.assert_close(rotated_query[..., 64:], query[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(rotated_key[..., 64:], key[..., 64:], rtol=0, atol=0)
    assert not torch.equal(rotated_key[..., :64], key[..., :64])
    saved_gradient = rotated_key.clone()
    apply_rope(rotated_key, frequencies, 128, 64, True)
    torch.testing.assert_close(rotated_key, saved_gradient, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native xattn requires CUDA')
@pytest.mark.parametrize('use_bos_mask', [False, True])
def test_packed_xattn_trims_metadata_along_sequence_axis(use_bos_mask):
    from xllm.models.fused_blocks.utils import flash_attention_fwd, xattn_fwd

    query = torch.randn(2, 128, 4, 128, device='cuda', dtype=torch.bfloat16)
    key = torch.randn(2, 256, 1, 128, device='cuda', dtype=torch.bfloat16)
    value = torch.randn_like(key)
    bos_mask = torch.zeros(2, 256, device='cuda', dtype=torch.bool)
    bos_mask[:, 0] = True
    bos_mask[:, 128] = True
    segments = bos_mask.cumsum(dim=1, dtype=torch.int64)
    bos_mask = bos_mask if use_bos_mask else None
    scale = 128 ** -0.5
    actual = flash_attention_fwd(
        query, key, value, bos_mask, (segments, 128), scale, 0, 0,
        backend='xattn', save_aux=True,
    )
    expected = xattn_fwd(
        query, key[:, -128:], value[:, -128:], scale,
        bos_mask[:, -128:] if bos_mask is not None else None,
        segments[:, -128:], 0, requires_grad=True,
    )
    for result, reference in zip(actual[:3], expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
