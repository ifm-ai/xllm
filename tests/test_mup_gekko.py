"""CPU tests for muP / CompleteP with dion_muon (alex.moreno/mup-completep-gekko). Run: pytest tests/test_mup_gekko.py"""
import math

import pytest
import torch
from torch import nn

from xllm.config import ModelConf, MuPConf, OptimConf, RolesConf, RoleConf
from xllm.modules.residual.build import build_residual
from xllm.optim.dion_muon import param_subrole, parameter_groups


def gekko_model_conf(**kw):
    base = dict(arch='gekko', num_layers=8, model_dim=344, ffn_hidden_dim=896, num_heads=2, num_kv_heads=1,
                head_dim=256, v_head_dim=256, full_attention_interval=4, init_mode='he', swiglu=True,
                vocab_size=64256, timenorm_num_groups=16, causal_attn_backend='flash')
    base.update(kw)
    return ModelConf(**base)


def dion_optim(**kw):
    base = dict(optimizer='dion_muon', lr=2e-3, muon_lr=1e-3, muon_adjust_lr='spectral_norm', weight_decay=0.2,
                epsilon=1e-8, scheduler='wsd', wsd_decay_duration_steps=10, disable_wd_emb=True, disable_wd_out=True)
    base.update(kw)
    return OptimConf(**base)


class Fake(nn.Module):
    """Parameter names as in hybrid Gekko: embed, layers.<i>.<projections / vectors>, output.final_norm, output.output."""
    arch, num_experts, num_values = 'gekko', 0, 0

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        layer = nn.Module()
        layer.wq = nn.Linear(4, 4, bias=False)
        layer.norm = nn.Module()
        layer.norm.weight = nn.Parameter(torch.ones(4))
        layer.q_conv = nn.Module()
        layer.q_conv.weight = nn.Parameter(torch.ones(4, 4))  # 2-D depthwise kernel: not a projection, AdamW
        self.layers = nn.ModuleList([layer])
        self.output = nn.Module()
        self.output.final_norm = nn.Module()
        self.output.final_norm.weight = nn.Parameter(torch.ones(4))
        self.output.output = nn.Linear(4, 10, bias=False)


def by_id(groups):
    return {g['group_id']: g for g in groups}


def test_residual_multiplier():
    r = build_residual('base', 1, 4, None, multiplier=0.5)
    h, x = torch.ones(2, 4), torch.full((2, 4), 3.0)
    assert torch.equal(r(h, x), x + 0.5)
    assert torch.equal(build_residual('base', 1, 4, None)(h, x), x + 1)
    with pytest.raises(AssertionError):
        build_residual('ortho', 1, 4, 2, multiplier=0.5)


def test_default_groups_unchanged():
    groups, _ = parameter_groups(Fake(), dion_optim())
    assert [g['group_id'] for g in groups] == ['muon', 'adamw', 'adamw_no_wd']
    assert [len(g['params']) for g in groups] == [1, 3, 2]


def test_mup_factors_and_resolve():
    model, optim = gekko_model_conf(num_layers=20, model_dim=1032), dion_optim()
    mup = MuPConf(base_model_dim=344, base_num_layers=8, alpha=1.0)
    mup.resolve(model, optim)
    m_n, m_l = 1032 / 344, 20 / 8
    assert math.isclose(model.branch_scale, 1 / m_l)
    assert math.isclose(model.init_embed_std, 1 / math.sqrt(344))
    assert math.isclose(model.init_logits_std, 1 / math.sqrt(344) / m_n)
    f = optim._mup_role_factors
    assert math.isclose(f['readout']['lr'], 1 / m_n) and math.isclose(f['embed']['eps'], 1 / m_n)
    assert math.isclose(f['hidden_vector']['eps'], (1 / m_l) / m_n) and f['hidden_matrix']['lr'] == 1.0
    # resolving again (a resumed config.json) is a no-op
    mup.resolve(model, optim)
    assert math.isclose(model.branch_scale, 1 / m_l)


def test_mup_alpha_half_depth_lr():
    model, optim = gekko_model_conf(num_layers=32), dion_optim()
    MuPConf(base_model_dim=344, base_num_layers=8, alpha=0.5).resolve(model, optim)
    assert math.isclose(model.branch_scale, 0.5)
    assert math.isclose(optim._mup_role_factors['hidden_matrix']['lr'], 4 ** -0.5)


def test_mup_base_shape_is_plain_run():
    model, optim = gekko_model_conf(), dion_optim()
    MuPConf(base_model_dim=344, base_num_layers=8).resolve(model, optim)
    assert model.branch_scale == 1.0 and math.isclose(model.init_logits_std, 1 / math.sqrt(344))
    groups, _ = parameter_groups(Fake(), optim)
    for g in groups:
        assert g['lr'] == (1e-3 if g['algorithm'] == 'muon' else 2e-3)
        assert g.get('epsilon', 1e-8) == 1e-8


@pytest.mark.parametrize('bad', [dict(muon_adjust_lr='rms_norm')])
def test_mup_rejects(bad):
    with pytest.raises(ValueError):
        MuPConf(base_model_dim=344, base_num_layers=8).resolve(gekko_model_conf(), dion_optim(**bad))
    with pytest.raises(ValueError):
        MuPConf(base_model_dim=344, base_num_layers=8).resolve(gekko_model_conf(init_std=0.02), dion_optim())
    with pytest.raises(ValueError):  # explicit std that disagrees
        MuPConf(base_model_dim=344, base_num_layers=8).resolve(gekko_model_conf(init_logits_std=0.5), dion_optim())


def test_role_groups_with_user_factors_and_mup():
    roles = RolesConf(readout=RoleConf(lr=4.0), embed=RoleConf(lr=0.5), hidden_matrix=RoleConf(lr=2.0, weight_decay=0.5))
    optim = dion_optim(roles=roles)
    model = gekko_model_conf(model_dim=688, num_layers=16)
    MuPConf(base_model_dim=344, base_num_layers=8).resolve(model, optim)
    groups, contract = parameter_groups(Fake(), optim)
    g = by_id(groups)
    assert set(g) == {'muon', 'adamw_embed_no_wd', 'adamw_hidden_vector', 'adamw_final_norm', 'adamw_readout_no_wd'}
    assert math.isclose(g['muon']['lr'], 1e-3 * 2.0) and math.isclose(g['muon']['weight_decay'], 0.2 * 0.5)
    assert math.isclose(g['adamw_readout_no_wd']['lr'], 2e-3 * 4.0 / 2) and g['adamw_readout_no_wd']['weight_decay'] == 0.0
    assert math.isclose(g['adamw_embed_no_wd']['lr'], 2e-3 * 0.5) and math.isclose(g['adamw_embed_no_wd']['epsilon'], 1e-8 / 2)
    assert math.isclose(g['adamw_hidden_vector']['epsilon'], 1e-8 * 0.5 / 2)  # c = 1/2, m_N = 2
    assert math.isclose(g['adamw_hidden_vector']['weight_decay'], 0.2)
    assert len(contract) == len(groups)
    names = {n for grp in groups for n in grp['params']}
    assert len(names) == len(list(Fake().parameters()))


def test_roles_require_dion():
    with pytest.raises(ValueError):
        OptimConf(optimizer='adamw', roles=RolesConf(readout=RoleConf(lr=2.0)))
    with pytest.raises(ValueError):
        RolesConf(hidden_matrix=RoleConf(eps=2.0))


def test_model_multiplier_checks():
    with pytest.raises(ValueError):
        gekko_model_conf(residual_multiplier=0.5, fused_block=True, layerwise_ckpt=False)
    with pytest.raises(ValueError):
        gekko_model_conf(residual_multiplier=0.0)
    gekko_model_conf(residual_multiplier=0.5)


class FakeBlock(nn.Module):
    """Gekko-like names: attention.{wq,wk,wv(+bias),wo,q_conv,timenorm,query_norm}, feed_forward.{fc1,fc2,fc3,norm}."""
    arch, num_experts, num_values = 'gekko', 0, 0

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        layer = nn.Module()
        attn = nn.Module()
        attn.wq, attn.wk, attn.wo = (nn.Linear(4, 4, bias=False) for _ in range(3))
        attn.wv = nn.Linear(4, 4, bias=True)
        attn.q_conv = nn.Module()
        attn.q_conv.weight = nn.Parameter(torch.ones(4, 4))
        for norm in ('timenorm', 'query_norm'):
            setattr(attn, norm, nn.Module())
            getattr(attn, norm).weight = nn.Parameter(torch.ones(4))
        ffn = nn.Module()
        ffn.fc1, ffn.fc3 = nn.Linear(4, 8, bias=False), nn.Linear(4, 8, bias=False)
        ffn.fc2 = nn.Linear(8, 4, bias=False)
        ffn.norm = nn.Module()
        ffn.norm.weight = nn.Parameter(torch.ones(4))
        layer.attention, layer.feed_forward = attn, ffn
        self.layers = nn.ModuleList([layer])
        self.output = nn.Module()
        self.output.final_norm = nn.Module()
        self.output.final_norm.weight = nn.Parameter(torch.ones(4))
        self.output.output = nn.Linear(4, 10, bias=False)


def test_param_subrole_names():
    assert param_subrole('layers.0.attention.wr.weight', 'hidden_matrix') == 'attn_in'
    assert param_subrole('layers.3.attention.wo.weight', 'hidden_matrix') == 'attn_out'
    assert param_subrole('layers.0.feed_forward.fc3.weight', 'hidden_matrix') == 'ffn_in'
    assert param_subrole('layers.0.feed_forward.fc2.weight', 'hidden_matrix') == 'ffn_out'
    assert param_subrole('layers.0.attention.k_conv.weight', 'hidden_vector') == 'vec_conv'
    assert param_subrole('layers.0.attention.timenorm.bias', 'hidden_vector') == 'vec_norm'
    assert param_subrole('layers.0.attention.attn_norm.weight', 'hidden_vector') == 'vec_norm'
    assert param_subrole('layers.0.attention.wv.bias', 'hidden_vector') == 'vec_bias'
    with pytest.raises(ValueError):
        param_subrole('layers.0.attention.mystery.weight', 'hidden_matrix')


def test_unsplit_layout_unchanged():
    roles = RolesConf(embed=RoleConf(lr=8.0), hidden_vector=RoleConf(weight_decay=0.0))
    groups, _ = parameter_groups(FakeBlock(), dion_optim(roles=roles))
    assert [g['group_id'] for g in groups] == ['muon', 'adamw_embed_no_wd', 'adamw_hidden_vector',
                                               'adamw_final_norm', 'adamw_readout_no_wd']


def test_subrole_groups():
    roles = RolesConf(hidden_matrix=RoleConf(lr=2.0), ffn_out=RoleConf(lr=0.5), attn_out=RoleConf(weight_decay=0.0),
                      hidden_vector=RoleConf(lr=8.0, weight_decay=0.0), vec_conv=RoleConf(lr=0.25))
    optim = dion_optim(roles=roles)
    groups, contract = parameter_groups(FakeBlock(), optim)
    g = by_id(groups)
    assert {k for k in g if k.startswith('muon')} == {'muon_attn_in', 'muon_attn_out', 'muon_ffn_in', 'muon_ffn_out'}
    assert len(g['muon_attn_in']['params']) == 3 and len(g['muon_ffn_in']['params']) == 2
    assert math.isclose(g['muon_ffn_out']['lr'], 1e-3 * 2.0 * 0.5)
    assert math.isclose(g['muon_attn_in']['lr'], 1e-3 * 2.0) and math.isclose(g['muon_attn_in']['weight_decay'], 0.2)
    assert g['muon_attn_out']['weight_decay'] == 0.0
    assert math.isclose(g['adamw_vec_conv']['lr'], 2e-3 * 8.0 * 0.25)
    assert math.isclose(g['adamw_vec_norm']['lr'], 2e-3 * 8.0) and math.isclose(g['adamw_vec_bias']['lr'], 2e-3 * 8.0)
    assert g['adamw_vec_norm']['weight_decay'] == 0.0
    assert len(g['adamw_vec_norm']['params']) == 3 and len(g['adamw_vec_bias']['params']) == 1
    names = sum(len(grp['params']) for grp in groups)
    assert names == len(list(FakeBlock().parameters())) and len(contract) == len(groups)


def test_subrole_rejects_muon_eps():
    with pytest.raises(ValueError):
        RolesConf(ffn_out=RoleConf(eps=2.0))
