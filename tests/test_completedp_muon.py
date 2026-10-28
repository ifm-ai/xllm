"""Complete(d)P with Muon and the depth scale c folded into wo / fc2 (MuPConf, xllm/optim/mup.py). CPU.

    pytest tests/test_completedp_muon.py

Expected values are written out by hand, not recomputed with the formulas under test.
"""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from xllm.config import (
    DataLoaderConfig, ModelConf, MuPConf, OptimConf, SlurmConf, TokenizerConf, TrainerConf, ValidConf,
)


def mup_conf(**kw):
    """m_N = 512 / 256 = 2, m_L = 16 / 4 = 4 with model_conf(); alpha 0.5 gives c = 4^-0.5 = 0.5."""
    return MuPConf(**dict(base_model_dim=256, base_num_layers=4, base_init_hidden_std=0.02,
                          base_init_embed_std=0.5, base_init_unembed_std=0.04, alpha=0.5) | kw)


def model_conf(**kw):
    return ModelConf(**dict(model_dim=512, num_layers=16, num_heads=8, ffn_hidden_dim=1024,
                            init_mode="gaussian") | kw)


def test_resolve_sets_multipliers_and_init_stds():
    model, mup = model_conf(), mup_conf()
    mup.resolve(model, OptimConf(optimizer="muon"))
    assert (mup.width_multiplier, mup.depth_multiplier, mup.branch_scale) == (2.0, 4.0, 0.5)
    assert math.isclose(model.init_std, 0.01414213562373095, rel_tol=1e-12)  # 0.02 / sqrt(2)
    assert math.isclose(model.init_out_std, 0.007071067811865475, rel_tol=1e-12)  # x c
    assert model.init_embed_std == 0.5
    assert model.init_logits_std == 0.02  # 0.04 / 2

    model, mup = model_conf(), mup_conf(alpha=1.0)
    mup.resolve(model, OptimConf(optimizer="muon"))
    assert mup.branch_scale == 0.25  # L0 / L


def test_resolve_is_idempotent_but_rejects_conflicting_init():
    model, mup = model_conf(), mup_conf()
    mup.resolve(model, OptimConf(optimizer="muon"))
    mup.resolve(model, OptimConf(optimizer="muon"))  # a reloaded config.json holds the resolved values
    with pytest.raises(ValueError, match="model.init_std"):
        mup_conf().resolve(model_conf(init_std=0.02), OptimConf(optimizer="muon"))


@pytest.mark.parametrize("model_kw, optim_kw, mup_kw, match", [
    ({"init_mode": "he"}, {}, {}, "init_mode"),
    ({"arch": "gekko"}, {}, {}, "arch"),
    ({"num_experts": 4, "num_activated_experts": 1, "expert_inter_dim": 64}, {}, {}, "num_experts"),
    ({"residual_func": "ortho", "residual_heads": 8}, {}, {}, "residual_func"),
    ({"scale_emb": True}, {}, {}, "scale_emb"),
    ({}, {"optimizer": "adamw"}, {}, "optimizer"),
    ({}, {"muon_adjust_lr": "rms_norm"}, {}, "muon_adjust_lr"),
    ({}, {"muon_block_scale": "matrix"}, {}, "muon_block_scale"),
    ({}, {}, {"alpha": 0.4}, "alpha"),
    ({}, {}, {"base_num_layers": 0}, "base_num_layers"),
])
def test_resolve_rejects_unsupported(model_kw, optim_kw, mup_kw, match):
    with pytest.raises(ValueError, match=match):
        mup_conf(**mup_kw).resolve(model_conf(**model_kw), OptimConf(**dict(optimizer="muon") | optim_kw))


def trainer_conf(**kw):
    return TrainerConf(slurm=SlurmConf(), model=model_conf(), optim=OptimConf(optimizer="muon"), valid=ValidConf(),
                       tokenizer=TokenizerConf(), dataloader=DataLoaderConfig(), dcp_for_optimizer=True, **kw)


def test_trainer_conf_resolves_from_cli_and_config_json():
    args = ["--slurm.partition", "main", "--optim.optimizer", "muon", "--dcp_for_optimizer", "true", "--model.model_dim", "512",
            "--model.num_layers", "16", "--model.num_heads", "8", "--model.init_mode", "gaussian",
            "--mup.base_model_dim", "256", "--mup.base_num_layers", "4", "--mup.alpha", "0.5",
            "--mup.base_init_hidden_std", "0.02", "--mup.base_init_embed_std", "0.5",
            "--mup.base_init_unembed_std", "0.04"]
    cfg = TrainerConf.from_cli(vars(TrainerConf.to_cli().parse_args(args)))
    assert cfg.mup.branch_scale == 0.5
    assert math.isclose(cfg.model.init_out_std, 0.007071067811865475, rel_tol=1e-12)
    # train.py writes config.json with to_dict(); reloading resolves again to the same values
    reloaded = TrainerConf.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert reloaded.mup == cfg.mup and reloaded.model == cfg.model
    assert trainer_conf().mup is None and trainer_conf().model.init_out_std is None


def test_out_projection_init_is_c_times_plain_init():
    """The real Transformer: wo / fc2 are exactly c x the init without mup's c (c a power of two), the rest equal."""
    from xllm.models.transformer import Transformer
    cfg = model_conf(model_dim=128, num_layers=4, num_heads=4, ffn_hidden_dim=256, vocab_size=64,
                     apply_rmsnorm=True, qknorm=True, swiglu=True)
    mup = mup_conf(base_model_dim=64, base_num_layers=1, alpha=1.0)
    mup.resolve(cfg, OptimConf(optimizer="muon"))
    assert mup.branch_scale == 0.25
    plain = copy.deepcopy(cfg)
    plain.init_out_std = None
    tokenizer = SimpleNamespace(bos_id=1, eos_id=2, pad_id=0)
    torch.manual_seed(0)
    folded = Transformer(cfg, tokenizer)
    torch.manual_seed(0)
    reference = Transformer(plain, tokenizer)
    n_out = 0
    for (name, a), (_, b) in zip(reference.named_parameters(), folded.named_parameters()):
        if name.endswith(("attention.wo.weight", "nffn.fc2.weight")):
            n_out += 1
            assert torch.equal(b, a * 0.25), name
        else:
            assert torch.equal(b, a), name
    assert n_out == 2 * 4
