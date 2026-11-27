import json
import logging

import pytest

from xllm.base_model_check import (
    BaseModelConfigMismatch,
    check_base_model_config,
    explicit_model_fields,
    find_model_config_mismatches,
)


def run_model(**overrides):
    # the model fields of ConfStore["k2-horizon-3.7B"], as cfg.model.to_dict() gives them
    model = dict(
        arch="transformer", num_layers=36, model_dim=2560, num_heads=32, num_kv_heads=8, head_dim=128,
        qknorm=True, attn_act_func="softmax", apply_attn_gate=False, attn_gate_func="softplus",
        ffn_hidden_dim=10240, swiglu=True, vocab_size=151936,
        num_values=0, num_activated_values=0, num_experts=0, num_activated_experts=0, num_shared_experts=0,
        num_dense_layers=None, expert_inter_dim=0, moe_router_score_func="sigmoid", moe_router_bias=False,
        moe_router_bias_update_rate=None, moe_router_scaling_factor=None,
        layernorm_num_groups=2, apply_rmsnorm=True, rmsnorm_eps=1e-6, layernorm_eps=1e-5, norm_affine=True,
        rope_base=500000, rope_head_dim=None, residual_func="base", embed_multiplier=1.0,
        fused_block=True, init_mode="gaussian", init_std=0.02, causal_attn_backend="flash",
    )
    model.update(overrides)
    return model


def converted_model(**overrides):
    # what the xBridges HF -> xLLM converter writes for the K2 Horizon 3.7B release
    model = {
        "arch": "transformer", "num_layers": 36, "model_dim": 2560, "num_heads": 32, "num_kv_heads": 8,
        "head_dim": 128, "qknorm": True, "ffn_hidden_dim": 10240,
        "num_experts": 0, "num_activated_experts": 8, "num_shared_experts": 0, "num_dense_layers": None,
        "expert_inter_dim": 768, "moe_router_score_func": "softmax", "moe_router_bias": False,
        "moe_router_bias_update_rate": None, "moe_router_scaling_factor": 1.0,
        "vocab_size": 151936, "layernorm_num_groups": 2, "norm_eps": 1e-6, "apply_rmsnorm": True,
        "rope_base": 10000000.0, "rope_head_dim": None, "attention_dropout": 0.0, "swiglu": True,
        "num_values": 0, "num_activated_values": 0, "apply_attn_gate": False, "attn_gate_func": None,
        "init_std": 0.02, "fused_block": False, "init_mode": "none",
    }
    model.update(overrides)
    return model


def moe_fields(**overrides):
    fields = dict(
        num_experts=128, num_activated_experts=8, num_shared_experts=1, num_dense_layers=3, expert_inter_dim=768,
        moe_router_score_func="sigmoid", moe_router_bias=True, moe_router_scaling_factor=2.5,
    )
    fields.update(overrides)
    return fields


def write_config(tmp_path, model):
    (tmp_path / "config.json").write_text(json.dumps({"model": model, "seq_len": 32768}))
    return str(tmp_path)


def test_rope_base_of_a_converted_release_is_reported():
    # the case of the issue: ConfStore has 5e5, the release has 1e7, and nothing else differs
    assert find_model_config_mismatches(run_model(), converted_model()) == [("rope_base", 500000, 10000000.0)]


def test_matching_config_has_no_mismatch():
    assert find_model_config_mismatches(run_model(rope_base=1e7), converted_model()) == []


def test_training_only_fields_are_not_compared():
    base = converted_model(rope_base=500000)
    # fused_block, init_mode and backends differ between the two and must not be reported
    assert find_model_config_mismatches(run_model(), base) == []


def test_explicit_field_is_not_checked():
    assert find_model_config_mismatches(run_model(), converted_model(), explicit_fields={"rope_base"}) == []
    # an explicit field does not hide another mismatch
    mismatches = find_model_config_mismatches(
        run_model(layernorm_num_groups=4), converted_model(), explicit_fields={"rope_base"}
    )
    assert mismatches == [("layernorm_num_groups", 4, 2)]


def test_defaults_are_resolved_before_comparing():
    base = converted_model(rope_base=500000)
    # head_dim=None means model_dim // num_heads, rope_head_dim=None means head_dim
    assert find_model_config_mismatches(run_model(model_dim=4096, head_dim=None), dict(base, model_dim=4096)) == []
    assert find_model_config_mismatches(run_model(rope_head_dim=128), base) == []
    assert find_model_config_mismatches(run_model(rope_head_dim=64), base) == [("rope_head_dim", 64, 128)]
    # num_kv_heads=None means num_heads
    assert find_model_config_mismatches(run_model(num_kv_heads=None), base) == [("num_kv_heads", 32, 8)]
    # 'add' is another name of the 'base' residual
    assert find_model_config_mismatches(run_model(residual_func="add"), dict(base, residual_func="base")) == []


def test_rope_base_is_ignored_when_rope_is_off():
    assert find_model_config_mismatches(run_model(rope_head_dim=0), converted_model(rope_head_dim=0)) == []


def test_norm_eps_is_compared_under_both_names():
    base = converted_model(rope_base=500000, norm_eps=1e-5)
    assert find_model_config_mismatches(run_model(), base) == [("norm_eps", 1e-6, 1e-5)]
    assert find_model_config_mismatches(run_model(), base, explicit_fields={"rmsnorm_eps"}) == []
    # a native checkpoint config has rmsnorm_eps / layernorm_eps, and only the one in use counts
    native = run_model(layernorm_eps=1e-3)
    assert find_model_config_mismatches(run_model(), native) == []
    assert find_model_config_mismatches(run_model(), run_model(rmsnorm_eps=1e-5)) == [("norm_eps", 1e-6, 1e-5)]


def test_moe_fields_are_compared_only_with_moe():
    # dense models: the converter writes expert defaults of the HF config, which are unused
    assert find_model_config_mismatches(run_model(rope_base=1e7), converted_model(expert_inter_dim=1792)) == []

    run = run_model(**moe_fields())
    assert find_model_config_mismatches(run, run_model(**moe_fields())) == []
    assert find_model_config_mismatches(run, run_model(**moe_fields(num_activated_experts=4))) == [
        ("num_activated_experts", 8, 4)
    ]
    assert find_model_config_mismatches(run, run_model(**moe_fields(moe_router_score_func="softmax"))) == [
        ("moe_router_score_func", "sigmoid", "softmax")
    ]
    # no scaling factor is a factor of 1
    no_scaling = run_model(**moe_fields(moe_router_scaling_factor=None))
    assert find_model_config_mismatches(no_scaling, run_model(**moe_fields(moe_router_scaling_factor=1.0))) == []
    assert find_model_config_mismatches(no_scaling, run) == [("moe_router_scaling_factor", 1.0, 2.5)]
    # the router bias update rate only matters for training
    assert find_model_config_mismatches(run, run_model(**moe_fields(), moe_router_bias_update_rate=1e-3)) == []


def test_num_dense_layers_default():
    # None means every layer without MoE and no layer with MoE
    assert find_model_config_mismatches(run_model(), run_model(num_dense_layers=None)) == []
    moe = run_model(**moe_fields(num_dense_layers=None))
    assert find_model_config_mismatches(moe, run_model(**moe_fields(num_dense_layers=0))) == []
    assert find_model_config_mismatches(moe, run_model(**moe_fields(num_dense_layers=3))) == [
        ("num_dense_layers", 0, 3)
    ]
    # the dense FFN size is unused when every layer is MoE
    assert find_model_config_mismatches(moe, run_model(**moe_fields(num_dense_layers=0), ffn_hidden_dim=1)) == []


def test_attention_gate_function_is_compared_only_with_the_gate():
    assert find_model_config_mismatches(run_model(attn_gate_func="silu"), run_model()) == []
    gated = run_model(apply_attn_gate=True, attn_gate_func="silu")
    assert find_model_config_mismatches(gated, run_model(apply_attn_gate=True)) == [
        ("attn_gate_func", "silu", "softplus")
    ]


def test_bool_and_number_are_not_confused():
    assert find_model_config_mismatches(run_model(qknorm=1), converted_model(rope_base=500000)) == [
        ("qknorm", 1, True)
    ]


def test_check_raises_and_names_the_field(tmp_path):
    base_dir = write_config(tmp_path, converted_model())
    with pytest.raises(BaseModelConfigMismatch) as err:
        check_base_model_config(base_dir, run_model())
    message = str(err.value)
    assert "model.rope_base: run has 500000, base model has 10000000.0" in message
    assert "--model.rope_base 10000000.0" in message

    check_base_model_config(base_dir, run_model(rope_base=1e7))
    check_base_model_config(base_dir, run_model(), explicit_fields={"rope_base"})


def test_check_reports_every_mismatch(tmp_path):
    base_dir = write_config(tmp_path, converted_model())
    with pytest.raises(BaseModelConfigMismatch) as err:
        check_base_model_config(base_dir, run_model(num_heads=16, num_kv_heads=4))
    message = str(err.value)
    for name in ("model.num_heads", "model.num_kv_heads", "model.rope_base"):
        assert name in message


@pytest.mark.parametrize("content", [None, "not json", "[]", json.dumps({"seq_len": 8192})])
def test_check_is_skipped_without_a_usable_config(tmp_path, caplog, content):
    if content is not None:
        (tmp_path / "config.json").write_text(content)
    with caplog.at_level(logging.WARNING):
        check_base_model_config(str(tmp_path), run_model())
    assert "not checked against the base model" in caplog.text


def test_explicit_model_fields():
    argv = ["--model", "k2-horizon-3.7B", "--model.rope_base", "10000000", "--model.rope_head_dim=128",
            "--optim.adamw.lr", "3e-4", "--base_model_dir", "/ckpt"]
    assert explicit_model_fields(argv) == {"rope_base", "rope_head_dim"}
    assert explicit_model_fields(["--model", "k2-horizon-7B"]) == set()
