"""Compare a run's model config with the config.json saved next to a base model.

``--base_model_dir`` loads weights only. The model config comes from the ConfStore entry named by
``--model`` plus CLI overrides, so a field that changes the forward pass without changing any
tensor shape (``rope_base`` is the usual one) can differ from the checkpoint without any error.
``check_base_model_config`` fails the run on such a mismatch.

This module only uses the standard library, so it can be imported and tested without torch.
"""
import json
import math
import sys
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = getLogger()

CONFIG_NAME = "config.json"
_MISSING = object()

# Fields compared as they are.
_PLAIN_FIELDS = (
    "arch", "num_layers", "model_dim", "num_heads", "qknorm", "attn_act_func", "apply_attn_gate",
    "swiglu", "vocab_size", "layernorm_num_groups", "apply_rmsnorm", "norm_affine", "embed_multiplier",
    "full_attn_layer_list", "apply_nope_in_causal_attn", "num_experts", "num_values",
)
# Fields that only matter when MoE is on.
_MOE_FIELDS = ("num_activated_experts", "num_shared_experts", "expert_inter_dim",
               "moe_router_score_func", "moe_router_bias")
# Compared name -> CLI fields that set it, when they differ from the compared name.
_SOURCE_FIELDS = {"norm_eps": ("rmsnorm_eps", "layernorm_eps")}


class BaseModelConfigMismatch(ValueError):
    pass


def explicit_model_fields(argv: Optional[Sequence[str]] = None, prefix: str = "--model.") -> Set[str]:
    """Names of the model fields set on the command line (``--model.rope_base 1e7`` -> ``rope_base``)."""
    argv = sys.argv[1:] if argv is None else argv
    return {arg[len(prefix):].split("=", 1)[0] for arg in argv if arg.startswith(prefix)}


def effective_model_fields(model: Mapping[str, Any]) -> Dict[str, Any]:
    """The values that define the forward pass, with defaults resolved and unused fields dropped.

    Fields absent from ``model`` are left out, so a partial config (for example one written by a
    checkpoint converter) is compared on the fields it has.
    """
    def get(name):
        return model.get(name, _MISSING)

    out: Dict[str, Any] = {name: get(name) for name in _PLAIN_FIELDS}

    num_heads, model_dim, num_layers = get("num_heads"), get("model_dim"), get("num_layers")

    kv_heads = get("num_kv_heads")
    out["num_kv_heads"] = num_heads if kv_heads is None else kv_heads

    head_dim = get("head_dim")
    if head_dim is None:
        head_dim = _MISSING if _MISSING in (model_dim, num_heads) else model_dim // num_heads
    out["head_dim"] = head_dim

    rope_head_dim = get("rope_head_dim")
    if rope_head_dim is None:
        rope_head_dim = head_dim
    out["rope_head_dim"] = rope_head_dim
    if rope_head_dim != 0:  # 0 disables RoPE
        out["rope_base"] = get("rope_base")

    if get("apply_attn_gate") is True:
        out["attn_gate_func"] = get("attn_gate_func")

    residual_func = get("residual_func")
    out["residual_func"] = "base" if residual_func == "add" else residual_func

    apply_rmsnorm = get("apply_rmsnorm")
    if apply_rmsnorm is not _MISSING:
        eps = get("rmsnorm_eps" if apply_rmsnorm else "layernorm_eps")
        out["norm_eps"] = get("norm_eps") if eps is _MISSING else eps

    num_experts = get("num_experts")
    if num_experts is not _MISSING:
        has_moe = num_experts > 0
        dense_layers = get("num_dense_layers")
        if dense_layers is None:
            dense_layers = 0 if has_moe else num_layers
        out["num_dense_layers"] = dense_layers
        if dense_layers is _MISSING or dense_layers > 0:
            out["ffn_hidden_dim"] = get("ffn_hidden_dim")
        if has_moe:
            for name in _MOE_FIELDS:
                out[name] = get(name)
            scaling = get("moe_router_scaling_factor")
            out["moe_router_scaling_factor"] = 1.0 if scaling is None else scaling  # None means no scaling

    num_values = get("num_values")
    if num_values is not _MISSING and num_values > 0:
        out["num_activated_values"] = get("num_activated_values")

    return {name: value for name, value in out.items() if value is not _MISSING}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    return a == b


def find_model_config_mismatches(
    run_model: Mapping[str, Any],
    base_model: Mapping[str, Any],
    explicit_fields: Iterable[str] = (),
) -> List[Tuple[str, Any, Any]]:
    """Returns ``(field, run value, base model value)`` for every field that differs.

    Fields named in ``explicit_fields`` are skipped: the user set them on purpose.
    """
    explicit = set(explicit_fields)
    run, base = effective_model_fields(run_model), effective_model_fields(base_model)
    mismatches = []
    for name in sorted(run.keys() & base.keys()):
        if explicit & {name, *_SOURCE_FIELDS.get(name, ())}:
            continue
        if not _same(run[name], base[name]):
            mismatches.append((name, run[name], base[name]))
    return mismatches


def check_base_model_config(
    base_model_dir: str,
    run_model: Mapping[str, Any],
    explicit_fields: Iterable[str] = (),
) -> None:
    """Raises ``BaseModelConfigMismatch`` when the run's model config differs from the base model's.

    ``run_model`` is the run's model config as a dict (``cfg.model.to_dict()``). The check is skipped
    with a warning when ``base_model_dir`` has no usable ``config.json``.
    """
    config_path = Path(base_model_dir) / CONFIG_NAME
    if not config_path.exists():
        logger.warning(f"No {CONFIG_NAME} in {base_model_dir}: the model config is not checked against the base model.")
        return

    try:
        with config_path.open("r") as fp:
            base_model = json.load(fp).get("model")
    except (OSError, ValueError, AttributeError) as e:
        logger.warning(f"Could not read {config_path} ({e}): the model config is not checked against the base model.")
        return
    if not isinstance(base_model, Mapping):
        logger.warning(f"{config_path} has no 'model' section: the model config is not checked against the base model.")
        return

    mismatches = find_model_config_mismatches(run_model, base_model, explicit_fields)
    if not mismatches:
        logger.info(f"Model config matches {config_path}.")
        return

    lines = [f"  model.{name}: run has {run_value!r}, base model has {base_value!r}"
             for name, run_value, base_value in mismatches]
    raise BaseModelConfigMismatch(
        f"The model config differs from the base model's {config_path}:\n"
        + "\n".join(lines)
        + "\nPass the base model's value (for example --model.{0} {1}) to match it. A field set explicitly "
          "on the command line is not checked, so pass the run's value to keep a difference on purpose."
          .format(mismatches[0][0], mismatches[0][2])
    )
