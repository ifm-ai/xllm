import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("block", ["moe", "mova"])
def test_fused_routed_blocks_forward_layer_identity_and_balance_backward(block):
    path = ROOT / "xllm/models/fused_blocks/transformer" / f"{block}.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    backward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "backward")
    returned = next(node.value for node in reversed(backward.body) if isinstance(node, ast.Return))
    assert forward.args.args[-1].arg == "layer_id"
    assert len(returned.elts) == len(forward.args.args) - 1

    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "record_query_key_stats"]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == ["layer_id", "xq", "xk"]
    # These are the local tensors returned by the attention helper, not CP-expanded keys.
    unpack = next(node for node in forward.body if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == "(xq, xk, xv)")
    assert ast.unparse(unpack.value) == "xqkv"
    assert unpack.lineno < calls[0].lineno


def test_both_routed_model_callers_supply_layer_id():
    tree = ast.parse((ROOT / "xllm/models/transformer.py").read_text())
    for name in ("TransformerMoEBlock", "TransformerMoVABlock"):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        calls = [node for node in ast.walk(cls) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "apply"]
        assert len(calls) == 1
        assert ast.unparse(calls[0].args[-1]) == "self.layer_id"
