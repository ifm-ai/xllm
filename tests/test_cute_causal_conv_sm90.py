"""Hopper CuTe DSL causal depthwise conv1d (cute/causal_conv_sm90.py, forward and backward) against the repo's
reference implementation."""
import importlib.util
import os
import sys
import types

import pytest
import torch

sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
cv = pytest.importorskip("xllm.modules.fused_ops.cute.causal_conv_sm90")
_rspec = importlib.util.spec_from_file_location(
    "causal_conv1d_reference", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xllm", "modules", "fused_ops", "conv", "triton",
                                            "causal_conv1d_reference.py"))
_ref = importlib.util.module_from_spec(_rspec)
_rspec.loader.exec_module(_ref)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason="needs Hopper or newer")


@pytest.mark.parametrize("B,L,D,act,bosp", [(2, 512, 2048, None, 0.01), (2, 300, 1024, "silu", 0.02), (1, 128, 8192, "silu", 0.0),
                                            (3, 100, 256, "silu", 0.1), (1, 35, 512, None, 0.0), (2, 1000, 1024, "silu", 0.3)])
def test_causal_conv_sm90(B, L, D, act, bosp):
    torch.manual_seed(0)
    x = torch.randn(B, L, D, device="cuda").to(torch.bfloat16)
    w = torch.softmax(torch.randn(D, 4, device="cuda"), -1)
    bos = None
    if bosp > 0:
        bos = torch.rand(B, L, device="cuda") < bosp
        bos[:, 0] = True
    g = torch.randn_like(x)
    out_ref, _ = _ref.causal_conv1d_reference_fwd(x, w, None, None, bos, act)
    out = cv.conv_act_fwd(x, w, bos, act)
    assert torch.allclose(out.float(), out_ref.float(), atol=2e-2, rtol=1e-2)
    dx_ref, _, dw_ref, _ = _ref.causal_conv1d_reference_bwd(g, None, x, w, None, None, bos, act)
    dx, dw = cv.conv_act_bwd(g, x, w, bos, act)
    assert torch.allclose(dx.float(), dx_ref.float(), atol=2e-2, rtol=1e-2)
    assert torch.allclose(dw, dw_ref, atol=1e-3 * dw_ref.abs().max().item(), rtol=1e-3)
