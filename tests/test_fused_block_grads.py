"""Fused blocks against the unfused modules: same loss and the same gradient for every parameter.

The fused blocks (model.fused_block) reimplement the blocks' forward and backward by hand. A model built with them and
the same model built without them, from the same weights, are run on the same tokens; the LM loss and the aux loss
are backpropagated separately (each with the other times 0, so the graph is the same), and every parameter's
gradient is compared. Fused kernels round differently in bf16, so the check is a relative tolerance per parameter;
a wrong backward (e.g. an aux-loss gradient halved per layer) is far outside it.

Needs a GPU (flash attention, te experts / values): pytest tests/test_fused_block_grads.py
Run as a script to print the measured gaps: python tests/test_fused_block_grads.py
"""
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from xllm.config import ModelConf
from xllm.distributed import destroy_model_parallel, initialize_model_parallel

TOK = SimpleNamespace(bos_id=1, eos_id=2, pad_id=0)
BASE = dict(model_dim=256, num_layers=4, num_heads=4, ffn_hidden_dim=512, vocab_size=512, swiglu=True,
            init_mode="gaussian", init_std=0.02, chunk_size=64, causal_attn_backend="flash")
MOE = dict(apply_rmsnorm=True, num_dense_layers=1, num_experts=4, num_activated_experts=2, num_shared_experts=1,
           expert_inter_dim=128, moe_router_score_func="sigmoid", moe_router_bias=True,
           moe_router_bias_update_rate=1e-3, moe_router_scaling_factor=2.5)
MOVA = MOE | dict(num_values=4, num_activated_values=2, apply_attn_gate=True)
CASES = {  # name: (model overrides, load balancing type)
    "dense layernorm": (dict(), None),
    "dense rmsnorm qknorm gate": (dict(apply_rmsnorm=True, qknorm=True, apply_attn_gate=True, layernorm_num_groups=2), None),
    "dense ortho residual": (dict(apply_rmsnorm=True, residual_func="ortho", residual_heads=4), None),
    "moe dot": (MOE, "dot"),
    "moe no aux loss": (MOE, None),
    "mova dot": (MOVA, "dot"),
    "mova no aux loss": (MOVA, None),
    "gekko": (dict(arch="gekko", init_mode="he", init_std=None, sca_backend="swift"), None),
}
# The fused blocks add the residuals plainly and ignore residual_func="ortho"; drop this once they apply it.
KNOWN_MISMATCH = {"dense ortho residual": "fused blocks ignore residual_func='ortho'"}
RTOL = 0.05  # per-parameter relative gradient error, fused vs unfused; measured at most 0.01 on one H100


@pytest.fixture(scope="module")
def one_rank():
    dist.init_process_group("nccl", rank=0, world_size=1, store=dist.HashStore())
    initialize_model_parallel(1, 1, timeout=600)
    yield
    destroy_model_parallel()
    dist.destroy_process_group()


def build(overrides, fused):
    kw = BASE | overrides | dict(fused_block=fused)
    if kw.get("arch") == "gekko":
        from xllm.models.gekko import Gekko as cls
    else:
        from xllm.models.transformer import Transformer as cls
    torch.manual_seed(0)
    return cls(ModelConf(**kw), TOK).cuda().to(torch.bfloat16).train()


def losses_and_grads(model, buffers, tokens, targets, lb_type, part):
    """(loss value, {name: grad}) with `part` ("lm" or "aux") backpropagated and the other times 0."""
    with torch.no_grad():  # the router bias updates in backward: start each pass from the same buffers
        for name, buf in model.named_buffers():
            buf.copy_(buffers[name])
    model.zero_grad(set_to_none=True)
    out, aux, _ = model(tokens, True, targets, None, lb_type, 0, True)
    lm = out.float().mean()
    if aux is None:
        loss, value = lm, lm
    else:
        aux = aux.float()
        loss, value = (lm + 0 * aux, lm) if part == "lm" else (0 * lm + aux, aux)
    loss.backward()
    return value.item(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}


def compare(name):
    """[(part, loss_unfused, loss_fused, worst relative gradient error, its parameter, missing grads)]"""
    overrides, lb_type = CASES[name]
    unfused, fused = build(overrides, False), build(overrides, True)
    fused.load_state_dict(unfused.state_dict())
    buffers = {n: b.clone() for n, b in unfused.named_buffers()}
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(3, BASE["vocab_size"], (2, 256), generator=g).cuda()
    tokens[:, 100] = TOK.eos_id  # two segments
    targets = torch.roll(tokens, -1, 1)
    rows = []
    for part in ("lm", "aux") if lb_type is not None else ("lm",):
        loss_u, grads_u = losses_and_grads(unfused, buffers, tokens, targets, lb_type, part)
        loss_f, grads_f = losses_and_grads(fused, buffers, tokens, targets, lb_type, part)
        worst, worst_name = 0.0, ""
        for n, gu in grads_u.items():
            if n not in grads_f or gu.norm() == 0:
                continue
            err = ((grads_f[n] - gu).norm() / gu.norm()).item()
            if err > worst:
                worst, worst_name = err, n
        missing = sorted(set(grads_u) ^ set(grads_f))
        rows.append((part, loss_u, loss_f, worst, worst_name, missing))
    return rows


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused blocks need a GPU")


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(reason=KNOWN_MISMATCH[n], strict=True)) if n in KNOWN_MISMATCH else n
    for n in CASES
])
def test_fused_blocks_match_the_unfused_modules(one_rank, name):
    for part, loss_u, loss_f, worst, worst_name, missing in compare(name):
        assert not missing, f"{name} {part}: gradient only on one side: {missing}"
        assert loss_f == pytest.approx(loss_u, rel=1e-2), f"{name} {part}: loss fused {loss_f} unfused {loss_u}"
        assert worst <= RTOL, f"{name} {part}: {worst_name} gradient differs by {worst:.3f} (relative)"


if __name__ == "__main__":
    dist.init_process_group("nccl", rank=0, world_size=1, store=dist.HashStore())
    initialize_model_parallel(1, 1, timeout=600)
    for name in sys.argv[1:] or CASES:
        try:
            for part, loss_u, loss_f, worst, worst_name, missing in compare(name):
                print(f"{name:28s} {part:3s} loss {loss_u:.5f} / {loss_f:.5f}  worst grad rel {worst:.2e} ({worst_name})"
                      f"{'  MISSING ' + str(missing) if missing else ''}", flush=True)
        except Exception as e:
            print(f"{name:28s} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
