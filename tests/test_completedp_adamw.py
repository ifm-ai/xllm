"""Complete(d)P (arXiv 2512.22382, Table 1) with AdamW through the roles: the launcher's per-role values, and the depth
factor on the embedding (model.embed_multiplier with model.stream_norm_eps) in place of the factor on the branches.

1. The roles split a dense Transformer's parameters as Table 1 does: the input embedding, the matrices in the blocks,
   the vectors in the blocks (norm gains, biases), the final norm and the readout. CPU.
2. The depth factor c on the branches (each block's wo and fc2 outputs × c) and the same model with
   embed_multiplier 1/c and the stream norms' eps × 1/c² get the same gradients (tests/test_stream_norm_eps.py), so
   AdamW with per-role values takes the same steps: with c a power of 2 the parameters after 3 steps match bit for
   bit, unfused (a forward hook on wo / fc2 is the branch factor). GPU.

    PYTHONPATH=. pytest -q tests/test_completedp_adamw.py
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from xllm.config import AdamWConf, ModelConf, OptimConf, RoleConf, RolesConf
from xllm.optim import build_optimizer
from xllm.optim.roles import param_roles

C = 0.25  # the depth factor m_L^-alpha; a power of 2
STEPS = 3
SEQ = 2048
TRANSFORMER = ModelConf(
    arch="transformer", model_dim=256, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=64, ffn_hidden_dim=512,
    swiglu=True, qknorm=True, apply_rmsnorm=True, init_mode="he", vocab_size=1024,
)
GEKKO = ModelConf(
    arch="gekko", model_dim=768, num_layers=2, num_heads=4, num_kv_heads=2, head_dim=256, v_head_dim=256,
    rope_head_dim=128, rope_base=500000, ffn_hidden_dim=2048, swiglu=True, qknorm=True, attn_gate_func="softplus",
    causal_conv_width=4, causal_conv_weight_normalization=True, causal_conv_backend="triton", sca_backend="swift",
    apply_rmsnorm=True, layernorm_num_groups=2, timenorm_num_groups=16, timenorm_beta1=0.999, timenorm_beta2=0.9999,
    init_mode="he", vocab_size=1024, chunk_size=1024,
)
BRANCH_OUT = ("gda.wo", "attention.wo", "nffn.fc2")
ADAMW = OptimConf(optimizer="adamw", adamw=AdamWConf(lr=1e-3, epsilon=1e-8, weight_decay=0.1))
# Table 1's AdamW values at m_N = 2, m_L = 4, alpha = 1 (c = m_L^-alpha = 1/4) from the base lr 1e-3, eps 1e-8, wd 0.1
ROLES = RolesConf(
    embed=RoleConf(eps=1e-8 / 2),
    hidden_matrix=RoleConf(lr=1e-3 / 2, eps=1e-8 / 4 / 2, weight_decay=0.1 * 2),
    hidden_vector=RoleConf(eps=1e-8 / 4 / 2),
    readout=RoleConf(lr=1e-3 / 2, weight_decay=0.1 * 2),
)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need a GPU")


def build(cfg, branch_scale=None):
    from xllm.models.gekko import Gekko
    from xllm.models.transformer import Transformer
    torch.manual_seed(0)
    model = (Gekko if cfg.arch == "gekko" else Transformer)(cfg, SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    if branch_scale is not None:
        hooked = [m.register_forward_hook(lambda m, i, out: out * branch_scale)
                  for n, m in model.named_modules() if n.endswith(BRANCH_OUT)]
        assert len(hooked) == 2 * cfg.num_layers, hooked
    return model


def test_roles_are_table_1s():
    model = build(TRANSFORMER)
    hidden = list(model.layers.parameters())
    want = {
        "embed": list(model.embed.parameters()),
        "hidden_matrix": [p for p in hidden if p.ndim > 1],
        "hidden_vector": [p for p in hidden if p.ndim <= 1],
        "final_norm": list(model.output.final_norm.parameters()),
        "readout": [model.output.output.weight],
    }
    role_of = param_roles(model)
    got = {role: [p for p in model.parameters() if role_of[id(p)] == role] for role in want}
    for role in want:
        assert [id(p) for p in got[role]] == [id(p) for p in want[role]], role
    assert all(role in want for role in role_of.values())


def exact(cfg):
    m = 1 / C
    eps = cfg.rmsnorm_eps if cfg.apply_rmsnorm else cfg.layernorm_eps
    return replace(cfg, embed_multiplier=m, stream_norm_eps=eps * m * m, timenorm_eps=cfg.timenorm_eps * m * m)


def train(cfg, branch_scale=None):
    model = build(cfg, branch_scale).cuda().to(torch.bfloat16).train()
    opt, scheduler = build_optimizer(model, ADAMW, 10, "fp32", ROLES)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(STEPS):
        tokens = torch.randint(3, cfg.vocab_size, (1, SEQ), generator=g).cuda()
        targets = torch.randint(3, cfg.vocab_size, (1, SEQ), generator=g).cuda()
        loss, _, _ = model(tokens=tokens, targets=targets, multi_segments=False)
        loss = loss.float().sum()
        loss.backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()
        losses.append(loss.detach())
    return torch.stack(losses), {n: p.detach() for n, p in model.named_parameters()}


@gpu
@pytest.mark.parametrize("base", [TRANSFORMER, GEKKO], ids=["transformer", "gekko"])
def test_adamw_steps_match_branch_scale(base):
    want_losses, want = train(base, branch_scale=C)
    got_losses, got = train(exact(base))
    assert torch.equal(got_losses, want_losses), (got_losses, want_losses)
    differ = [n for n in want if not torch.equal(got[n], want[n])]
    assert differ == [], differ
    # the parameters moved, so the steps are compared, not only the initialisation
    assert not torch.equal(got["layers.0.nffn.fc2.weight"], build(base).layers[0].nffn.fc2.weight.cuda().to(torch.bfloat16))
