"""Complete(d)P with Muon on FSDP2: the folded run against a residual-multiplier run, on the real Transformer.

    torchrun --nproc_per_node=8 tests/test_completedp_muon_gpu.py

Two models from one seed, built and stepped like train.py (build_model, build_optimizer with mup, clip_grad_norm_):
- folded: mup's init (wo / fc2 x c), Muon update scales c on wo / fc2, clipping norm with clip_grad_scales;
- multiplier: the plain init, forward hooks that multiply the wo / fc2 outputs by c, no update or gradient scales.
c = 1/4 is a power of two, so the init, the first loss and the first gradient norm agree exactly up to reduction
order. Later steps may differ by bf16 Newton-Schulz noise: it turns a 1e-7 difference in the gradients (another
reduction order) into ~1e-2 in an update, so they are held to tests/test_muon.py's bf16 tolerance, far below the
negative control. The CPU tests (tests/test_completedp_muon.py) check the fold to 1e-6 in float64.
"""
import copy
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from xllm.config import ModelConf, MuPConf, OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.models import build_model
from xllm.optim import build_optimizer
from xllm.optim.mup import clip_grad_scales, out_projections
from xllm.utils import clip_grad_norm_

STEPS = 10
VOCAB, SEQ, BATCH = 512, 128, 2
TOL = 3e-2  # bf16 Newton-Schulz, as in tests/test_muon.py


def build(model_cfg, optim_cfg, mup):
    torch.manual_seed(0)
    model = build_model(model_cfg, dtype="fp32", fully_sharded_size=None, fp32_reduce_scatter=True,
                        reshard_after_forward=True, forward_prefetch=False,
                        tokenizer=SimpleNamespace(bos_id=1, eos_id=2, pad_id=0))
    optimizer, _ = build_optimizer(model, optim_cfg, STEPS, "fp32", mup=mup)
    return model, optimizer


def train(model, optimizer, grad_scales):
    torch.manual_seed(1 + dist.get_rank())
    losses, norms = [], []
    for _ in range(STEPS):
        tokens = torch.randint(0, VOCAB, (BATCH, SEQ + 1), device="cuda")
        tok_loss, _, _ = model(tokens=tokens[:, :-1], targets=tokens[:, 1:], multi_segments=False)
        loss = tok_loss.mean()
        loss.backward()
        norms.append(clip_grad_norm_(fsdp_module=model, max_norm=1e9, grad_scales=grad_scales))
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach())
    return torch.stack(losses).tolist(), norms


def fold_error(multiplier, folded, c):
    """Max relative difference between the folded weights and the multiplier run's (wo / fc2 x c)."""
    out = {id(p) for p in out_projections(folded)}
    err = 0.0
    for a, b in zip(multiplier.parameters(), folded.parameters()):
        a, b = a.full_tensor(), b.full_tensor()
        want = a * c if id(b) in out else a
        err = max(err, ((b - want).abs().max() / want.abs().max()).item())
    return err


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    initialize_model_parallel(1, 1, 600)
    rank0 = dist.get_rank() == 0

    model_cfg = ModelConf(model_dim=256, num_layers=8, num_heads=4, ffn_hidden_dim=512, vocab_size=VOCAB,
                          chunk_size=64, apply_rmsnorm=True, qknorm=True, swiglu=True, init_mode="gaussian")
    optim_cfg = OptimConf(optimizer="muon", lr=3e-3, muon_lr=0.02, weight_decay=0.1, warmup=0, scheduler="constant")
    mup = MuPConf(base_model_dim=128, base_num_layers=2, base_init_hidden_std=0.02, base_init_embed_std=0.02,
                  base_init_unembed_std=0.02, alpha=1.0)
    mup.resolve(model_cfg, optim_cfg)
    c = mup.branch_scale
    assert c == 0.25
    plain_cfg = copy.deepcopy(model_cfg)
    plain_cfg.init_out_std = None

    folded, opt_folded = build(model_cfg, optim_cfg, mup)
    multiplier, opt_multiplier = build(plain_cfg, optim_cfg, mup)
    opt_multiplier.update_scales = [1.0] * len(opt_multiplier.update_scales)  # c sits in the forward instead
    for layer in multiplier.layers:
        for module in (layer.attention.wo, layer.nffn.fc2):
            module.register_forward_hook(lambda module, args, output: output * c)
    assert sorted(set(opt_folded.update_scales)) == [0.25, 1.0]
    init_error = fold_error(multiplier, folded, c)

    losses_m, norms_m = train(multiplier, opt_multiplier, None)
    losses_f, norms_f = train(folded, opt_folded, clip_grad_scales(folded, mup))
    loss_error = max(abs(a - b) / abs(a) for a, b in zip(losses_m, losses_f))
    norm_error = max(abs(a - b) / a for a, b in zip(norms_m, norms_f))
    first_error = max(abs(losses_m[0] - losses_f[0]) / abs(losses_m[0]), abs(norms_m[0] - norms_f[0]) / norms_m[0])
    error = fold_error(multiplier, folded, c)

    # negative control: the folded model without its update scales drifts away
    unscaled, opt_unscaled = build(model_cfg, optim_cfg, mup)
    opt_unscaled.update_scales = [1.0] * len(opt_unscaled.update_scales)
    train(unscaled, opt_unscaled, clip_grad_scales(unscaled, mup))
    control = fold_error(multiplier, unscaled, c)

    if rank0:
        print(f"losses {losses_m[0]:.4f} -> {losses_m[-1]:.4f}; init error {init_error:.1e}, first loss / grad norm "
              f"error {first_error:.1e}, loss error {loss_error:.1e}, grad norm error {norm_error:.1e}, weight error "
              f"{error:.1e}; without update scales {control:.1e}")
    assert init_error == 0.0, "wo / fc2 init is not c x the plain init"
    assert first_error < 1e-5, "the folded forward or its clipping norm differs from the multiplier run's"
    assert loss_error < TOL and norm_error < TOL and error < TOL, "the folded run differs from the multiplier run"
    assert control > 10 * TOL, "the test cannot tell a missing update scale"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
