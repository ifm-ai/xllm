"""Multi-token prediction under tensor parallelism: gradients against a plain PyTorch reference on full weights.

The MTP modules (xllm/modules/mtp.py) run with the real norms, TP mappings and fused output layer, at TP = the number
of processes and one norm group (the norms gather their input), with a stand-in block that acts on each rank's slice
of the model dim (so it adds no TP handling of its own). The reference recomputes the same model on every rank from
the full weights, without TP. Each loss (the main one and each module's) is checked on its own: its token losses and
its gradients w.r.t. the trunk's last hidden state, the embeddings and every weight it reaches. A gradient summed
over the TP ranks instead of split comes out TP times (or TP^k times) too large. --checkpoint runs the modules under
activation checkpointing, so that their norms are recomputed in backward.

    torchrun --nproc_per_node 2 tests/test_mtp_tp.py [--layernorm] [--checkpoint]
"""
import argparse
import dataclasses
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from xllm.configuration import ConfStore
from xllm.distributed import get_model_parallel_rank, get_model_parallel_world_size, initialize_model_parallel
from xllm.models.transformer import TransformerOutputLayer
from xllm.modules.mtp import IGNORE_INDEX, MultiTokenPrediction, mtp_targets

EOS = 2
DEPTH = 2


class SliceBlock(nn.Module):
    """A stand-in for a Transformer block, elementwise on the rank's slice of the model dim."""

    def __init__(self, dim: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(dim))

    def forward(self, x, *args):
        return x * (1 + self.a) + torch.tanh(x), None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layernorm", action="store_true", help="LayerNorm (with a bias) instead of RMSNorm")
    parser.add_argument("--checkpoint", action="store_true", help="recompute the modules in backward")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    initialize_model_parallel(dist.get_world_size(), 1, 60)
    tp, rank = get_model_parallel_world_size(), get_model_parallel_rank()
    cfg = dataclasses.replace(
        ConfStore["k2-horizon-0.9B"], num_layers=2, model_dim=128, vocab_size=512, mtp_depth=DEPTH,
        layernorm_num_groups=1, apply_rmsnorm=not args.layernorm, norm_affine=True, recompute_logits=False,
    )
    dim, vocab = cfg.model_dim, cfg.vocab_size
    eps = cfg.layernorm_eps if args.layernorm else cfg.rmsnorm_eps
    part = dim // tp
    output = TransformerOutputLayer(cfg).cuda().train()
    mtp = MultiTokenPrediction(cfg, lambda c, i: SliceBlock(part), lambda m: m).cuda().train()
    assert output.final_norm.gather_input == (tp > 1)

    # the same full weights at every TP size, sliced per rank (all sharded on dim 0)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def full(*shape, scale):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    weights = {}  # name -> (full weight, this rank's parameter)
    shards = {"output": (output.output.weight, vocab, 0.1), "final_norm": (output.final_norm.weight, dim, 0.3)}
    for k, layer in enumerate(mtp.layers, start=1):
        shards.update({
            f"mtp_norm{k}": (output.mtp_norms[k - 1].weight, dim, 0.3),
            f"hidden_norm{k}": (layer.hidden_norm.weight, dim, 0.3),
            f"embed_norm{k}": (layer.embed_norm.weight, dim, 0.3),
            f"proj{k}": (layer.proj.weight, dim, 0.1),
            f"block{k}": (layer.block.a, dim, 0.3),
        })
    if args.layernorm:
        norms = {"final_norm": output.final_norm}
        for k, layer in enumerate(mtp.layers, start=1):
            norms.update({f"mtp_norm{k}": output.mtp_norms[k - 1], f"hidden_norm{k}": layer.hidden_norm,
                          f"embed_norm{k}": layer.embed_norm})
        shards.update({f"{name}.bias": (norm.bias, dim, 0.3) for name, norm in norms.items()})
    with torch.no_grad():
        for name, (param, rows, scale) in shards.items():
            w = full(rows, *param.shape[1:], scale=scale)
            n = rows // tp
            param.copy_(w[rank * n:(rank + 1) * n])
            weights[name] = (w.requires_grad_(True), param)

    bsz, seq_len = 2, 32
    x_full = full(bsz, seq_len, dim, scale=1.0)
    emb_full = full(bsz, seq_len, dim, scale=1.0)
    tokens = torch.randint(3, vocab, (bsz, seq_len), generator=gen, device="cuda")
    tokens[0, 10] = EOS  # a document boundary
    dslice = slice(rank * part, (rank + 1) * part)

    def reference(x, emb):
        W = {name: w for name, (w, _) in weights.items()}

        def norm(h, name):
            if args.layernorm:
                return F.layer_norm(h, (dim,), eps=eps) * (W[name] + 1) + W[f"{name}.bias"]
            return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (W[name] + 1)

        def token_loss(h, norm_name, y):
            logits = norm(h, norm_name) @ W["output"].t()
            return F.cross_entropy(logits.view(-1, vocab), y.view(-1), ignore_index=IGNORE_INDEX,
                                   reduction="none").view(bsz, seq_len)

        losses = [token_loss(x, "final_norm", tokens)]
        hidden, prev_norm = x, "final_norm"
        for k in range(1, DEPTH + 1):
            nxt = torch.zeros_like(emb)
            nxt[:, :-k] = emb[:, k:]
            normed = norm(hidden, prev_norm)
            h = torch.cat([norm(normed, f"hidden_norm{k}"), norm(nxt, f"embed_norm{k}")], -1)
            h = h @ W[f"proj{k}"].t()
            hidden = h * (1 + W[f"block{k}"]) + torch.tanh(h)
            y, _ = mtp_targets(tokens, None, EOS, k)
            losses.append(token_loss(hidden, f"mtp_norm{k}", y))
            prev_norm = f"mtp_norm{k}"
        return losses

    def gathered(t, dim=0):
        parts = [torch.empty_like(t) for _ in range(tp)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim=dim)

    errors = []
    for term, label in enumerate(["main"] + [f"mtp {k}" for k in range(1, DEPTH + 1)]):
        for w, param in weights.values():
            w.grad, param.grad = None, None
        x = x_full[..., dslice].clone().requires_grad_(True)
        emb = emb_full[..., dslice].clone().requires_grad_(True)
        if args.checkpoint:
            main_loss = checkpoint(mtp, output, x, emb, tokens, None, EOS, (), False, use_reentrant=False)
        else:
            main_loss = mtp(output, x, emb, tokens, None, EOS, (), False)
        mtp_losses, _, _ = mtp.take_outputs()
        tok_loss = ([main_loss] + mtp_losses)[term]
        tok_loss.float().sum().backward()

        x_ref, emb_ref = x_full.clone().requires_grad_(True), emb_full.clone().requires_grad_(True)
        ref_loss = reference(x_ref, emb_ref)[term]
        ref_loss.sum().backward()

        if not torch.allclose(tok_loss.float(), ref_loss, rtol=1e-4, atol=1e-5):
            errors.append(f"{label}: token losses differ (max |diff| {(tok_loss.float() - ref_loss).abs().max():.3e})")
        grads = {"trunk hidden": (gathered(x.grad, -1), x_ref.grad)}
        if emb_ref.grad is not None:
            grads["embeddings"] = (gathered(emb.grad, -1), emb_ref.grad)
        grads.update({name: (gathered(param.grad), w.grad) for name, (w, param) in weights.items()
                      if w.grad is not None and w.grad.norm() > 0})
        for name, (got, want) in grads.items():
            rel = ((got - want).norm() / want.norm()).item()
            ratio = (got.norm() / want.norm()).item()
            if rank == 0:
                print(f"tp={tp} {label:6s} grad {name:13s} |ref| {want.norm().item():.3e} ratio {ratio:.4f} "
                      f"rel. diff {rel:.2e}")
            if not rel < 1e-3:
                errors.append(f"{label}: gradient of {name} differs (rel. {rel:.2e}, |got|/|ref| {ratio:.3f})")

    if rank == 0:
        kind = ("layernorm" if args.layernorm else "rmsnorm") + (", checkpointed" if args.checkpoint else "")
        print("\n".join(errors) if errors else f"tp={tp} {kind}: MTP gradients match the reference (PASS)")
    dist.destroy_process_group()
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
