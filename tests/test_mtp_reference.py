"""Multi-token prediction against a plain PyTorch reference, on one GPU.

A small Transformer with two MTP modules (plain attention, fp32, no FSDP) runs one training forward. Each module's
input is checked against one built by hand from the hidden state of the module before it (or of the main model's last
layer) after that one's final norm, and the embedding of the token k ahead. The token losses it returns (through the fused output layer) are compared with a
reference built from each module's block output: its own final norm, the shared output weight and torch's
cross-entropy on the token k+1 ahead. Then the gradients of both w.r.t. the MTP and shared weights and the main
model's weights. Packed sequences with an EOS check the masking. With --moe the model's layers after the first,
and so the MTP blocks, are MoE layers (their router aux loss is checked too); with --fused they run as fused blocks.

    torchrun --nproc_per_node 1 tests/test_mtp_reference.py [--moe] [--fused]
"""
import argparse
import dataclasses
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xllm.configuration import ConfStore
from xllm.distributed import initialize_model_parallel
from xllm.models.transformer import Transformer
from xllm.modules.mtp import IGNORE_INDEX, mtp_targets

EOS = 2
DEPTH = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe", action="store_true", help="MoE layers after the first, so MoE MTP blocks")
    parser.add_argument("--fused", action="store_true", help="fused blocks")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    initialize_model_parallel(1, 1, 60)
    torch.manual_seed(0)
    cfg = dataclasses.replace(
        ConfStore["k2-horizon-0.9B"], num_layers=2, model_dim=128, num_heads=4, num_kv_heads=2, head_dim=32,
        ffn_hidden_dim=256, vocab_size=512, chunk_size=64, mtp_depth=DEPTH, fused_block=False,
        causal_attn_backend=None, layerwise_ckpt=False, dropout=0.0, ddp_backend="fsdp2",
    )
    if args.moe:
        cfg = dataclasses.replace(
            cfg, num_dense_layers=1, num_experts=8, num_activated_experts=2, num_shared_experts=1, expert_inter_dim=64,
            # no router bias: its in-place update in backward would break the second backward over the graph below
            moe_router_score_func="sigmoid", moe_router_bias=False,
            moe_expert_backend="torch", moe_permutation_backend="torch",
        )
    if args.fused:
        # fused blocks need flash attention, so bf16
        cfg = dataclasses.replace(cfg, fused_block=True, causal_attn_backend="flash")
    dtype = torch.bfloat16 if args.fused else torch.float32
    # tolerances: (module input rtol, atol), (token loss rtol, atol), gradient relative difference
    tol = ((2e-2, 2e-2), (2e-2, 2e-2), 5e-2) if args.fused else ((1e-5, 1e-6), (1e-4, 1e-5), 1e-3)
    model = Transformer(cfg, SimpleNamespace(bos_id=1, eos_id=EOS, pad_id=0)).cuda().to(dtype).train()

    bsz, seq_len = 2, 64
    tokens = torch.randint(3, cfg.vocab_size, (bsz, seq_len + 1), device="cuda")
    tokens[0, 20] = EOS  # a document boundary in row 0
    tokens[1, 40] = EOS
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    captured = {}
    model.layers[-1].register_forward_hook(lambda module, inputs, output: captured.update(main=output[0]))
    model.embed.register_forward_hook(lambda module, inputs, output: captured.update(emb=output))
    for k, layer in enumerate(model.mtp.layers, start=1):
        layer.register_forward_hook(lambda module, inputs, output, k=k: captured.update({f"out{k}": output[0]}))
        layer.block.register_forward_pre_hook(lambda module, inputs, k=k: captured.update({f"in{k}": inputs[0]}))
    tok_loss, _, _ = model(tokens=x, targets=y, token_mask=None, multi_segments=True,
                           moe_router_load_balancing_type="dot" if args.moe else None)
    mtp_losses, mtp_masks, mtp_aux_loss = model.mtp.take_outputs()

    errors = []
    if args.moe and (mtp_aux_loss is None or not torch.isfinite(mtp_aux_loss)):
        errors.append(f"MoE MTP blocks: no finite aux loss ({mtp_aux_loss})")
    if not args.moe and mtp_aux_loss is not None:
        errors.append("dense MTP blocks returned an aux loss")
    refs = []
    hidden = captured["main"]
    for k, layer in enumerate(model.mtp.layers, start=1):
        # the module's input: [norm(previous hidden state after its final norm); norm(embedding of token j+k)],
        # projected; the previous final norm is the main model's for k = 1, module k-1's after
        nxt = torch.zeros_like(captured["emb"])
        nxt[:, :-k] = captured["emb"][:, k:]
        prev_norm = model.output.final_norm if k == 1 else model.output.mtp_norms[k - 2]
        ref_in = layer.proj(torch.cat([layer.hidden_norm(prev_norm(hidden)), layer.embed_norm(nxt)], dim=-1))
        if not torch.allclose(captured[f"in{k}"], ref_in, rtol=tol[0][0], atol=tol[0][1]):
            errors.append(f"module {k}: input differs from the previous final-normed hidden state and token j+{k}")
        hidden = captured[f"out{k}"]

        # reference loss: the module's final norm, shared output weight, torch cross-entropy on token j+1+k
        mtp_y, valid = mtp_targets(y, None, EOS, k)
        logits = model.output.output(model.output.mtp_norms[k - 1](hidden)).float()
        ref = F.cross_entropy(logits.view(-1, cfg.vocab_size), mtp_y.view(-1), ignore_index=IGNORE_INDEX,
                              reduction="none").view(bsz, seq_len)
        refs.append(ref)
        if not torch.equal(mtp_masks[k - 1], valid):
            errors.append(f"module {k}: mask differs from mtp_targets")
        # no target of module k across the EOS at token 20 (row 0) and 40 (row 1): positions 20-k .. 19
        if valid[0, 20 - k:20].any() or valid[1, 40 - k:40].any():
            errors.append(f"module {k}: a target across an EOS counts")
        if valid[:, seq_len - k:].any():
            errors.append(f"module {k}: a target outside the sequence counts")
        if not torch.allclose(mtp_losses[k - 1].float(), ref, rtol=tol[1][0], atol=tol[1][1]):
            errors.append(f"module {k}: token losses differ: max |diff| "
                          f"{(mtp_losses[k - 1].float() - ref).abs().max().item():.3e}")
        if (mtp_losses[k - 1][~valid] != 0).any():
            errors.append(f"module {k}: masked positions have a loss")
        print(f"module {k}: MTP token loss mean over valid {mtp_losses[k - 1][valid].mean().item():.4f}, "
              f"valid targets {int(valid.sum())} of {valid.numel()}")

    params = {"output.weight (shared)": model.output.output.weight, "embed (shared)": model.embed.weight,
              "final norm (main)": model.output.final_norm.weight,
              "layer 0 (main model)": model.layers[0].attention.wq.weight}
    for k, layer in enumerate(model.mtp.layers, start=1):
        params.update({f"mtp {k} proj": layer.proj.weight, f"mtp {k} hidden_norm": layer.hidden_norm.weight,
                       f"mtp {k} embed_norm": layer.embed_norm.weight,
                       f"mtp {k} final norm": model.output.mtp_norms[k - 1].weight})
        if args.moe:
            params.update({f"mtp {k} experts": layer.block.moe.experts.weight1,
                           f"mtp {k} router": layer.block.moe.router.weight})
    names, tensors = list(params), list(params.values())
    g_fused = torch.autograd.grad(sum(loss.sum() for loss in mtp_losses), tensors, retain_graph=True)
    g_ref = torch.autograd.grad(sum(ref.sum() for ref in refs), tensors)
    for name, a, b in zip(names, g_fused, g_ref):
        rel = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
        print(f"grad {name:24s} |ref| {b.norm().item():.3e}  rel. diff {rel:.2e}")
        if not rel < tol[2]:
            errors.append(f"gradient of {name} differs (rel. {rel:.2e})")
        if b.norm().item() == 0:
            errors.append(f"gradient of {name} is zero")

    print(f"moe={args.moe} fused={args.fused}: main token loss mean {tok_loss.mean().item():.4f}")
    print("\n".join(errors) if errors else "MTP matches the reference (PASS)")
    dist.destroy_process_group()
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
