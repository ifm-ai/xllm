# Draft 400B hybrid Gekko MoE: three variants for feedback

This is a spec for discussion, not code. Nothing here is imported by xLLM.

- **Not runnable on main.** All three variants assume that Gekko layers can carry MoE (a Gekko + MoE block). Main's
  `HybridGekko` only allows MoE on its full-attention layers.
- **Every flag is a real `ModelConf` field.** Each variant passes `ModelConf(...)` on a local integration branch:
  main, plus PR #33, plus PR #41's hyper-connection commits. That branch is not pushed.
- **On main alone the check fails,** because `ihc`, `dynamic_qb` and `moe_latent_alpha` come from those PRs.

## The three variants

Parameter counts leave out embeddings, as the name `k2-horizon-375B-a23B` does.

| | A: minimum | B: A + LatentMoE | C: deeper |
|---|---|---|---|
| Layers (full attention) | 64 (8) | 64 (8) | 80 (10) |
| Width d, heads × 128 | 6144, 48 Q / 8 KV | 6144, 48 Q / 8 KV | 5120, 40 Q / 8 KV |
| Experts, top-k, width | 288, 8, 1216, plus 1 shared | 1152, 32, 1216 in a d/4 = 1536 latent space | 288, 8, 1152, plus 1 shared |
| Total / active | 404B / 21.4B | 406B / 22.9B | 402B / 20.2B |
| Idea | The features Max's team has ablated, on a body like the 375B | Alex's list, which adds LatentMoE to A | Depth instead of width, to leave room for looping later |

- **B's extra active parameters:** each layer's latent projections (2 × d × d/4) and its larger router add 1.5B.
- **Settings shared by all three:**
  - The attention setup follows the `hybrid-gekko-*` presets in `xllm/models/hybrid.py`:
    - chunk 4096;
    - head dim 128, with QK norm, a softplus output gate, convolutions on the attention projections;
    - RoPE 500k on the Gekko layers and no RoPE on the full-attention layers.
  - The MoE recipe follows `k2-horizon-375B-a23B`: 3 dense layers first, 1 shared expert, and an FFN of about 2.7 × d.
  - Full attention sits on every 8th layer, ending on the last layer, as in the presets.

## Feature status

| Feature | In the drafts | Source / status |
|---|---|---|
| Hybrid Gekko | Yes | Main backbone; dense-only today |
| MoE in Gekko layers | Assumed | Not in main ("not yet", per Alex) |
| iHC (`residual_func ihc`) | Yes | PR #41, closed unmerged 21 Nov but still called "pending" by Max; ablated by Max's team |
| QB-MoE (`moe_router_type dynamic_qb`) | Yes | PR #33 (open) |
| LatentMoE (`moe_latent_alpha`) | B only | PR #33; on Alex's list |
| Looping | No flags | Not in xLLM; "possibly not from scratch", so it could be added to C later |
| Causal encoder, Engrams, extra sparse attention, layer or expert pruning | Out | Still being explored or debated |

## Ablation axes

Each row is a change from A, so any ablation is A plus one row. These cover the options raised in the architecture
threads.

| Axis | Options | Flags, as a change from A | Raised by |
|---|---|---|---|
| Residual | base / ihc / ohc / mhc | `--model.residual_func "base"` (or `"ohc"`, `"mhc"`) | Junlin: hyper-connections over attention residuals. Chaitanya: check deep (100+ effective layers), low-TPP models |
| Router | QB / top-k with bias (the 375B recipe) | `--model.moe_router_type "topk" --model.moe_router_bias "true" --model.moe_router_bias_update_rate 0.001 --model.moe_router_scaling_factor 2.5` | Alex: QB ("Kimi-style routing") |
| LatentMoE | off / α = 4 | `--model.moe_latent_alpha 4` (that is, B) | Alex's list |
| Full-attention ratio | 1 in 8 / 1 in 4 / about 1 in 10 | `--model.full_attn_layer_list "3,7,11,...,63"` or `"3,13,23,33,43,53,63"` | Hector: don't use full attention everywhere |
| Depth vs width | 64 × 6144 / 80 × 5120 | C's shape flags | Chaitanya: a deeper 400B with looping |
| RoPE on full-attention layers | none / RoPE | `--model.apply_nope_in_causal_attn "false"` | Xuezhe suggested none, as the presets do |

**Options with no xLLM flag yet:**
- looping, with input injection and hyper-connections across loops;
- attention residuals;
- DeepSeek's causal encoder;
- Engrams;
- removing layers or experts during pretraining.

**At what scale to run these?** Max validates new features above 500 tokens per parameter, counting active
parameters. Chaitanya asked about the deep, low-TPP regime that the 400B's looping would create.

## Open questions for Max / Junlin / Lynn

1. **Full-attention ratio.** The drafts use 1 layer in 8; the presets use about 1 in 9–10.
2. **QB settings.**
   - The scaling factor is left unset: the 375B used 2.5 with top-k sigmoid routing, but with QB's global normalization it means something else.
   - Sigmoid or softmax scores?
3. **LatentMoE.** Should α be 4, with 32 active experts per token under QB? Is the post-RMSNorm wanted?
4. **Attention width.** Heads × 128 = d here, as in the 375B. The hybrid presets use 1.4–1.6 × d, which would add about 3B active.
5. **Hyper-connections.**
   - ihc or ohc?
   - Is a fused iHC + MoE block planned? Without one, the model trains unfused.
6. **Norm groups at this width.**
   - `layernorm_num_groups` is 1 as in the 375B; the presets use 2–4.
   - `timenorm_num_groups` follows the presets' d/64 rule.
7. **Who designs the Gekko + MoE block,** and will its flags differ from these?

## Files

- **`drafts.py`:** the three presets (`ModelConf` keyword arguments).
  - `python3 drafts.py` prints the flags.
  - `--validate` checks them against `ModelConf`. It needs a branch with PR #33 and PR #41 applied, on
    `PYTHONPATH`, and a torch environment.
- **`flags.txt`:** the printed flag blocks.
- **`params.py`:** parameter counts that follow xLLM's per-block `num_parameters` formulas, including LatentMoE and iHC. It gives the `k2-horizon-375B-a23B` preset 376.1B / 23.6B, which matches the preset's name.
