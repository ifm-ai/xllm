# Multi-token prediction

Multi-token prediction (MTP), as in DeepSeek-V3, trains the model to also predict the tokens after next. With depth D,
at every position j, D extra modules predict the tokens j+2, ..., j+1+D, with the shared output weight. Module k takes
the hidden state of the module before it after that module's final norm (for k = 1, the main model's last hidden
state after its final norm) and the embedding of the real token j+k. The main model is unchanged; the modules are used
in training only.

## Usage

```
--model.mtp_depth 1        # D, the number of extra tokens; 0 (default) = off
--mtp_loss_weight 0.3      # λ in: loss = next-token loss + λ / D × (sum of the D MTP losses) (DeepSeek-V3: D = 1, λ = 0.3, then 0.1)
```

The losses are logged: `avg_loss` is the next-token loss, as without MTP, and `avg_mtp_loss` the mean of the MTP
losses; with D > 1 also `avg_mtp_loss_<k>` for each module. Compare runs on `avg_loss`.

## How it works

Code: [xllm/modules/mtp.py](../xllm/modules/mtp.py). For each module k = 1, ..., D:

1. Take the previous hidden state after its final norm (for k = 1, the main model's last hidden state after the
   main final norm; after, module k-1's output after module k-1's final norm), and normalize it and the embedding of
   token j+k. This is the convention of DeepSeek-V3's MTP weights in vLLM and SGLang, which give the MTP layers the
   final-normed hidden states. The modules run inside the output layer's forward, where these norms are gathered.
2. Concatenate them (2 × model_dim) and project back to model_dim: h′. This happens at every position.
3. Run one Transformer block over the sequence of h′. The block is built like the main model's last layer (dense,
   MoE or MoVA, fused or not), so all parallelism (TP, CP, FSDP, expert parallelism) applies to it as to any layer.
   Its output, after the module's final norm, is the next module's input.
4. The output layer applies the module's final norm and the shared output weight, and computes the cross-entropy on
   token j+1+k (one more call of its fused loss per module).

New weights per module: the two input norms, the projection (2 × model_dim × model_dim), the block and a final
norm. The embedding table and the output weight are shared. Each module is its own FSDP unit.

A position j gets no target of module k when token j+1+k is outside the sequence, when one of the tokens j+1, ...,
j+k is an end-of-document token (token j+1+k is then in another document), or when token j+1+k is masked or ignored
as a main target. Each MTP loss is normalized like the main loss, over its own target count. The routers of MoE or
MoVA modules add their aux loss with the weight of one main layer's.

## Cost

On k2-horizon-moe-36B-a4B (48 layers, 2 nodes, TP 2, 4 × 8,192 tokens per GPU, team data and tokenizer, steps
101–200 from scratch, 2 runs each on the same nodes), D = 1: step time 3,745 → 4,057 ms (+8.3%), peak memory 55.2 →
55.4 GiB. Most of it is the extra pass of the output layer: with a 250k-token vocabulary and model_dim 2,560 the
output weight is a large share of the per-token compute. Each further module should add about as much (not measured).

## Limits

- `arch=transformer` only (not Gekko).
- Under context parallelism the last k positions of each rank's slice get no target of module k.
- Not compatible with `health_logits_stats` (the MTP rows would be counted in the logit statistics).
- A checkpoint saved without MTP, or with another depth, has other weights: an MTP run starts from scratch.
- The MTP weights (`mtp.*`, `output.mtp_norms.*`) are in the checkpoints. The xBridges Hugging Face export does not
  convert them yet and refuses a checkpoint with unconverted weights.
- The logged parameter count includes the MTP modules.
