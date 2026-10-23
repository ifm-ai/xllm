---
date: 2026-10-16
---

# Attention correctness fixes

Based on `release_v1.01` at `cd40545`. No hybrid, optimizer or training-recipe changes.

| Fix | Affected settings |
| --- | --- |
| Preserve inputs during partial RoPE | Fused attention with `0 < rope_head_dim < head_dim`; normalization backward can otherwise read rotated values. |
| Trim packed metadata on the sequence axis; allow segment IDs without a BOS mask | Regular full-attention xattn and shared fused xattn forward/backward helpers. |
| Keep normalized convolution weights/gradients FP32 through softmax backward | Gekko with normalized Triton short convolutions; BF16 rounding amplified small kernel-gradient errors, especially with long-context CP. The fast kernels remain enabled. |

The last issue appeared in our [32K CP4 qualification](https://wandb.ai/mbzuai-llm/xllm-release-v1.01-muon/runs/9z1d7xgm);
the correction passed [the follow-up](https://wandb.ai/mbzuai-llm/xllm-release-v1.01-muon/runs/5bpgfqwd).
These identify observed qualification failures, not an audit of production checkpoints.
The release already contains the earlier fused backward/backend repairs.
The separate memory-efficient RMSNorm divide-by-zero issue is not fixed here.

Focused CUDA regressions:

```bash
python -m pytest -q tests/attn/test_attention_regressions.py
```

[All six cases pass](https://wandb.ai/mbzuai-llm/xllm-release-v1.01-muon/runs/r4j7jj4w);
five fail on the [unmodified release](https://wandb.ai/mbzuai-llm/xllm-release-v1.01-muon/runs/8np8ec93).
[74 selected existing convolution checks](https://wandb.ai/mbzuai-llm/xllm-release-v1.01-muon/runs/srla7y0p) also pass.
