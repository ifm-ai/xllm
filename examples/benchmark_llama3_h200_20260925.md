The following performance benchmarks were done by the `xLLM` tem in Sept. 2026, with torch=2.11.0+cu12.8 and flash-attn-3. 
The number of TorchTitan are from the [benchmark docs](https://github.com/pytorch/torchtitan/blob/main/benchmarks/)

## Experimental Setup

Llama3-8B, sequence length 8,192, local batch size 4, dtype=bfloat16. 
All jobs are running 128 H200 GPUs (8 GPUs/node x 16 nodes) with 1D Parallelism (FSDP2).

## Results

| Configuration | TPS/GPU | Throughput | MFU (%) | Memory (GiB) |
|---|--------:|-----------:|--------:|-------------:|
| TorchTitan + `torch.compile` |   6,514 |        329 |   33.2% |         62.0 |
| xLLM (eager) |   8,330 |        429 |   43.3% |         26.0 |
| xLLM (fused block) |   8,495 |        437 |   44.1% |         35.5 |
| xLLM (fused block) + recompute |  10,050 |        516 |   52.1% |        109.1 |

For xLLM with fused block mode, we attempted to maximize efficiency by tuning recomputation options to improve memory usage.
