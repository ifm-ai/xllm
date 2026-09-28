The following performance benchmarks were done by the `xLLM` tem in Sept. 2026, with torch=2.11.0+cu12.8 and flash-attn-3.

## Experimental Setup

K2-Horizon models, including 0.9B, 3.7B, 7B and MoVA-36B-A4B. 
Sequence length is set to 8192. For the 0.9B model, we use 32 H200 GPUs (8 GPUs/node x 4 nodes).  
Other models are trained in 128 H200 GPUs (8 GPUs/node x 16 nodes). 

## Results

For xLLM with fused block mode, we attempted to maximize efficiency by tuning recomputation options to improve memory usage.

| Model | BSZ | Parallelism | TPS/GPU | Throughput | MFU (%) | Memory (GiB) |
|---|:---:|---|--------:|:----------:|:-------:|:------------:|
| 0.9B | 32  | FSDP 32, TP 1, CP 1 |  48,450 |    421     |  42.5%  |    118.1     |
| 3.7B |  8  | FSDP 128, TP 1, CP 1 |  13,760 |    464     |  46.9%  |    135.9     |
| 7B |  8  | FSDP 128, TP 1, CP 1 |   8,660 |    477     |  48.2%  |    113.0     |
| MoVA-36B-A4B | 16  | FSDP 64, TP 2, CP 1 |   6,346 |    264     |  26.7%  |    108.5     |

