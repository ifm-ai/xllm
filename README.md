<div align="center">

<h1>xLLM</h1>

<p><strong>Efficient LLM training for extra-long contexts.</strong></p>

<p>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>
</p>

<p>
  <a href="#highlights">Highlights</a> &nbsp;|&nbsp;
  <a href="#installation">Installation</a> &nbsp;|&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;|&nbsp;
  <a href="#documentation">Documentation</a> &nbsp;|&nbsp;
  <a href="https://github.com/ifm-ai/xllm/issues">Issues</a>
</p>

</div>

xLLM is a PyTorch-based framework for long-context language modeling, with
distributed training, online data preparation, evaluation, and model export
in one repository.

## Highlights

| Area                       | Capabilities                                                                                                                                                                                                                          |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Model architectures**    | Transformer architectures, with dense, MoE, and MoVA components.                                                                                                                                                                      |
| **Distributed training**   | FSDP1/FSDP2, data parallelism, model parallelism, and context parallelism.                                                                                                                                                            |
| **Efficient computation**  | Custom CUDA kernels, fused blocks, recomputation, and FlashAttention backends. See the H200 benchmarks for [K2 Horizon](examples/benchmark_k2-horizon_h200_20260925.md) and [Llama 3 8B](examples/benchmark_llama3_h200_20260925.md). |
| **Online data pipeline**   | Parallel tokenization, asynchronous preparation, buffered shuffle, and [bestfit packing](xllm/data/README.md#bestfit-packing).                                                                                                        |
| **Training to deployment** | Checkpoint/resume, perplexity and task evaluation, Hugging Face export, and vLLM integration (via [xBridges](https://github.com/ifm-ai/xbridges)).                                                                                                                      |

## Installation

Start with **PyTorch >= 2.11** and **CUDA >= 12.8**. Follow the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for your
environment before building xLLM.

> Building the native extensions requires a CUDA development environment.
> PyTorch must be installed before installing xLLM: `requirements.txt` does not
> include `torch`, and `setup.py` imports it to build the extensions (hence
> `--no-build-isolation`).

### 1. Install xLLM

```bash
git clone https://github.com/ifm-ai/xllm.git
cd xllm

python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation --config-settings editable_mode=compat
```

### 2. Set Up Attention Backends

Install FlashAttention for the Quick Start below:

```bash
python -m pip install flash-attn --no-build-isolation
```

<details>
<summary><strong>FlashAttention 3 or 4</strong></summary>

Follow the upstream installation instructions for
[FlashAttention 3](https://github.com/Dao-AILab/flash-attention/tree/main?tab=readme-ov-file#flashattention-3-beta-release)
or [FlashAttention 4](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
Then select the installed backend before launching training:

```bash
# FlashAttention 3
export ENABLE_FLASH_ATTENTION_3=TRUE

# Alternatively, use FlashAttention 4 instead:
# export ENABLE_FLASH_ATTENTION_4=TRUE
```

</details>

<details>
<summary><strong>xattn: efficient attention modules</strong></summary>

Install [xattn](https://github.com/ifm-ai/xattn) from source.

```bash
git clone https://github.com/ifm-ai/xattn.git
cd xattn

git submodule update --init --recursive
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation .
```

</details>

<details>
<summary><strong>Flash Linear Attention: modules and operators</strong></summary>

Install [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
for models using its modules or operators:

```bash
python -m pip install flash-linear-attention
```

</details>

### 3. Add Tokenizer Support

For Hugging Face tokenizers (required for the Quick Start below):

```bash
python -m pip install "transformers[torch]"
```

## Quick Start

Start a short, single-GPU training run with the K2 Horizon 0.9B configuration.
Replace the paths below with a directory containing `*.chunk*.jsonl` text data
and a Hugging Face tokenizer with BOS/EOS tokens. See the
[data format examples](xllm/data/README.md#data-format).

```bash
DATA_DIR=/path/to/text-data
TOKENIZER=/path/to/tokenizer

torchrun --standalone --nproc_per_node=1 train.py \
  --model k2-horizon-0.9B \
  --model.causal_attn_backend flash --model.chunk_size 1024 \
  --data "${DATA_DIR}:1.0:text:text" \
  --tokenizer.type huggingface --tokenizer.path "$TOKENIZER" \
  --dataloader.packing_type bestfit --dataloader.buffer_size 512 \
  --batch_size 1 --seq_len 2048 --dtype bf16 \
  --steps 100 --optim.warmup 10 \
  --log_freq 10 --dump_freq 100 --eval_freq -1 \
  --keep_eval_checkpoints false \
  --dump_dir saved_models/quickstart
```

`batch_size` is per data-parallel rank. Alternatively, set `global_batch_size`,
divisible by the data-parallel size; do not set both. Use a fresh `dump_dir` for
a new run; an existing checkpoint in that directory is resumed automatically.

When changing these values, keep the constraints checked in
[`xllm/config.py`](xllm/config.py):

- `seq_len` must be divisible by `model.chunk_size` (times
  `context_parallel_size` when using context parallelism). The default
  `chunk_size` is 2048; the Quick Start uses 1024 so that `seq_len` 2048 splits
  into two chunks.
- `dump_freq` must be divisible by `log_freq`, and a positive `eval_freq` must be
  divisible by `log_freq`.
- With `keep_eval_checkpoints true` (the default), a positive `eval_freq` must
  also be divisible by `dump_freq`.

## Checkpoint Conversion & Serving
Conversion and serving checkpoints of xLLM is supported by [xBridges](https://github.com/ifm-ai/xbridges).

## Documentation

| I want to... | Start here |
| --- | --- |
| **Prepare training data** | [Data loader](xllm/data/README.md): JSONL formats, online tokenization, packing, and resume. |
| **Configure and launch training** | [Experiment scripts](examples/) and [configuration](xllm/config.py); entry point: [train.py](train.py). |
| **Evaluate a model** | [Evaluation guide](xllm/eval/README.md); entry point: [eval.py](eval.py). |
| **Export to Hugging Face** | [Checkpoint conversion](https://github.com/ifm-ai/xbridges/blob/main/xbridges/huggingface/README.md). |
| **Serve with vLLM** | [vLLM integration](https://github.com/ifm-ai/xbridges/blob/main/xbridges/vllm/README.md). |

## Repository Layout

```text
xllm/
  configuration/  Config dataclass and command-line parsing
  csrc/           Custom C++/CUDA kernels
  data/           Online data loading, tokenization, packing, and resume
  models/         Model architectures and fused blocks
  modules/        Attention, expert layers, normalization, and operators
  distributed/    Parallelism and distributed training utilities
  optim/          Optimizers and learning-rate schedulers
  eval/           Evaluation tasks and runners
examples/         Training launch scripts and benchmarks
tests/            Tests and validation utilities
```

## Contributing

Bug reports and focused pull requests are welcome. For runtime issues, include
the relevant configuration, package versions, hardware, and a minimal reproducer.
Include tests for behavior changes.

## License

xLLM is released under the [Apache 2.0 License](LICENSE).
