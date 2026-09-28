# XLLM Data Loader

## Features

- **Online tokenization:** Read raw JSONL, apply templates, and tokenize during
  training. No offline tokenized dataset is required.
- **Parallel loading:** Data-parallel ranks process their assigned shards.
  Within each rank, `num_workers` threads read and tokenize different sources
  concurrently; these workers are threads, not subprocesses.
- **Asynchronous preparation:** Double buffering lets the background pipeline
  read, tokenize, and pack the next buffer while the model trains on the current
  one, overlapping CPU data preparation with GPU computation.
- **Buffered shuffle and mixing:** Mix weighted sources and shuffle documents
  before packing, then shuffle the packed sequences. A larger `buffer_size`
  provides a wider shuffle window, at the cost of more CPU memory and refill work.
- **Text and chat packing:** Support pretraining and chat/SFT with assistant-token
  masks, using `simple` concatenation or padding-efficient `bestfit` packing.
- **Resumable data order:** Restore the same subsequent batches from checkpoints
  when the data and loader configuration are unchanged.

## Data Format

Each training source uses `directory:weight:json_key:source_format`.
Separate sources with commas; positive weights are normalized automatically.

```bash
DATA="/data/web:0.7:text:text,/data/chat:0.3:conversation:chat_assistant"
```

The directory must contain uncompressed chunk files such as
`web.chunk0.jsonl` and `web.chunk1.jsonl`. Training accepts directories,
not direct file paths; files named simply `train.jsonl` are not selected.

Each JSONL line is one complete JSON object. `json_key` selects the field.
In the example above, these fields are `text` and `conversation`.
The examples below are real records with text truncated (`...`); only the fields
needed for these examples are shown.

**Text** (An example when `json_key` is `text`):

```jsonl
{"text": "Suppose 4*b + 8 = 0, -b = 2*x + x - 10. Let f be 10/24 + x/(-6). Let w(j) = -3*j + 14. Let t be w(6). What is the second biggest value in f, t, 2/3?\n\n..."}
```

**Chat** (An example when `json_key` is `conversation`):

```jsonl
{"conversation": [{"role": "user", "content": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\nThere are 20 red marbles, 10 blue marbles, and 5 white marbles in a jar. Select a marble without looking, note the color, and then ..."}, {"role": "assistant", "content": "To guarantee that a red marble has been drawn we must consider ..."}]}
```

| `source_format` | Field value | Processing |
| --- | --- | --- |
| `text` / `content` | String | Tokenize with BOS/EOS for pretraining. |
| `chat_assistant` | Message list | Use the tokenizer's named chat template and assistant mask. |
| `chat_assistant_tools` | Message list | Use the tokenizer's named tool-chat template and assistant mask. |

Chat requires a compatible HF tokenizer with the selected template and
assistant-mask support. Text fields must be strings, not token-ID arrays.

## Training Arguments

Add these data arguments to your `train.py` launch:

```bash
DATA_CFGS=(
  --data "$DATA"
  --tokenizer.type huggingface
  --tokenizer.path /path/to/tokenizer
  --seq_len 8192
  --batch_size 4
  --dataloader.buffer_size 512
  --dataloader.num_workers 2
  --dataloader.packing_type bestfit
  --dataloader.skip_long_docs false
)
```

| Setting | Meaning |
| --- | --- |
| `batch_size` | Batch size per data-parallel rank. |
| `global_batch_size` | Alternative to `batch_size`: total batch size across data-parallel ranks; must be divisible by the data-parallel size. |
| `buffer_size` | Controls the shuffle window: approximately `buffer_size * seq_len` tokens per rank per refill. |
| `num_workers` | Threads reading and tokenizing different sources; one source is not split across workers. |
| `packing_type` | `simple` concatenates tokens; `bestfit` uses length-aware packing to reduce padding, keeping documents intact when they fit in one sequence. |
| `skip_long_docs` | With `bestfit`, `true` skips overlong chat records; text records are still split. |

Set either `batch_size` or `global_batch_size`, not both.

## Bestfit Packing

Our `bestfit` implementation is inspired by
[Fewer Truncations Improve Language Modeling (ICML 2024)](https://arxiv.org/abs/2404.10830).
It packs **online, within each rank's buffer**, rather than preprocessing the
full dataset. `buffer_size` controls a token budget, not a count of documents.

Data from different sources is shuffled together before packing. Chunks are
sorted by decreasing length, then placed in the sequence with the smallest
remaining space that can fit them. Documents that fit are kept intact. Overlong
documents are split, or skipped for chat when `skip_long_docs=true`; split chunks
overlap by one token to preserve next-token prediction. Chunks that cannot yet be
emitted are retained for the next refill. Packed sequences are shuffled before
training.

This trades buffer memory and refill work for less padding and a wider shuffle
window, without a full-dataset packing pass. With `log_wandb=true`, training
already reports `train/padding ratio` and `train/truncation ratio` for `bestfit`.

## Output and Resume

Training batches contain NumPy `x`, `y`, and boolean `mask` arrays of shape
`[batch_size, seq_len]`. `y` is the next-token target; ignored positions use
`y=-100` and `mask=False`. Input padding uses EOS.

The loader supports checkpoint/resume. Keep the datasets, tokenizer, data mix,
packing/buffer settings, batch size, and data-parallel layout unchanged to
reproduce subsequent batches.

## Evaluation

- **PPL:** reads a single JSONL file with a `text` field, without the training
  source string or chunk naming requirement.
- **Tasks such as GSM8K:** use their task-specific data formats and processing.
