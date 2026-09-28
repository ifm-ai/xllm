"""Streaming training batches with weighted sources, packing, and resumable prefetch.

PPLDataLoader provides a separate finite JSONL path for perplexity evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
import numpy as np

from xllm.data.dataset_streamer.dataset_streamer import DatasetStreamer
from xllm.data.dataset_streamer.data_iterator.data_iterator import (
    JSONLFileIterator,
    JSONLFolderIterator,
)
from xllm.data.dataset_streamer.feature_builder.feature_builder import (
    ChatTemplateFeatureBuilder,
    DefaultFeatureBuilder,
    FeatureBuilder,
)
from xllm.data.dataset_streamer.templator.templator import SimpleJsonlTemplator
from xllm.data.buffer_assembler.buffer_assembler import BufferAssembler
from xllm.data.buffer_assembler.packer.packer import SimpleConcatPacker, BestFitPacker
from xllm.data.data_types import PackedSeq, Batch
from xllm.data.dataset_streamer.tokenizer import Tokenizer

CHAT_SOURCE_FORMATS = {"chat_assistant", "chat_assistant_tools"}
TEXT_SOURCE_FORMATS = {"text", "content"}
SUPPORTED_SOURCE_FORMATS = CHAT_SOURCE_FORMATS | TEXT_SOURCE_FORMATS

class MultiSourceDataLoader:
    """
    MultiSourceDataLoader:
    - source-level worker threads and double-buffered prefetch
    - creates DatasetStreamers (per dataset) and BufferAssembler
    - yields Batch

    Arguments:
    - data_mix_str: e.g. "path1:0.6:text:text,path2:0.4:conversation:chat_assistant"
    - seq_len: Sequence length.
    - batch_size: Batch size per data-parallel rank.
    - num_buffered_seq: Refill token-budget multiplier. Each refill targets roughly
      num_buffered_seq * seq_len tokens across the weighted data mix before packing.
      The actual number of packed sequences can differ because records are read
      whole, over-read is credited against later refills, and best-fit may carry
      pending chunks across refills.
    - world_rank: Rank of current GPU process.
    - world_size: Total number of ranks.
    - packing_type: "simple" concatenation or "bestfit" bin packing.
    - num_workers: Number of tokenizer worker threads used during buffer refill.
    - max_consecutive_skips: Fail a source after this many consecutive records
      fail to produce tokenized features.
    """
    def __init__(
        self,
        tokenizer,
        *,
        data_mix_str: str,
        seq_len: int,
        batch_size: int,
        num_buffered_seq: int,
        world_rank: int,
        world_size: int, 
        packing_type: str = "simple",
        num_workers: int = 1,
        max_consecutive_skips: int = 10000,
        skip_long_docs: bool = False,
    ):
        dataset_weights, dataset_json_keys, dataset_source_formats = self._parse_data_mix_str(data_mix_str)

        self.seq_len = seq_len
        self.num_buffered_seq = num_buffered_seq
        self.batch_size = batch_size

        # shared components
        self.tokenizer = tokenizer

        dataset_name_candidates = [Path(dataset_dir).name for dataset_dir in dataset_weights]
        if len(dataset_name_candidates) == len(set(dataset_name_candidates)):
            self._dataset_display_names = {
                dataset_dir: Path(dataset_dir).name for dataset_dir in dataset_weights
            }
        else:
            self._dataset_display_names = {
                dataset_dir: dataset_dir for dataset_dir in dataset_weights
            }
        self.src_names = list(self._dataset_display_names.values())

        eos_id = int(self.tokenizer.eos_id)
        if eos_id < 0:
            raise ValueError("Tokenizer must provide a non-negative eos_id")
        self.pad_id = eos_id

        _packing_type = packing_type.lower()
        if _packing_type == "bestfit":
            self.packer = BestFitPacker(
                seq_len=seq_len,
                max_active_bins=max(1, int(num_buffered_seq * 1.02)),
                pad_id=self.pad_id,
                skip_long_docs=skip_long_docs,
                source_formats=dataset_source_formats,
            )
        elif _packing_type == "simple":
            self.packer = SimpleConcatPacker(seq_len=seq_len, pad_id=self.pad_id)
        else:
            raise ValueError(f"Unknown packing_type: {packing_type!r}. Supported: 'simple', 'bestfit'")

        # build streamers
        self.streamers: Dict[str, DatasetStreamer] = {}  # keyed by dataset_dir
        for dataset_dir in dataset_weights.keys():
            self.streamers[dataset_dir] = self._build_streamer(
                dataset_dir=dataset_dir,
                json_key=dataset_json_keys[dataset_dir],
                source_format=dataset_source_formats[dataset_dir],
                world_rank=world_rank,
                world_size=world_size,
            )

        self.assembler = BufferAssembler(
            dataset_streamers=self.streamers,
            dataset_weights=dataset_weights,
            packer=self.packer,
            seq_len=seq_len,
            num_buffered_seq=num_buffered_seq,
            num_tokenizer_workers=num_workers,
            max_consecutive_skips=max_consecutive_skips,
            world_rank=world_rank,
            world_size=world_size,
        )
        self._steps = 0

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        seqs: List[PackedSeq] = []
        for _ in range(self.batch_size):
            seqs.append(self.assembler.pop_packed())
        self._steps += 1
        
        # Compute padding and truncation ratios
        total_tokens = self.batch_size * self.seq_len
        total_padding = sum(seq.padding_count for seq in seqs)
        truncation_count = sum(1 for seq in seqs if seq.has_truncation)
        
        padding_ratio = total_padding / total_tokens if total_tokens > 0 else 0.0
        truncation_ratio = truncation_count / self.batch_size if self.batch_size > 0 else 0.0
        
        # Keep the model input contract uniform across ranks and batches.
        masks = [seq.mask for seq in seqs]
        combined_mask = np.array([
            m if m is not None else np.ones(self.seq_len, dtype=bool)
            for m in masks
        ])

        src_infos = (
            [seq.src_infos for seq in seqs]
            if any(seq.src_infos is not None for seq in seqs)
            else None
        )
        src_names = [self._seq_src_names(seq.src_infos) for seq in seqs]

        return Batch(
            x=np.array([seq.x for seq in seqs]),
            y=np.array([seq.y for seq in seqs]),
            mask=combined_mask,
            src_names=src_names,
            src_infos=src_infos,
            padding_ratio=padding_ratio,
            truncation_ratio=truncation_ratio,
        )

    @staticmethod
    def _parse_data_mix_str(data_mix_str: str) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str]]:
        # Supported entries:
        #   path:weight:json_key:source_format
        # Weights not summed to one are allowed.
        data_mix = {}
        data_json_keys = {}
        data_source_formats = {}
        total_weight = 0.0
        for part in data_mix_str.split(','):
            part = part.strip()
            if not part:
                continue
            fields = part.rsplit(':', 3)
            if len(fields) != 4:
                raise ValueError(
                    "Dataset mix entries must use "
                    "'path:weight:json_key:source_format', got "
                    f"{part!r}"
                )
            dataset_dir, weight_str, json_key, source_format = fields
            weight = float(weight_str)
            if not json_key:
                raise ValueError(f"Dataset json_key must be non-empty for {dataset_dir!r}")
            if source_format not in SUPPORTED_SOURCE_FORMATS:
                raise ValueError(
                    f"Unknown source_format {source_format!r} for {dataset_dir!r}. "
                    f"Supported: {sorted(SUPPORTED_SOURCE_FORMATS)}"
                )

            if weight <= 0:
                raise ValueError(f"Dataset weight must be positive for {dataset_dir!r}: {weight}")
            total_weight += weight
            data_mix[dataset_dir] = weight
            data_json_keys[dataset_dir] = json_key
            data_source_formats[dataset_dir] = source_format

        if total_weight <= 0 or not data_mix:
            raise ValueError(f"Empty or invalid data_mix_str: {data_mix_str!r}")

        for dataset_dir, weight in data_mix.items():
            data_mix[dataset_dir] = weight / total_weight
        return data_mix, data_json_keys, data_source_formats

    def _build_streamer(
        self,
        *,
        dataset_dir: str,
        json_key: str,
        source_format: str,
        world_rank: int,
        world_size: int,
    ) -> DatasetStreamer:
        feature_builder = self._build_feature_builder(json_key, source_format)
        return DatasetStreamer(
            dataset_dir=dataset_dir,
            data_iterator=JSONLFolderIterator(dataset_dir, world_rank, world_size, infinite=True),
            tokenizer=self.tokenizer,
            feature_builder=feature_builder,
        )

    def _build_feature_builder(self, json_key: str, source_format: str) -> FeatureBuilder:
        if source_format in CHAT_SOURCE_FORMATS:
            return ChatTemplateFeatureBuilder(source_format, conversation_key=json_key)
        return DefaultFeatureBuilder(SimpleJsonlTemplator(json_key))

    def _seq_src_names(self, infos) -> List[str]:
        if not infos:
            return []
        names: List[str] = []
        seen = set()
        for info in infos:
            name = self._dataset_display_names.get(info.dataset, info.dataset)
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def get_state(self) -> Dict[str, Any]:
        return {
            "steps": self._steps,
            "assembler": self.assembler.get_state(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._steps = int(state.get("steps", 0))
        self.assembler.set_state(state.get("assembler", {}))

    def close(self) -> None:
        """Stop all background threads. Call before process exit."""
        self.assembler.close()

# -------------------------
# PPL JSONL dataloader
# -------------------------


class PPLDataLoader:
    """Finite JSONL dataloader for validation perplexity."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data: str,
        seq_len: int,
        batch_size: int,
        world_rank: int,
        world_size: int,
        keep_tail: bool = True,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        assert data.endswith(".jsonl"), data
        self.tokenizer = tokenizer
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.world_rank = world_rank
        self.world_size = world_size
        self.keep_tail = keep_tail
        self.reader = JSONLFileIterator(data, world_rank, world_size, infinite=False)

    def __iter__(self) -> Iterator[Batch]:
        return self._batch_iterator()

    def _sample_to_tokens(self, sample: Any) -> List[int]:
        text = sample["text"]
        return self.tokenizer.encode(text, bos=True, eos=True)

    def _batch_iterator(self) -> Iterator[Batch]:
        n_buffer_toks = self.seq_len * self.batch_size
        tokens: List[int] = []

        for instance in self.reader:
            tokens.extend(self._sample_to_tokens(instance.raw_data))
            while len(tokens) > n_buffer_toks:
                x = np.array(tokens[:n_buffer_toks], dtype=np.int64).reshape(
                    self.batch_size, self.seq_len
                )
                y = np.array(tokens[1:n_buffer_toks + 1], dtype=np.int64).reshape(
                    self.batch_size, self.seq_len
                )
                tokens = tokens[n_buffer_toks:]
                yield Batch(x=x, y=y)

        yield from self._tail_batches(tokens)

    def _tail_batches(self, tokens: List[int]) -> Iterator[Batch]:
        if not tokens:
            return

        num_seqs = (len(tokens) - 1) // self.seq_len
        num_batches = num_seqs // self.batch_size
        if num_seqs > 0:
            batched_tokens = np.array(
                tokens[:num_seqs * self.seq_len + 1], dtype=np.int64
            )
            for i in range(num_batches):
                start = i * self.batch_size * self.seq_len
                end = (i + 1) * self.batch_size * self.seq_len
                yield Batch(
                    x=batched_tokens[start:end].reshape(self.batch_size, self.seq_len),
                    y=batched_tokens[start + 1:end + 1].reshape(
                        self.batch_size, self.seq_len
                    ),
                )

            if num_batches * self.batch_size < num_seqs:
                start = num_batches * self.batch_size * self.seq_len
                yield Batch(
                    x=batched_tokens[start:-1].reshape(-1, self.seq_len),
                    y=batched_tokens[start + 1:].reshape(-1, self.seq_len),
                )

        tail_tokens = tokens[num_seqs * self.seq_len:]
        if self.keep_tail and len(tail_tokens) > 1:
            tail_x, tail_y, tail_mask = self._pad_tail_sequence(tail_tokens)
            yield Batch(
                x=tail_x.reshape(1, self.seq_len),
                y=tail_y.reshape(1, self.seq_len),
                mask=tail_mask.reshape(1, self.seq_len),
            )

    def dummy_batch(self, length: int) -> Batch:
        x = np.full(
            (self.batch_size, length), fill_value=self.tokenizer.bos_id, dtype=np.int64
        )
        y = np.full(
            (self.batch_size, length), fill_value=self.tokenizer.eos_id, dtype=np.int64
        )
        return Batch(x=x, y=y)

    def _pad_tail_sequence(self, tokens: List[int]):
        pad_len = self.seq_len - len(tokens) + 1
        x = np.array(tokens[:-1] + [self.tokenizer.pad_id] * pad_len, dtype=np.int64)
        y = np.array(tokens[1:] + [self.tokenizer.pad_id] * pad_len, dtype=np.int64)
        mask = y != self.tokenizer.pad_id
        x[~mask] = self.tokenizer.eos_id
        y[~mask] = self.tokenizer.eos_id
        return x, y, mask

    def close(self) -> None:
        self.reader.close()
