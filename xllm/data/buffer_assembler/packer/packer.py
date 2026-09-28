from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
import numpy as np

from xllm.data.data_types import Instance, PackedSeq, SourceInfo


_Chunk = Tuple[List[int], Optional[List[bool]], SourceInfo]
IGNORE_INDEX = -100


def _apply_ignore_index_to_targets(y: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return y
    y[~mask] = IGNORE_INDEX
    return y


def _chunk_from_instance(ds: str, inst: Instance) -> _Chunk:
    assert inst.tokens is not None
    if inst.target_mask is not None:
        assert len(inst.target_mask) == len(inst.tokens), (len(inst.target_mask), len(inst.tokens))
    info = SourceInfo(
        dataset=ds,
        filename=inst.filename,
        line_num=inst.line_num,
        is_truncation=False,
    )
    mask = None if inst.target_mask is None else list(inst.target_mask)
    return list(inst.tokens), mask, info


def _slice_chunk(
    chunk: _Chunk,
    start: int,
    end: int,
    *,
    is_truncation: Optional[bool] = None,
) -> _Chunk:
    tokens, mask, info = chunk
    assert 0 <= start <= end <= len(tokens), (start, end, len(tokens))
    new_info = SourceInfo(
        dataset=info.dataset,
        filename=info.filename,
        line_num=info.line_num,
        is_truncation=info.is_truncation if is_truncation is None else is_truncation,
    )
    new_mask = None if mask is None else list(mask[start:end])
    return list(tokens[start:end]), new_mask, new_info


def _flatten_chunks(
    chunks: List[_Chunk],
) -> Tuple[List[int], List[bool], bool, List[Tuple[int, int, SourceInfo]], bool]:
    tokens: List[int] = []
    target_mask: List[bool] = []
    has_mask = False
    has_truncation = False
    spans: List[Tuple[int, int, SourceInfo]] = []

    cursor = 0
    for chunk_idx, (chunk_tokens, chunk_mask, info) in enumerate(chunks):
        start = cursor
        tokens.extend(chunk_tokens)
        cursor += len(chunk_tokens)
        if chunk_mask is not None:
            chunk_target_mask = list(chunk_mask)
            has_mask = True
        else:
            chunk_target_mask = [True] * len(chunk_tokens)

        # TODO: EOS should not predict BOS across packed document boundaries.
        # if chunk_idx > 0 and chunk_target_mask:
        #     chunk_target_mask[0] = False
        #     has_mask = True
        target_mask.extend(chunk_target_mask)

        spans.append((start, cursor, info))
        if info.is_truncation:
            has_truncation = True

    return tokens, target_mask, has_mask, spans, has_truncation


def _src_infos_for_window(
    spans: List[Tuple[int, int, SourceInfo]],
    token_start: int,
    token_end: int,
) -> List[SourceInfo]:
    return [
        info for (start, end, info) in spans
        if start < token_end and end > token_start
    ]


class Packer:
    @abstractmethod
    def prefill(self, ds: str, ex: Instance) -> None:
        raise NotImplementedError

    @abstractmethod
    def pack(self) -> List[PackedSeq]:
        """Pack the prefill buffer and return the resulting sequences."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return the tokenized chunks that must survive across refills."""
        return {}

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        pass


class SimpleConcatPacker(Packer):
    def __init__(self, seq_len: int, pad_id: int = 0):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.prefill_buffer: List[Tuple[str, Instance]] = []
        self._leftover_chunks: List[_Chunk] = []

    def prefill(self, ds: str, ex: Instance) -> None:
        self.prefill_buffer.append((ds, ex))

    def pack(self) -> List[PackedSeq]:
        stream_chunks: List[_Chunk] = list(self._leftover_chunks)
        for ds, inst in self.prefill_buffer:
            stream_chunks.append(_chunk_from_instance(ds, inst))

        all_tokens, target_mask, has_mask, token_spans, _has_truncation = _flatten_chunks(stream_chunks)

        seqs: List[PackedSeq] = []
        n_complete = max(0, (len(all_tokens) - 1) // self.seq_len)
        for seq_idx in range(n_complete):
            slice_start = seq_idx * self.seq_len
            slice_end = slice_start + self.seq_len
            token_end = slice_end + 1
            x_arr = np.array(all_tokens[slice_start:slice_end], dtype=np.int64)
            y_arr = np.array(all_tokens[slice_start + 1:token_end], dtype=np.int64)
            mask_arr = (
                np.array(target_mask[slice_start + 1:token_end], dtype=np.bool_)
                if has_mask
                else None
            )
            y_arr = _apply_ignore_index_to_targets(y_arr, mask_arr)
            src_infos = _src_infos_for_window(token_spans, slice_start, token_end)
            seqs.append(
                PackedSeq(
                    x=x_arr,
                    y=y_arr,
                    mask=mask_arr,
                    src_infos=src_infos,
                    padding_count=0,
                    has_truncation=False,
                )
            )

        keep_start = n_complete * self.seq_len
        self._leftover_chunks = []
        if keep_start < len(all_tokens):
            cursor = 0
            for chunk in stream_chunks:
                tokens = chunk[0]
                start, end = cursor, cursor + len(tokens)
                if end > keep_start:
                    chunk_keep_start = max(keep_start, start) - start
                    chunk_keep_end = end - start
                    self._leftover_chunks.append(_slice_chunk(chunk, chunk_keep_start, chunk_keep_end))
                cursor = end

        self.prefill_buffer = []
        return seqs

    def get_state(self) -> Dict[str, Any]:
        return {
            "leftover_chunks": deepcopy(self._leftover_chunks),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.prefill_buffer = []
        self._leftover_chunks = []
        if not state:
            return
        self._leftover_chunks = deepcopy(state.get("leftover_chunks", []))


class BestFitPacker(Packer):
    """Best-Fit Decreasing bin packer over complete token chunks."""

    def __init__(
        self,
        seq_len: int,
        max_active_bins: int = 1024,
        pad_id: int = 0,
        min_fill_ratio: float = 0.99,
        skip_long_docs: bool = False,
        source_formats: Optional[Dict[str, str]] = None,
    ):
        self.seq_len = seq_len
        self.bin_capacity = seq_len + 1
        self.max_active_bins = max_active_bins
        self.pad_id = pad_id
        self.min_fill_ratio = min_fill_ratio
        self.skip_long_docs = skip_long_docs
        self.source_formats = dict(source_formats or {})

        self.prefill_buffer: List[Tuple[str, Instance]] = []
        self.pending_chunks: List[_Chunk] = []
        self.bins: Dict[int, List[_Chunk]] = defaultdict(list)
        self.bin_lengths: Dict[int, int] = defaultdict(int)
        self.next_bin_id: int = 0
        self._reset_tree()

    def _reset_tree(self) -> None:
        self.next_bin_id = 0
        self.tree: List[int] = [0] * (self.bin_capacity * 2)
        self.space2bin: Dict[int, deque] = defaultdict(deque)

    def _tree_update(self, k: int, target: int) -> None:
        if k < 1 or k > self.bin_capacity:
            return
        leaf_idx = len(self.tree) // 2 - 1 + k
        if self.tree[leaf_idx] == target:
            return
        self.tree[leaf_idx] = target
        i = leaf_idx // 2
        while i > 0:
            self.tree[i] = max(self.tree[i * 2], self.tree[i * 2 + 1])
            i //= 2

    def _find_best_fit_bin(self, doc_len: int) -> Tuple[Optional[int], Optional[int]]:
        if doc_len > self.bin_capacity:
            return None, None
        if self.tree[1] >= doc_len:
            i = 1
            while i < len(self.tree) // 2:
                left, right = i * 2, i * 2 + 1
                i = left if self.tree[left] >= doc_len else right
            rem = self.tree[i]
            bin_id = self.space2bin[rem].popleft()
            if not self.space2bin[rem]:
                del self.space2bin[rem]
                self._tree_update(rem, 0)
            return bin_id, rem
        if self.next_bin_id < self.max_active_bins:
            bin_id = self.next_bin_id
            self.next_bin_id += 1
            return bin_id, self.bin_capacity
        return None, None

    def prefill(self, ds: str, ex: Instance) -> None:
        self.prefill_buffer.append((ds, ex))

    def pack(self) -> List[PackedSeq]:
        for ds, inst in self.prefill_buffer:
            full_chunk = _chunk_from_instance(ds, inst)
            n = len(full_chunk[0])
            if n < 2:
                continue
            if n > self.bin_capacity:
                source_format = self.source_formats.get(ds, "text")
                if self.skip_long_docs and source_format not in {"text", "content"}:
                    continue
                for start in range(0, n - 1, self.seq_len):
                    end = min(start + self.bin_capacity, n)
                    self.pending_chunks.append(
                        _slice_chunk(
                            full_chunk,
                            start,
                            end,
                            is_truncation=(end < n),
                        )
                    )
            else:
                self.pending_chunks.append(full_chunk)
        self.prefill_buffer = []

        self.pending_chunks.sort(key=lambda c: len(c[0]), reverse=True)

        unpacked: List[_Chunk] = []
        for tokens, mask, info in self.pending_chunks:
            bin_id, old_space = self._find_best_fit_bin(len(tokens))
            if bin_id is not None:
                self.bins[bin_id].append((tokens, mask, info))
                self.bin_lengths[bin_id] += len(tokens)
                new_space = old_space - len(tokens)
                if new_space > 0:
                    self.space2bin[new_space].append(bin_id)
                    self._tree_update(new_space, new_space)
            else:
                unpacked.append((tokens, mask, info))
        self.pending_chunks = unpacked

        seqs: List[PackedSeq] = []
        recycled: List[_Chunk] = []
        for bin_id, chunks in list(self.bins.items()):
            pair_count = max(0, self.bin_lengths[bin_id] - 1)
            fill_ratio = pair_count / self.seq_len
            if fill_ratio >= self.min_fill_ratio:
                tokens, target_mask, has_mask, token_spans, has_truncation = _flatten_chunks(chunks)
                pair_count = min(max(0, len(tokens) - 1), self.seq_len)
                if pair_count == 0:
                    continue
                x = list(tokens[:pair_count])
                y = list(tokens[1:pair_count + 1])
                masks = list(target_mask[1:pair_count + 1])
                padding_count = self.seq_len - pair_count
                if padding_count:
                    x.extend([self.pad_id] * padding_count)
                    y.extend([self.pad_id] * padding_count)
                    masks.extend([False] * padding_count)
                x_arr = np.array(x, dtype=np.int64)
                y_arr = np.array(y, dtype=np.int64)
                mask_arr = (
                    np.array(masks, dtype=np.bool_)
                    if has_mask or padding_count
                    else None
                )
                y_arr = _apply_ignore_index_to_targets(y_arr, mask_arr)
                src_infos = _src_infos_for_window(token_spans, 0, pair_count + 1)
                seqs.append(
                    PackedSeq(
                        x=x_arr,
                        y=y_arr,
                        mask=mask_arr,
                        src_infos=src_infos,
                        padding_count=padding_count,
                        has_truncation=has_truncation,
                    )
                )
            else:
                recycled.extend(chunks)

        self.bins.clear()
        self.bin_lengths.clear()
        self._reset_tree()
        self.pending_chunks.extend(recycled)

        return seqs

    def get_state(self) -> Dict[str, Any]:
        return {
            "pending_chunks": deepcopy(self.pending_chunks),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.prefill_buffer = []
        self.pending_chunks = []
        self.bins.clear()
        self.bin_lengths.clear()
        self._reset_tree()
        if not state:
            return
        self.pending_chunks = deepcopy(state.get("pending_chunks", []))
