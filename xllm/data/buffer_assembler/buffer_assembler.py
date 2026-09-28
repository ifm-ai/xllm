from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import threading
import numpy as np

from xllm.data.dataset_streamer.dataset_streamer import DatasetStreamer
from xllm.data.buffer_assembler.packer.packer import Packer
from xllm.data.data_types import Instance, PackedSeq


class BufferAssembler:
    """
    Coordinates DatasetStreamers and a Packer to yield packed sequences.

    Owns the double buffer and prefetch thread. Enforces dataset mixing
    ratios via per-dataset token budgets. The Packer is responsible only
    for the packing logic itself.

    Checkpoint design (minimal storage):
      - active_start_state  : streamer positions at the START of the current
                              active slot's prefill — the rewind point.
      - active_rng_state    : numpy RNG state captured just before the shuffle
                              that produced the active slot's popping order.
      - active_packer_state : tokenized pending chunks captured BEFORE the
                              active slot's pack cycle.
      - active_token_credits_state: per-source over-read credits captured
                              BEFORE the active slot's prefill.
      - popping_idx         : how far into the active slot we've consumed.
      On reload we re-run one prefill+pack+shuffle synchronously (using the
      saved RNG state) to rebuild the active slot, then restore popping_idx.
    """

    def __init__(
        self,
        dataset_streamers: Dict[str, DatasetStreamer],
        dataset_weights: Dict[str, float],
        packer: Packer,
        seq_len: int,
        num_buffered_seq: int,
        num_tokenizer_workers: int = 1,
        max_consecutive_skips: int = 10000,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        self.streamers = dataset_streamers
        self.streamer_iters = {ds: st.iter_raw() for ds, st in self.streamers.items()}
        self.packer = packer
        self.seq_len = int(seq_len)
        self.world_rank = int(world_rank)
        self.world_size = int(world_size)
        if num_tokenizer_workers < 1:
            raise ValueError(f"num_tokenizer_workers must be >= 1, got {num_tokenizer_workers}")
        if max_consecutive_skips < 1:
            raise ValueError(f"max_consecutive_skips must be >= 1, got {max_consecutive_skips}")
        self.num_tokenizer_workers = int(num_tokenizer_workers)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self._tokenizer_executor: Optional[ThreadPoolExecutor] = None
        if self.num_tokenizer_workers > 1:
            self._tokenizer_executor = ThreadPoolExecutor(
                max_workers=self.num_tokenizer_workers,
                thread_name_prefix="xllm-tokenizer",
            )
        self.refill_token_budget = seq_len * num_buffered_seq
        self.ds_budgets: Dict[str, float] = {
            ds: w * self.refill_token_budget
            for ds, w in dataset_weights.items()
        }
        self._refill_order = list(self.streamers)
        # Positive credit means previous buffers over-read this source by that
        # many tokens. Future buffers subtract the credit from that source's
        # target before reading more records.
        self._token_credits: Dict[str, float] = {ds: 0.0 for ds in self.streamers}

        # Double buffer owned here; Packer knows nothing about it.
        self._buffers: List[List[PackedSeq]] = [[], []]
        self._idxs: List[int] = [0, 0]
        self._active: int = 0

        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_pending = False
        self._prefetch_error: Optional[BaseException] = None
        self._close_event = threading.Event()
        self._rng = np.random.RandomState((self.world_rank, self.world_size))
        self._state_lock = threading.Lock()

        # Snapshots captured before each prefill cycle.
        self._active_start_state: Dict[str, Any] = {}
        self._inactive_start_state: Dict[str, Any] = {}
        self._active_rng_state: Optional[Tuple] = None
        self._inactive_rng_state: Optional[Tuple] = None
        self._active_packer_state: Dict[str, Any] = {}
        self._inactive_packer_state: Dict[str, Any] = {}
        self._active_token_credits_state: Dict[str, float] = {}
        self._inactive_token_credits_state: Dict[str, float] = {}

        self._timing_enabled = os.environ.get("XLLM_LOG_DATALOADER_TIMING", "0").lower() in {
            "1", "true", "yes", "on",
        }
        self._timing_fh = None
        self._first_refill_timed = False
        if self._timing_enabled:
            rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
            log_dir = Path(os.environ.get("XLLM_DATALOADER_TIMING_DIR", "data_timing_logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            self._timing_fh = (log_dir / f"rank_{rank:05d}.jsonl").open("a", buffering=1)
        self._sync_refill = os.environ.get("XLLM_DATALOADER_SYNC_REFILL", "0").lower() in {
            "1", "true", "yes", "on",
        }

    def _write_first_refill_timing(self, total_seconds: float) -> None:
        if self._timing_fh is None:
            return
        record = {
            "event": "first_refill",
            "loader": "dataloader2",
            "total_seconds": float(total_seconds),
            "num_tokenizer_workers": int(self.num_tokenizer_workers),
            "world_rank": int(self.world_rank),
            "world_size": int(self.world_size),
        }
        self._timing_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Prefetch internals
    # ------------------------------------------------------------------

    def _tokenize_source_to_budget(
        self, ds: str, target_tokens: float
    ) -> Tuple[List[Instance], int]:
        streamer = self.streamers[ds]
        streamer_iter = self.streamer_iters[ds]

        instances: List[Instance] = []
        tokens_pushed = 0
        consecutive_skips = 0
        while tokens_pushed < target_tokens:
            if self._close_event.is_set():
                raise RuntimeError("Dataloader is closing")
            raw = next(streamer_iter)
            if self._close_event.is_set():
                raise RuntimeError("Dataloader is closing")
            ex = streamer.build_features(raw)
            if ex is None:
                consecutive_skips += 1
                if consecutive_skips >= self.max_consecutive_skips:
                    raise RuntimeError(
                        f"Dataset {ds!r} produced no tokenized features for "
                        f"{consecutive_skips} consecutive records; last record was "
                        f"{raw.filename}:{raw.line_num}"
                    )
                continue
            consecutive_skips = 0
            if ex.tokens is None:
                raise RuntimeError(f"Dataset {ds!r} produced an instance without tokens")
            instances.append(ex)
            tokens_pushed += len(ex.tokens)

        return instances, tokens_pushed

    def _tokenize_sources_to_budget_parallel(
        self,
        source_targets: List[Tuple[str, float, float, float]],
    ) -> List[Tuple[str, Instance]]:
        assert self._tokenizer_executor is not None

        futures: Dict[Future, Tuple[str, float, float, float]] = {}
        results: Dict[str, Tuple[List[Instance], int]] = {}

        for ds, target_tokens, budget, credit in source_targets:
            future = self._tokenizer_executor.submit(
                self._tokenize_source_to_budget,
                ds,
                target_tokens,
            )
            futures[future] = (ds, target_tokens, budget, credit)

        while futures:
            done, _pending = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                ds, _target_tokens, _budget, _credit = futures.pop(future)
                results[ds] = future.result()

        prefill_items: List[Tuple[str, Instance]] = []
        for ds, target_tokens, _budget, _credit in source_targets:
            instances_list, tokens = results[ds]
            prefill_items.extend((ds, instance) for instance in instances_list)
            self._token_credits[ds] = int(tokens) - float(target_tokens)
        return prefill_items

    def _refill(self, rng_state: Optional[Tuple] = None) -> None:
        """
        Core prefill+pack+shuffle.  If *rng_state* is provided it is restored
        before the shuffle so the popping order is exactly reproducible.
        Always updates self._inactive_rng_state with the pre-shuffle state.
        """
        measure_first_refill = self._timing_fh is not None and not self._first_refill_timed
        refill_start = time.perf_counter() if measure_first_refill else None
        if self._close_event.is_set():
            raise RuntimeError("Dataloader is closing")
        if rng_state is not None:
            self._rng.set_state(rng_state)
        self._inactive_rng_state = self._rng.get_state()

        source_targets: List[Tuple[str, float, float, float]] = []
        prefill_items: List[Tuple[str, Instance]] = []

        for ds in self._refill_order:
            if self._close_event.is_set():
                raise RuntimeError("Dataloader is closing")
            budget = self.ds_budgets[ds]
            credit = self._token_credits.get(ds, 0.0)
            target_tokens = budget - credit

            if target_tokens <= 0:
                self._token_credits[ds] = credit - budget
                continue

            if self._tokenizer_executor is None:
                instances, tokens_pushed = self._tokenize_source_to_budget(ds, target_tokens)
                prefill_items.extend((ds, instance) for instance in instances)
                self._token_credits[ds] = tokens_pushed - target_tokens
            else:
                source_targets.append((ds, target_tokens, budget, credit))

        if source_targets:
            prefill_items.extend(self._tokenize_sources_to_budget_parallel(source_targets))

        self._rng.shuffle(prefill_items)
        for ds, instance in prefill_items:
            self.packer.prefill(ds, instance)

        inactive = 1 - self._active
        seqs = self.packer.pack()

        self._rng.shuffle(seqs)

        self._buffers[inactive] = seqs
        self._idxs[inactive] = 0

        if measure_first_refill:
            assert refill_start is not None
            self._first_refill_timed = True
            self._write_first_refill_timing(time.perf_counter() - refill_start)

    def _do_refill(self) -> None:
        """Background wrapper: catches exceptions for the prefetch thread."""
        try:
            self._refill()
        except BaseException as e:
            if not self._close_event.is_set():
                self._prefetch_error = e

    def _kick_prefetch(self) -> None:
        if self._prefetch_pending:
            raise RuntimeError("Dataloader prefetch is already pending")
        self._prefetch_error = None
        # Snapshot packer and streamer state BEFORE the prefill thread starts.
        self._inactive_packer_state = self.packer.get_state()
        self._inactive_start_state = {ds: st.get_state() for ds, st in self.streamers.items()}
        self._inactive_token_credits_state = dict(self._token_credits)
        self._prefetch_pending = True
        if self._sync_refill:
            self._do_refill()
            return
        t = threading.Thread(target=self._do_refill, daemon=True)
        t.start()
        self._prefetch_thread = t

    def _wait_prefetch(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        self._prefetch_pending = False
        if self._prefetch_error is not None:
            raise RuntimeError("Prefetch thread failed") from self._prefetch_error

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pop_packed(self) -> PackedSeq:
        with self._state_lock:
            active = self._active
            idx = self._idxs[active]
            buf = self._buffers[active]
            if idx < len(buf):
                self._idxs[active] += 1
                return buf[idx]

        # Active slot exhausted: wait for prefetch, swap, start next prefetch.
        if not self._prefetch_pending:
            self._kick_prefetch()
        self._wait_prefetch()
        with self._state_lock:
            self._active_start_state = self._inactive_start_state
            self._active_rng_state = self._inactive_rng_state
            self._active_packer_state = self._inactive_packer_state
            self._active_token_credits_state = self._inactive_token_credits_state
            self._active = 1 - self._active

            active = self._active
            idx = self._idxs[active]
            buf = self._buffers[active]
            if idx >= len(buf):
                raise RuntimeError("Refill did not produce enough tokens to pop a packed sequence.")
            self._idxs[active] += 1
            seq = buf[idx]

        self._kick_prefetch()
        return seq

    def get_state(self) -> Dict[str, Any]:
        # Checkpoint only describes the active slot.  The inactive prefetch slot
        # is a performance cache and is intentionally discarded on resume.
        with self._state_lock:
            return copy.deepcopy({
                "active_start_state": self._active_start_state,
                "active_rng_state": self._active_rng_state,
                "active_packer_state": self._active_packer_state,
                "active_token_credits_state": self._active_token_credits_state,
                "popping_idx": self._idxs[self._active],
            })

    def set_state(self, state: Dict[str, Any]) -> None:
        self._wait_prefetch()

        # 1. Restore streamers to the start of the active slot's data.
        for ds, st_state in state.get("active_start_state", {}).items():
            self.streamers[ds].set_state(st_state)
        self.streamer_iters = {ds: st.iter_raw() for ds, st in self.streamers.items()}

        # 2. Restore tokenized chunks left pending before this refill, along
        #    with source budget credits.
        self.packer.set_state(state.get("active_packer_state", {}))
        saved_credits = state.get("active_token_credits_state", {})
        self._token_credits = {ds: float(saved_credits.get(ds, 0.0)) for ds in self.streamers}

        # 3. Rebuild the active slot synchronously, using the saved RNG state
        #    so the shuffle produces the same popping order.
        self._refill(rng_state=state.get("active_rng_state"))
        with self._state_lock:
            self._active = 1 - self._active  # make the just-filled slot active
            self._active_start_state = state.get("active_start_state", {})
            self._active_rng_state = state.get("active_rng_state")
            self._active_packer_state = state.get("active_packer_state", {})
            self._active_token_credits_state = state.get("active_token_credits_state", {})

            # 4. Restore exact popping position within the active slot.
            self._idxs[self._active] = state.get("popping_idx", 0)

        # 5. Kick prefetch for the next slot.
        self._kick_prefetch()

    def close(self) -> None:
        """Stop all background threads cleanly before process exit."""
        self._close_event.set()
        if self._tokenizer_executor is not None:
            self._tokenizer_executor.shutdown(wait=True, cancel_futures=True)
            self._tokenizer_executor = None
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
        for st in self.streamers.values():
            st.close()
        if self._timing_fh is not None:
            self._timing_fh.close()
            self._timing_fh = None
