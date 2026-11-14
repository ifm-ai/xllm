"""Fast restore of the buffer assembler: the saved active sequences and next-refill state give the batches of an
uninterrupted run, over several refills after the checkpoint, at any point of a slot (and at its end), with the
refill inline or in its thread; a state without them (an older checkpoint) still restores by the rebuild."""
import copy
import json
import os

import numpy as np
import pytest

from xllm.data.buffer_assembler.buffer_assembler import decode_packed_seqs, encode_packed_seqs
from xllm.data.dataloader import MultiSourceDataLoader
from xllm.data.data_types import PackedSeq, SourceInfo

from tests.data.test_local_resume import _Tokenizer, _assert_batch_equal


def _build_loader(data_dir, packing_type, sync_refill):
    # 4 buffered sequences of 4 tokens: a refill every few batches, and enough tokens per refill for the simple
    # packer (with 2, a refill of this data can produce no sequence)
    loader = MultiSourceDataLoader(
        tokenizer=_Tokenizer(), data_mix_str=f"{data_dir}:1.0:text:text", seq_len=4, batch_size=1,
        num_buffered_seq=4, world_rank=0, world_size=1, packing_type=packing_type, num_workers=1,
    )
    loader.assembler._sync_refill = sync_refill
    return loader


def _write_data(data_dir, n_docs=200):
    with open(os.path.join(data_dir, "data.chunk0.jsonl"), "w", encoding="utf-8") as handle:
        for idx in range(n_docs):
            handle.write(json.dumps({"text": f"doc{idx:05d}" + "x" * (idx % 7)}) + "\n")


def _without_fast_restore(state):
    state = copy.deepcopy(state)
    state["assembler"].pop("active_seqs")
    state["assembler"].pop("next_refill_state")
    return state


@pytest.mark.parametrize("sync_refill", [True, False])
@pytest.mark.parametrize("packing_type", ["simple", "bestfit"])
@pytest.mark.parametrize("n_before", range(10))
def test_fast_restore_matches_uninterrupted_run(tmp_path, n_before, packing_type, sync_refill):
    _write_data(str(tmp_path))
    loaders = []
    try:
        reference = _build_loader(str(tmp_path), packing_type, sync_refill)
        loaders.append(reference)
        for _ in range(n_before):
            next(reference)
        state = copy.deepcopy(reference.get_state())
        expected = [next(reference) for _ in range(10)]
        if n_before > 0:
            assert "active_seqs" in state["assembler"] and "next_refill_state" in state["assembler"]
        for restore_state in ([state, _without_fast_restore(state)] if n_before > 0 else [state]):
            restored = _build_loader(str(tmp_path), packing_type, sync_refill)
            loaders.append(restored)
            restored.set_state(copy.deepcopy(restore_state))
            for expected_batch, actual_batch in zip(expected, [next(restored) for _ in range(10)]):
                _assert_batch_equal(expected_batch, actual_batch)
            # the state after the restore is again the uninterrupted run's
            again = restored.get_state()
            assert again["steps"] == reference.get_state()["steps"]
    finally:
        for loader in loaders:
            loader.close()


def test_packed_seqs_round_trip():
    rng = np.random.default_rng(0)
    seqs = []
    for i in range(6):
        n = int(rng.integers(1, 9))
        seqs.append(PackedSeq(
            x=rng.integers(0, 250_000, n, dtype=np.int64),
            y=np.where(rng.random(n) < 0.3, -100, rng.integers(0, 250_000, n)).astype(np.int64),
            mask=(rng.random(n) < 0.8) if i % 2 else None,
            src_infos=[SourceInfo("ds", f"f{i}.jsonl", i)] if i % 3 else None,
            padding_count=i, has_truncation=bool(i % 2),
        ))
    # one value beyond int32 keeps that array int64
    seqs[0].x[0] = 2 ** 40
    enc = encode_packed_seqs(seqs)
    assert enc["x"].dtype == np.int64 and enc["y"].dtype == np.int32
    for a, b in zip(seqs, decode_packed_seqs(enc)):
        assert a.x.dtype == b.x.dtype and np.array_equal(a.x, b.x)
        assert a.y.dtype == b.y.dtype and np.array_equal(a.y, b.y)
        assert (a.mask is None) == (b.mask is None) and (a.mask is None or np.array_equal(a.mask, b.mask))
        assert (a.src_infos, a.padding_count, a.has_truncation) == (b.src_infos, b.padding_count, b.has_truncation)
    assert decode_packed_seqs(encode_packed_seqs([])) == []
