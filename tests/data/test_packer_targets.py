import numpy as np

from xllm.data.buffer_assembler.packer.packer import (
    BestFitPacker,
    IGNORE_INDEX,
    SimpleConcatPacker,
    _apply_ignore_index_to_targets,
)
from xllm.data.data_types import Instance


def _instance(tokens, target_mask=None, line_num=1):
    return Instance(
        raw_data={},
        filename="test.jsonl",
        file_pos=0,
        line_num=line_num,
        tokens=list(tokens),
        target_mask=None if target_mask is None else list(target_mask),
    )


def test_simple_concat_predicts_bos_across_document_boundaries():
    packer = SimpleConcatPacker(seq_len=4, pad_id=0)
    packer.prefill("ds", _instance([10, 11, 12], line_num=1))
    packer.prefill("ds", _instance([20, 21, 22], line_num=2))

    seqs = packer.pack()

    assert len(seqs) == 1
    assert np.array_equal(seqs[0].x, np.array([10, 11, 12, 20], dtype=np.int64))
    assert seqs[0].mask is None
    assert np.array_equal(seqs[0].y, np.array([11, 12, 20, 21], dtype=np.int64))


def test_best_fit_predicts_bos_across_document_boundaries():
    packer = BestFitPacker(seq_len=5, pad_id=0, min_fill_ratio=0.0)
    packer.prefill("ds", _instance([10, 11, 12], line_num=1))
    packer.prefill("ds", _instance([20, 21, 22], line_num=2))

    seqs = packer.pack()

    assert len(seqs) == 1
    assert np.array_equal(seqs[0].x, np.array([10, 11, 12, 20, 21], dtype=np.int64))
    assert seqs[0].mask is None
    assert np.array_equal(seqs[0].y, np.array([11, 12, 20, 21, 22], dtype=np.int64))


def test_best_fit_uses_dl1_input_padding_and_ignore_index_targets():
    packer = BestFitPacker(seq_len=4, pad_id=1, min_fill_ratio=0.0)
    packer.prefill("ds", _instance([10, 11, 12]))

    seqs = packer.pack()

    assert len(seqs) == 1
    assert np.array_equal(seqs[0].x, np.array([10, 11, 1, 1], dtype=np.int64))
    assert np.array_equal(seqs[0].mask, np.array([True, True, False, False]))
    assert np.array_equal(seqs[0].y, np.array([11, 12, IGNORE_INDEX, IGNORE_INDEX], dtype=np.int64))


def test_best_fit_sets_sft_masked_targets_to_ignore_index():
    packer = BestFitPacker(seq_len=4, pad_id=0, min_fill_ratio=0.0)
    packer.prefill("ds", _instance(
        [100, 101, 102, 103, 104],
        target_mask=[False, True, False, True, True],
    ))

    seqs = packer.pack()

    assert len(seqs) == 1
    assert np.array_equal(seqs[0].x, np.array([100, 101, 102, 103], dtype=np.int64))
    assert np.array_equal(seqs[0].mask, np.array([True, False, True, True]))
    assert np.array_equal(seqs[0].y, np.array([101, IGNORE_INDEX, 103, 104], dtype=np.int64))


def test_ignore_index_is_applied_in_place():
    y = np.array([10, 11, 12], dtype=np.int64)
    mask = np.array([True, False, True], dtype=np.bool_)

    result = _apply_ignore_index_to_targets(y, mask)

    assert result is y
    assert np.array_equal(y, np.array([10, IGNORE_INDEX, 12], dtype=np.int64))


def test_best_fit_marks_every_non_terminal_full_chunk_as_truncated():
    packer = BestFitPacker(seq_len=4, pad_id=99, min_fill_ratio=0.0)
    packer.prefill("ds", _instance(list(range(11))))

    seqs = packer.pack()

    assert [seq.has_truncation for seq in seqs] == [True, True, False]


def test_best_fit_does_not_mark_terminal_full_chunk_as_truncated():
    packer = BestFitPacker(seq_len=4, pad_id=99, min_fill_ratio=0.0)
    packer.prefill("ds", _instance(list(range(9))))

    seqs = packer.pack()

    assert [seq.has_truncation for seq in seqs] == [True, False]


def test_best_fit_checkpoint_stores_tokenized_pending_chunks():
    packer = BestFitPacker(seq_len=4, pad_id=99)
    packer.prefill(
        "ds",
        _instance(
            list(range(8)),
            target_mask=[True, False, True, False, True, False, True, False],
        ),
    )
    packer.pack()

    state = packer.get_state()
    assert len(state["pending_chunks"]) == 1
    tokens, target_mask, source_info = state["pending_chunks"][0]
    assert tokens == [4, 5, 6, 7]
    assert target_mask == [True, False, True, False]
    assert source_info.dataset == "ds"
    assert source_info.is_truncation is False

    restored = BestFitPacker(seq_len=4, pad_id=99)
    restored.set_state(state)
    assert restored.pending_chunks == packer.pending_chunks

    state["pending_chunks"][0][0][0] = -1
    assert restored.pending_chunks[0][0][0] == 4


def test_simple_checkpoint_stores_tokenized_leftover_chunks():
    packer = SimpleConcatPacker(seq_len=4, pad_id=99)
    packer.prefill("ds", _instance([10, 11, 12]))
    assert packer.pack() == []

    state = packer.get_state()
    assert state["leftover_chunks"][0][0] == [10, 11, 12]

    restored = SimpleConcatPacker(seq_len=4, pad_id=99)
    restored.set_state(state)
    assert restored._leftover_chunks == packer._leftover_chunks
