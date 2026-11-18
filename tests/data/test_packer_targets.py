import numpy as np

from xllm.data.buffer_assembler.packer.packer import (
    BestFitHybridPacker,
    BestFitPacker,
    BestFitRoundRobinHybridPacker,
    BestFitRoundRobinPacker,
    BestFitTransversalPacker,
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



def _pack_lengths(packer_cls, seq_len, lengths, finish_threshold=None, **kwargs):
    packer = packer_cls(seq_len=seq_len, pad_id=0, min_fill_ratio=0.0, **kwargs)
    if finish_threshold is not None:
        packer.finish_threshold = finish_threshold
    for line_num, n in enumerate(lengths, start=1):
        packer.prefill("ds", _instance(range(100 * line_num, 100 * line_num + n), line_num=line_num))
    seqs = packer.pack()
    bins = sorted(tuple(info.line_num for info in seq.src_infos) for seq in seqs)
    return packer, bins


def test_best_fit_transversal_places_chunks_in_emptiest_bin():
    # Capacity 10, 20 tokens -> two bins opened up front.
    lengths = [5, 4, 4, 3, 2, 2]

    # Best-fit: 5,4 -> A; 4,3,2 -> B; last 2 opens a third bin.
    _, best = _pack_lengths(BestFitPacker, 9, lengths)
    assert best == [(1, 2), (3, 4, 5), (6,)]

    # Worst-fit: 5 -> A, 4 -> B, 4 -> B, 3 -> A, 2 -> A (tie, lowest id), 2 -> B.
    packer, transversal = _pack_lengths(BestFitTransversalPacker, 9, lengths)
    assert transversal == [(1, 4, 5), (2, 3, 6)]
    assert packer.pending_chunks == []


def test_best_fit_transversal_spreads_long_chunks():
    # Three long and three short chunks over three bins: one long per bin.
    packer, bins = _pack_lengths(BestFitTransversalPacker, 9, [8, 8, 8, 2, 2, 2])
    assert bins == [(1, 4), (2, 5), (3, 6)]
    assert packer.pending_chunks == []


def test_best_fit_transversal_carries_chunks_that_do_not_fit():
    # Capacity 5, 13 tokens -> two bins: {5}, {4}; the 4-chunk no longer fits.
    packer, bins = _pack_lengths(BestFitTransversalPacker, 4, [5, 4, 4])
    assert bins == [(1,), (2,)]
    assert [len(c[0]) for c in packer.pending_chunks] == [4]


def test_best_fit_transversal_caps_bins_at_max_active_bins():
    packer, bins = _pack_lengths(BestFitTransversalPacker, 9, [5, 5, 5, 5], max_active_bins=1)
    assert bins == [(1, 2)]
    assert len(packer.pending_chunks) == 2


def test_best_fit_hybrid_finish_threshold_is_5_percent_with_2000_floor():
    assert BestFitHybridPacker(seq_len=4096).finish_threshold == 2000
    assert BestFitHybridPacker(seq_len=131072).finish_threshold == 6553


def test_best_fit_hybrid_spreads_then_finishes_nearly_full_bins_best_fit():
    # Capacity 10, 20 tokens -> two bins; bins with < 3 tokens left are finished best-fit.
    # 5 -> A (5 left), 4 -> B (6 left), 4 -> B (2 left, finishing), 3 -> A (2 left, finishing),
    # 2 -> B (finishing, earliest of equal space), 2 -> A.
    packer, bins = _pack_lengths(BestFitHybridPacker, 9, [5, 4, 4, 3, 2, 2], finish_threshold=3)
    assert bins == [(1, 4, 6), (2, 3, 5)]
    assert packer.pending_chunks == []


def test_best_fit_hybrid_prefers_fullest_finishing_bin_over_emptiest_bin():
    # Capacity 10, three bins; bins with < 4 tokens left are finished best-fit.
    # 8 -> A (2 left), 7 -> B (3 left), 6 -> C (4 left, still open).
    # Hybrid: 3 -> B and 2 -> A close the finishing bins, then 2, 2 -> C.
    _, hybrid = _pack_lengths(BestFitHybridPacker, 9, [8, 7, 6, 3, 2, 2, 2], finish_threshold=4)
    assert hybrid == [(1, 5), (2, 4), (3, 6, 7)]
    # Worst-fit sends 3 to the emptiest bin C instead, leaving the last 2 unplaced.
    _, worst = _pack_lengths(BestFitTransversalPacker, 9, [8, 7, 6, 3, 2, 2, 2])
    assert worst == [(1, 6), (2, 5), (3, 4)]


def test_best_fit_hybrid_carries_chunks_that_fit_nowhere():
    packer, bins = _pack_lengths(BestFitHybridPacker, 4, [5, 4, 4], finish_threshold=2)
    assert bins == [(1,), (2,)]
    assert [len(c[0]) for c in packer.pending_chunks] == [4]


def test_best_fit_round_robin_cycles_and_skips_bins_that_are_too_full():
    # Capacity 10, 20 tokens -> two bins. 5 -> A, 4 -> B, 4 -> A, 3 -> B;
    # 2: A (1 left) is skipped -> B; last 2 fits nowhere and stays pending.
    packer, bins = _pack_lengths(BestFitRoundRobinPacker, 9, [5, 4, 4, 3, 2, 2])
    assert bins == [(1, 3), (2, 4, 5)]
    assert [len(c[0]) for c in packer.pending_chunks] == [2]


def test_best_fit_round_robin_caps_bins_at_max_active_bins():
    packer, bins = _pack_lengths(BestFitRoundRobinPacker, 9, [5, 5, 5, 5], max_active_bins=1)
    assert bins == [(1, 2)]
    assert len(packer.pending_chunks) == 2


def test_best_fit_round_robin_hybrid_uses_hybrid_finish_threshold():
    assert BestFitRoundRobinHybridPacker(seq_len=4096).finish_threshold == 2000
    assert BestFitRoundRobinHybridPacker(seq_len=131072).finish_threshold == 6553


def test_best_fit_round_robin_hybrid_finishes_best_fit_without_moving_cursor():
    # Capacity 10, 40 tokens -> four bins; bins with < 4 tokens left are finished best-fit.
    # 9 -> A, 7 -> B (3 left), 4 -> C; cursor at D.
    # 3 -> B (finishing, best-fit); the cursor stays at D, so the next 3 -> D.
    # 3 -> C (A, B full; C 3 left), 3 -> C (finishing), 3 -> D, 3 -> D; the 2 fits nowhere.
    packer, bins = _pack_lengths(
        BestFitRoundRobinHybridPacker, 9, [9, 7, 4, 3, 3, 3, 3, 3, 3, 2], finish_threshold=4
    )
    assert bins == [(1,), (2, 4), (3, 6, 7), (5, 8, 9)]
    assert [len(c[0]) for c in packer.pending_chunks] == [2]
