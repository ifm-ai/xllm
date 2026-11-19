"""Routed expert ownership with EP = TP x E_out. CPU only:

    pytest tests/moe/test_expert_layout.py
    python tests/moe/test_expert_layout.py
"""
import importlib.util
import itertools
import sys
from unittest.mock import MagicMock

# GPU-only packages xllm.modules imports at module load; stub them only when they are not installed.
_MISSING = {pkg for pkg in ("xllm_extension", "flash_attn_2_cuda") if importlib.util.find_spec(pkg) is None}
for name in ("xllm_extension", "xllm_extension.ops", "flash_attn_2_cuda"):
    if name.split(".")[0] in _MISSING:
        sys.modules.setdefault(name, MagicMock())

from xllm.modules.moe.layout import ExpertLayout  # noqa: E402

# (n_experts, TP, E_out)
SHAPES = [
    (8, 1, 1), (8, 2, 1), (8, 8, 1), (32, 4, 1), (100, 2, 1),
    (8, 1, 8), (8, 2, 2), (32, 2, 4), (64, 4, 2), (100, 2, 2), (192, 8, 1), (192, 8, 2), (192, 8, 4),
]


def layouts(n_experts, tp, e_out):
    """The layouts of one expert group, by (TP rank, outer rank)."""
    for tp_rank, outer_rank in itertools.product(range(tp), range(e_out)):
        yield ExpertLayout(n_experts, tp, e_out, tp_rank, outer_rank)


def test_every_expert_owned_once_per_expert_group():
    for n_experts, tp, e_out in SHAPES:
        owned = []
        for layout in layouts(n_experts, tp, e_out):
            assert layout.n_local == n_experts // (tp * e_out)
            assert layout.end - layout.start == layout.n_local
            owned.extend(range(layout.start, layout.end))
        assert sorted(owned) == list(range(n_experts)), (n_experts, tp, e_out)


def test_one_replica_per_expert_group_is_the_tp_rule():
    for n_experts, tp, e_out in SHAPES:
        if e_out != 1:
            continue
        for layout in layouts(n_experts, tp, e_out):
            k = layout.tp_rank
            n_local = n_experts // tp
            assert (layout.n_local, layout.start, layout.end) == (n_local, k * n_local, (k + 1) * n_local)


def test_outer_ranks_split_their_tp_rank_share():
    for n_experts, tp, e_out in SHAPES:
        for layout in layouts(n_experts, tp, e_out):
            tp_share = ExpertLayout(n_experts, tp, 1, layout.tp_rank, 0)
            assert tp_share.start <= layout.start < layout.end <= tp_share.end


def test_examples():
    # 375B: 192 experts, TP 8, EP 16
    assert (ExpertLayout(192, 8, 2, 3, 0).start, ExpertLayout(192, 8, 2, 3, 0).end) == (72, 84)
    assert (ExpertLayout(192, 8, 2, 3, 1).start, ExpertLayout(192, 8, 2, 3, 1).end) == (84, 96)
    assert (ExpertLayout(192, 8, 2, 7, 1).start, ExpertLayout(192, 8, 2, 7, 1).end) == (180, 192)
    assert (ExpertLayout(192, 8, 1, 3, 0).start, ExpertLayout(192, 8, 1, 3, 0).end) == (72, 96)
    # 32 experts, TP 2, EP 8
    assert [ExpertLayout(32, 2, 4, 1, o).start for o in range(4)] == [16, 20, 24, 28]


def test_bad_layouts_raise():
    for args in ((100, 8, 1, 0, 0), (192, 8, 5, 0, 0), (8, 2, 1, 2, 0), (8, 2, 2, 0, 2), (8, 2, 1, -1, 0)):
        try:
            ExpertLayout(*args)
        except AssertionError:
            continue
        raise AssertionError(f"ExpertLayout{args} should be rejected")


def test_without_process_groups_one_gpu_owns_every_expert():
    layout = ExpertLayout.from_process_groups(16)
    assert layout == ExpertLayout(16, 1, 1, 0, 0)
    assert (layout.n_local, layout.start, layout.end) == (16, 0, 16)


if __name__ == "__main__":
    for test_name, test in list(globals().items()):
        if test_name.startswith("test_"):
            test()
            print(f"{test_name} ok")
