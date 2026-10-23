"""register_muon and the layout parser of xllm/optim/muon.py, on CPU.

    PYTHONPATH=. pytest -q tests/test_muon_register.py

A weight is built from a stack of Block matrices with einops, so Block j must read back as
matrix j, and the per-Block fans must be matrix j's [fan_out, fan_in]; together the Blocks cover the
weight exactly once. No case has a square Block, so its orientation is unambiguous.
"""
import math

import pytest
import torch
from einops import rearrange
from torch import nn

from xllm.optim.muon_spec import muon_spec, register_muon


def boxes(blocks):
    return [(b.r0, b.r1, b.c0, b.c1) for b in blocks]


class Holder(nn.Module):
    def __init__(self, shape, tp_dim=None, declare=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))
        if declare:
            self.tp_dims = {"weight": tp_dim}


# layout, fan_out, fan_in, sizes given to register_muon, all axis sizes (for einops)
CASES = [
    ("(h d) m", "d", "m", {"d": 4}, {"h": 3, "d": 4, "m": 6}),
    ("(h k d) m", "k d", "m", {"k": 2, "d": 4}, {"h": 3, "k": 2, "d": 4, "m": 6}),  # softdelta q: 2 queries per head
    ("(h k d) m", "d", "m", {"k": 2, "d": 4}, {"h": 3, "k": 2, "d": 4, "m": 6}),    # one Block per query
    ("(h d) m", "h d", "m", {"d": 4}, {"h": 3, "d": 4, "m": 6}),                    # all heads as one matrix
    ("o (h i)", "o", "i", {"i": 4}, {"h": 3, "o": 6, "i": 4}),                       # wo per head
    ("(n s) d", "s", "d", {"s": 5}, {"n": 4, "s": 5, "d": 6}),                       # experts
    ("(g h d) m", "d", "m", {"h": 2, "d": 4}, {"g": 3, "h": 2, "d": 4, "m": 6}),    # two Block axes
    ("o i", "o", "i", {}, {"o": 6, "i": 4}),
    ("(h d) (g m)", "d", "m", {"d": 4, "m": 3}, {"h": 3, "d": 4, "g": 2, "m": 3}),  # Block axes in both dims
    ("(n i) o", "o", "i", {"i": 5}, {"n": 4, "i": 5, "o": 6}),                       # stored [in, out]
]


@pytest.mark.parametrize("layout, fan_out, fan_in, sizes, all_sizes", CASES)
def test_blocks_match_einops(layout, fan_out, fan_in, sizes, all_sizes):
    fans = fan_out.split() + fan_in.split()
    block_axes = [a for a in layout.replace("(", " ").replace(")", " ").split() if a not in fans]
    group = lambda axes: f"({' '.join(axes)})"
    n = math.prod(all_sizes[a] for a in block_axes)
    mats = torch.randn(n, math.prod(all_sizes[a] for a in fan_out.split()), math.prod(all_sizes[a] for a in fan_in.split()))
    w = rearrange(mats, f"{group(block_axes)} {group(fan_out.split())} {group(fan_in.split())} -> {layout}", **all_sizes)
    m = Holder(w.shape)
    register_muon(m, "weight", layout, fan_out=fan_out, fan_in=fan_in, **sizes)
    spec = m.muon_specs["weight"]
    assert spec == muon_spec(layout, fan_out=fan_out, fan_in=fan_in, **sizes)
    blocks = spec.blocks(tuple(w.shape))
    assert len(blocks) == n and spec.fans(tuple(w.shape), per_block=True) == tuple(mats[0].shape)
    cover = torch.zeros(w.shape, dtype=torch.int32)
    for j, b in enumerate(blocks):
        block = w[b.r0:b.r1, b.c0:b.c1]
        assert torch.equal(block if block.shape == mats[j].shape else block.mT, mats[j]), (layout, j)
        cover[b.r0:b.r1, b.c0:b.c1] += 1
    assert bool((cover == 1).all())


@pytest.mark.parametrize("layout, fan_out, fan_in, sizes, shape, fans", [
    ("(h d) m", "d", "m", {"d": 4}, (12, 6), (12, 6)),
    ("o (h i)", "o", "i", {"i": 4}, (6, 12), (6, 12)),
    ("(n i) o", "o", "i", {"i": 5}, (20, 6), (6, 20)),   # stored [in, out]
])
def test_whole_weight_fans(layout, fan_out, fan_in, sizes, shape, fans):
    assert muon_spec(layout, fan_out=fan_out, fan_in=fan_in, **sizes).fans(shape) == fans


def test_linear_default_is_one_matrix():
    lin = nn.Linear(6, 4, bias=False)
    lin.tp_dims = {"weight": None}
    register_muon(lin)
    spec = lin.muon_specs["weight"]
    assert str(spec) == "o i" and spec.tp_dim is None
    assert boxes(spec.blocks((4, 6))) == [(0, 4, 0, 6)] and spec.fans((4, 6), per_block=True) == (4, 6)


def test_tensor_parallel_global_shape():
    """Registered on a TP-local shard, resolved to the global weight: the outer axis is counted globally."""
    m = Holder((2 * 4, 6), tp_dim=0)                            # 2 local heads of 4 rows
    register_muon(m, "weight", "(h d) m", fan_out="d", fan_in="m", d=4)
    spec = m.muon_specs["weight"]
    assert spec.global_shape((8, 6), 2) == (16, 6) and boxes(spec.blocks((16, 6))) == [(4 * j, 4 * j + 4, 0, 6) for j in range(4)]
    m = Holder((6, 2 * 4), tp_dim=1)                            # per-head columns, TP cuts the columns
    register_muon(m, "weight", "o (h i)", fan_out="o", fan_in="i", i=4)
    spec = m.muon_specs["weight"]
    assert spec.global_shape((6, 8), 2) == (6, 16) and boxes(spec.blocks((6, 16))) == [(0, 6, 4 * j, 4 * j + 4) for j in range(4)]


@pytest.mark.parametrize("make, match", [
    (lambda m: register_muon(m, "weight", "(h d) m", fan_out="d", fan_in="m", h=2, d=4), "outer axis"),
    (lambda m: register_muon(m, "weight", "(h k d) m", fan_out="d", fan_in="m", d=4), "sizes of"),
    (lambda m: register_muon(m, "weight", "(d h) m", fan_out="d", fan_in="m", h=2), "innermost"),
    (lambda m: register_muon(m, "weight", "(h d m)", fan_out="d", fan_in="m", d=4, m=6), "names 1 dims"),
    (lambda m: register_muon(m, "weight", "(h d) m", fan_out="d", fan_in="m", d=4, x=3), "unknown axes"),
    (lambda m: register_muon(m, "weight", "(h d) d", fan_out="d", fan_in="d"), "repeated"),
    (lambda m: register_muon(m, "weight", "(h d) m", fan_out="d", fan_in="m", d=3), "not a multiple"),
    (lambda m: register_muon(Holder((8, 6), declare=False), "weight", "(h d) m", fan_out="d", fan_in="m", d=4),
     "must declare tp_dims"),
    (lambda m: register_muon(m, "weight", "(h d) m", d=4), "innermost"),           # fan_out / fan_in left at "o" / "i"
    (lambda m: register_muon(m, "weight", "(h d m", fan_out="d", fan_in="m", d=4), "unbalanced"),
    (lambda m: register_muon(m, "weight", "(h d) m", fan_out="h", fan_in="m", d=4), "innermost"),  # h is not innermost
])
def test_bad_registrations_are_rejected(make, match):
    with pytest.raises(AssertionError, match=match):
        make(Holder((8, 6)))


def test_only_2d_parameters():
    with pytest.raises(AssertionError, match="2-D"):
        register_muon(Holder((2, 4, 6)), "weight", "e o i", fan_out="o", fan_in="i")
