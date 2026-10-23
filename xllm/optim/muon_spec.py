"""Muon weight metadata, independent of the optimizer and distributed runtime.

Specs live on modules so parameter replacement by FSDP2 preserves registration.
The layout helpers describe geometry; supported training currently requires
TP=CP=1 and dense models. Expert and TP layouts are internal groundwork.
"""
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch import nn


@dataclass
class Block:
    """Box [r0, r1) x [c0, c1) of a Muon weight that Newton-Schulz orthogonalizes on its own (one or
    more heads, one expert, or the whole weight)."""
    r0: int
    r1: int
    c0: int
    c1: int
    param: int = -1  # index of its weight in the optimizer
    owner: int = 0  # rank that gathers the block and runs Newton-Schulz on it

    @property
    def shape(self) -> Tuple[int, int]:
        return self.r1 - self.r0, self.c1 - self.c0


@dataclass
class MuonSpec:
    """How a weight maps to Newton-Schulz matrices (see register_muon)."""
    dims: Tuple[Tuple[str, ...], ...]  # axis names of tensor dims 0 and 1, outer first
    fan_out: Tuple[str, ...]  # axes of the matrix rows, innermost in their dim
    fan_in: Tuple[str, ...]  # axes of the matrix columns, innermost in their dim
    sizes: Dict[str, int]  # sizes of the inner axes; each dim's outermost axis is inferred
    tp_dim: Optional[int] = None  # dim split over the tensor-parallel group, at its outermost axis

    def __post_init__(self):
        axes = [a for dim in self.dims for a in dim]
        assert len(self.dims) == 2 and all(self.dims), f"names {len(self.dims)} dims, needs 2"
        assert len(set(axes)) == len(axes), "repeated axis name"
        fan_dims = [d for fan in (self.fan_out, self.fan_in) for d, dim in enumerate(self.dims)
                    if dim[len(dim) - len(fan):] == fan]
        assert sorted(fan_dims) == [0, 1], "fan_out and fan_in must be the innermost axes of different dims"
        assert set(self.sizes) <= set(axes), f"sizes of unknown axes {sorted(set(self.sizes) - set(axes))}"
        for d, dim in enumerate(self.dims):
            assert dim[0] not in self.sizes, f"{dim[0]!r} is the outer axis of dim {d} and is inferred; give inner sizes"
            assert set(dim[1:]) <= set(self.sizes), f"give the sizes of {[a for a in dim[1:] if a not in self.sizes]}"

    def __str__(self) -> str:
        return " ".join(f"({' '.join(dim)})" if len(dim) > 1 else dim[0] for dim in self.dims)

    def fans(self, shape: Tuple[int, int], per_block: bool = False) -> Tuple[int, int]:
        """fan_out x fan_in of one Block, or of the whole weight (Block axes included)."""
        if per_block:
            size = self._sizes(shape)
            return math.prod(size[a] for a in self.fan_out), math.prod(size[a] for a in self.fan_in)
        d = 0 if self.fan_out[0] in self.dims[0] else 1
        return shape[d], shape[1 - d]

    @property
    def tp_cuts_matrix(self) -> bool:
        """True when tensor parallelism cuts the Newton-Schulz matrix itself: the outermost axis of the
        split dim is a fan axis (wo, FFN), so the Blocks need the TP group to be assembled. False when
        it splits a Block axis (whole heads or experts per rank) or nothing, so every Block is local."""
        return self.tp_dim is not None and self.dims[self.tp_dim][0] in self.fan_out + self.fan_in

    def global_shape(self, shape: Tuple[int, int], tp: int) -> Tuple[int, int]:
        """The shape of the whole weight, from one tensor-parallel rank's shape."""
        return tuple(n * tp if d == self.tp_dim else n for d, n in enumerate(shape))

    def _sizes(self, shape: Tuple[int, int]) -> Dict[str, int]:
        """Every axis size, each dim's outermost axis inferred from the shape."""
        size = dict(self.sizes)
        for d, dim in enumerate(self.dims):
            inner = math.prod(self.sizes[a] for a in dim[1:])
            assert shape[d] % inner == 0, f"dim {d} ({shape[d]}) is not a multiple of {[self.sizes[a] for a in dim[1:]]}"
            size[dim[0]] = shape[d] // inner
        return size

    def blocks(self, shape: Tuple[int, int]) -> List[Block]:
        """The Blocks of a weight of this shape, row-major over the Block axes."""
        size = self._sizes(shape)
        fans = self.fan_out + self.fan_in
        ranges = []
        for dim in self.dims:
            span = math.prod(size[a] for a in dim if a in fans)
            starts = [0]
            for a in (a for a in dim if a not in fans):
                starts = [s * size[a] + i for s in starts for i in range(size[a])]
            ranges.append([(s * span, (s + 1) * span) for s in starts])
        return [Block(r0, r1, c0, c1) for r0, r1 in ranges[0] for c0, c1 in ranges[1]]


def _parse_layout(layout: str) -> Tuple[Tuple[str, ...], ...]:
    """"(h d) m" -> (("h", "d"), ("m",))"""
    groups = re.findall(r"\(([^()]*)\)|([^\s()]+)", layout)
    assert layout.count("(") == layout.count(")") == sum(bool(g) for g, _ in groups), "unbalanced parentheses"
    return tuple(tuple((group or axis).split()) for group, axis in groups)


def muon_spec(layout: str = "o i", *, fan_out: str = "o", fan_in: str = "i", tp_dim: Optional[int] = None,
              **sizes: int) -> MuonSpec:
    """A MuonSpec from a layout (see register_muon), e.g. muon_spec("(h d) m", fan_out="d", fan_in="m", d=64)."""
    return MuonSpec(_parse_layout(layout), tuple(fan_out.split()), tuple(fan_in.split()), sizes, tp_dim)


def register_muon(module: nn.Module, name: str = "weight", layout: str = "o i", *,
                  fan_out: str = "o", fan_in: str = "i", **sizes: int):
    """Mark module.<name> as a Muon weight; call it in the module's __init__.

    layout: the axes of the weight's two dims, named as the forward rearranges them (parentheses
        flatten several axes into one dim, outer first). Default: an [out, in] weight, one matrix.
    fan_out / fan_in: the axes of the Newton-Schulz matrix rows / columns, innermost in their dim;
        every other axis indexes separate Blocks.
    sizes: the sizes of the inner axes; each dim's outermost axis is inferred from the shape.

    The module declares the dim tensor parallelism splits: module.tp_dims[name] (None: not split).
    """
    param = getattr(module, name)
    where = f"{type(module).__name__}.{name} {list(param.shape)} {layout!r}"
    assert param.dim() == 2, f"{where}: Muon needs a 2-D parameter"
    assert name in getattr(module, "tp_dims", {}), f"{where}: the module must declare tp_dims[{name!r}] (None if not split)"
    try:
        spec = muon_spec(layout, fan_out=fan_out, fan_in=fan_in, tp_dim=module.tp_dims[name], **sizes)
        spec.blocks(tuple(param.shape))
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from None
    vars(module).setdefault("muon_specs", {})[name] = spec

