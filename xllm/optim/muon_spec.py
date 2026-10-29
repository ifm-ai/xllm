"""Muon weight metadata, independent of the optimizer and distributed runtime.

Specs live on modules so parameter replacement by FSDP2 preserves registration.
The layout helpers describe geometry; supported training currently requires
TP=CP=1 and dense models. Expert and TP layouts are internal groundwork.
"""
import math
import re
from dataclasses import dataclass

from torch import nn

Box = tuple[int, int, int, int]  # rows [r0, r1) x cols [c0, c1) of a weight
Index = tuple[slice, slice]  # a box as an index into a tensor


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
    local: bool = False  # the owner stores all of it: no communication

    @property
    def shape(self) -> tuple[int, int]:
        return self.r1 - self.r0, self.c1 - self.c0

    @property
    def box(self) -> Box:
        return self.r0, self.r1, self.c0, self.c1

    @property
    def cost(self) -> int:
        """Newton-Schulz work on the block."""
        m, n = self.shape
        return m * n * min(m, n)

    def overlap(self, region: Box) -> Box | None:
        """The part of the block inside `region`, or None."""
        box = (max(self.r0, region[0]), min(self.r1, region[1]), max(self.c0, region[2]), min(self.c1, region[3]))
        return box if box[0] < box[1] and box[2] < box[3] else None


@dataclass
class MuonSpec:
    """How a weight maps to Newton-Schulz matrices (see register_muon)."""
    dims: tuple[tuple[str, ...], ...]  # axis names of tensor dims 0 and 1, outer first
    fan_out: tuple[str, ...]  # axes of the matrix rows, innermost in their dim
    fan_in: tuple[str, ...]  # axes of the matrix columns, innermost in their dim
    sizes: dict[str, int]  # sizes of the inner axes; each dim's outermost axis is inferred
    tp_dim: int | None = None  # dim split over the tensor-parallel group, at its outermost axis

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

    def fans(self, shape: tuple[int, int], per_block: bool = False) -> tuple[int, int]:
        """fan_out x fan_in of one Block, or of the whole weight (Block axes included)."""
        if per_block:
            size = self._sizes(shape)
            return math.prod(size[a] for a in self.fan_out), math.prod(size[a] for a in self.fan_in)
        d = 0 if self.fan_out[0] in self.dims[0] else 1
        return shape[d], shape[1 - d]

    def global_shape(self, shape: tuple[int, int], tp: int) -> tuple[int, int]:
        """The shape of the whole weight, from one tensor-parallel rank's shape."""
        return tuple(n * tp if d == self.tp_dim else n for d, n in enumerate(shape))

    def regions(self, shape: tuple[int, int], coords: list[tuple[int, int]], dp: int, tp: int) -> list[Box]:
        """The box of the whole weight (of this `shape`) that each rank (dp index, tp index) stores: TP gives
        rank t the t-th chunk along tp_dim (all of it if not split), FSDP2 Shard(0) cuts that into contiguous
        rows by torch.chunk (trailing ranks may get none)."""
        local = list(shape)
        if tp > 1 and self.tp_dim is not None:
            local[self.tp_dim] //= tp
        step = math.ceil(local[0] / dp)
        boxes = []
        for i, t in coords:
            r0, r1 = min(i * step, local[0]), min((i + 1) * step, local[0])
            row_off = t * local[0] if tp > 1 and self.tp_dim == 0 else 0
            col_off = t * local[1] if tp > 1 and self.tp_dim == 1 else 0
            boxes.append((row_off + r0, row_off + r1, col_off, col_off + local[1]))
        return boxes

    def _sizes(self, shape: tuple[int, int]) -> dict[str, int]:
        """Every axis size, each dim's outermost axis inferred from the shape."""
        size = dict(self.sizes)
        for d, dim in enumerate(self.dims):
            inner = math.prod(self.sizes[a] for a in dim[1:])
            assert shape[d] % inner == 0, f"dim {d} ({shape[d]}) is not a multiple of {[self.sizes[a] for a in dim[1:]]}"
            size[dim[0]] = shape[d] // inner
        return size

    def blocks(self, shape: tuple[int, int], param: int = -1) -> list[Block]:
        """The Blocks of a weight of this shape (`param` in the optimizer), row-major over the Block axes."""
        size = self._sizes(shape)
        fans = self.fan_out + self.fan_in
        ranges = []
        for dim in self.dims:
            span = math.prod(size[a] for a in dim if a in fans)
            starts = [0]
            for a in (a for a in dim if a not in fans):
                starts = [s * size[a] + i for s in starts for i in range(size[a])]
            ranges.append([(s * span, (s + 1) * span) for s in starts])
        return [Block(r0, r1, c0, c1, param) for r0, r1 in ranges[0] for c0, c1 in ranges[1]]


def _parse_layout(layout: str) -> tuple[tuple[str, ...], ...]:
    """"(h d) m" -> (("h", "d"), ("m",))"""
    groups = re.findall(r"\(([^()]*)\)|([^\s()]+)", layout)
    assert layout.count("(") == layout.count(")") == sum(bool(g) for g, _ in groups), "unbalanced parentheses"
    return tuple(tuple((group or axis).split()) for group, axis in groups)


def muon_spec(layout: str = "o i", *, fan_out: str = "o", fan_in: str = "i", tp_dim: int | None = None,
              **sizes: int) -> MuonSpec:
    """A MuonSpec from a layout (see register_muon), e.g. muon_spec("(h d) m", fan_out="d", fan_in="m", d=64)."""
    return MuonSpec(_parse_layout(layout), tuple(fan_out.split()), tuple(fan_in.split()), sizes, tp_dim)


_UNSET = object()


def register_muon(module: nn.Module, name: str = "weight", layout: str = "o i", *,
                  fan_out: str = "o", fan_in: str = "i", tp_dim=_UNSET, **sizes: int):
    """Mark module.<name> as a Muon weight; call it in the module's __init__.

    layout: the axes of the weight's two dims, named as the forward rearranges them (parentheses
        flatten several axes into one dim, outer first). Default: an [out, in] weight, one matrix.
    fan_out / fan_in: the axes of the Newton-Schulz matrix rows / columns, innermost in their dim;
        every other axis indexes separate Blocks.
    sizes: the sizes of the inner axes; each dim's outermost axis is inferred from the shape.
    tp_dim: the dim of the weight that tensor parallelism splits (0 or 1; None: not split). Required when
        the tensor-parallel size is above 1: the weight is then this rank's chunk along tp_dim, and the
        whole weight is tp times larger there.
    """
    import xllm.distributed as xd
    param = getattr(module, name)
    where = f"{type(module).__name__}.{name} {list(param.shape)} {layout!r}"
    assert param.dim() == 2, f"{where}: Muon needs a 2-D parameter"
    tp = xd.get_model_parallel_world_size()
    if tp_dim is _UNSET:
        assert tp == 1, f"{where}: with tensor parallelism, say which dim it splits: tp_dim=0, 1 or None"
        tp_dim = None
    assert tp_dim in (None, 0, 1), f"{where}: tp_dim must be 0, 1 or None, not {tp_dim!r}"
    try:
        spec = muon_spec(layout, fan_out=fan_out, fan_in=fan_in, tp_dim=tp_dim, **sizes)
        spec.blocks(spec.global_shape(tuple(param.shape), tp))  # check the layout on the whole weight
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from None
    vars(module).setdefault("muon_specs", {})[name] = spec


def box_index(box: Box, origin: Box) -> Index:
    """`box` as an index into the tensor whose top-left corner is `origin`'s."""
    return slice(box[0] - origin[0], box[1] - origin[0]), slice(box[2] - origin[2], box[3] - origin[2])


def name_pattern(fqn: str) -> str:
    """layers.3.attention.wq.weight -> layers.*.attention.wq.weight"""
    return re.sub(r"(?<![^.])\d+(?=\.|$)", "*", fqn)


def split_muon_params(model: nn.Module):
    """Registered weights go to Muon, everything else to AdamW. Call it after wrapping: the weights are looked
    up by name. Returns the Muon params, their specs and name patterns, and the AdamW params."""
    registered = {id(getattr(m, name)): (spec, name_pattern(f"{fqn}.{name}".lstrip(".")))
                  for fqn, m in model.named_modules() for name, spec in vars(m).get("muon_specs", {}).items()}
    muon_params = [p for p in model.parameters() if id(p) in registered]
    adamw_params = [p for p in model.parameters() if id(p) not in registered]
    specs = [registered[id(p)][0] for p in muon_params]
    names = [registered[id(p)][1] for p in muon_params]
    return muon_params, specs, names, adamw_params
