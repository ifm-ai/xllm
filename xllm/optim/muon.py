"""Muon on FSDP2 with per-head orthogonalization.

Modules register their Muon weights in __init__; everything else goes to AdamW:

    register_muon(self.wo)                                                                 # one matrix
    register_muon(self.wk, layout="(h d) m", fan_out="d", fan_in="m", d=self.head_dim)    # one block per head
    register_muon(self.wk, layout="(h d) m", fan_out="h d", fan_in="m", d=self.head_dim)  # all heads, one matrix
    register_muon(self, "weight1", "(n s) d", fan_out="s", fan_in="d", s=self.inter_dim)   # one block per expert

The spec sits on the module, not on the weight tensor: FSDP2 and Module._apply (.cuda(), .to())
replace or swap DTensor parameters and drop attributes set on them.

Terminology, for one Muon matrix W of shape [rows, cols]:

- Block: a box W[r0:r1, c0:c1] that Newton-Schulz orthogonalizes on its own; the Blocks of
  a matrix tile it exactly once. Chosen from the model, not from the sharding (the whole
  matrix, one or more heads of wq / wk / wv / wr, one expert).
- region: the box of W that one rank stores. FSDP2 Shard(0) stores contiguous row ranges
  with all columns, one per rank (torch.chunk).
- owner: the rank that gathers a Block and runs NS on it: the rank that stores most of
  it, then the least loaded one.
- slice: the overlap of a Block with one rank's region, a box. A Block is assembled from
  its slices.
- local / remote Block: the local Blocks of a matrix on a rank lie within its region, span its
  columns and follow each other with one height (experts, heads inside a shard): the rank
  owns them and orthogonalizes them in place, as one [M, R, cols] view of its shard, without
  communication. Every other Block is remote: gathered to its owner and scattered back.
- local / remote matrix: on a rank, a matrix whose shard is exactly its local Blocks is local
  (routed experts, heads); the others (with remote Blocks, or none of it stored there) are remote.
- batch: local matrices whose Blocks have one shape, stepped together: their updates are views of
  one bf16 [B, R, C] buffer, so Newton-Schulz runs once on it in place. Batches step one after
  another, so the step's temporary buffers are the size of a batch.
- plan (MatrixPlan): one matrix's Blocks compiled for one rank: its local Blocks as a view,
  the slices of its remote Blocks it sends and, for those it owns, receives.

Example: wk [8, 4] in 2 row Blocks of 4 rows on 3 ranks (torch.chunk: rows 0-2, 3-5, 6-7):

    rank 0  rows 0-2  a          block 0 = rows 0-3 = cat(a, b1), owner rank 0 (stores 3 rows)
    rank 1  rows 3-5  b1 | b2    block 1 = rows 4-7 = cat(b2, c), owner rank 1 (tie with rank 2)
    rank 2  rows 6-7  c          b1 and b2 are rank 1's slices of blocks 0 and 1
"""
import math
from collections import Counter, defaultdict
from itertools import accumulate
from logging import getLogger
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import xllm_extension.ops as xllm_ops
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.adamw import adamw

from xllm.config import OptimConf
from .muon_spec import Block, MuonSpec


logger = getLogger()

# Newton-Schulz steps X <- a X + b (X X^T) X + c (X X^T)^2 X: each applies p(s) = a s + b s^3 + c s^5 to the
# singular values. Polar Express (arXiv 2505.16932): each p is the minimax fit of 1 on the range the previous
# steps leave, starting from [1e-3, 1.01]; fitted above a cushion of 0.05 of the range's top, so no p dips
# near 0 inside it (rounding there would flip a sign), and 1% above its top. Remez in quad precision. In
# exact arithmetic the 7 steps take every s in [1e-3, 1] to 1 +- 3e-7.
NS_COEFFS = (
    (6.6855200767111675, -17.811269767700782, 12.619541303795451),
    (4.016121039752612, -3.861087728177999, 0.9871918227003456),
    (4.016121039752612, -3.861087728177999, 0.9871918227003456),
    (3.795262478265412, -3.5559258715900817, 0.9017288960572022),
    (2.3593913208894888, -1.696737173296388, 0.4096643411881119),
    (1.8910694529512306, -1.2577697543810034, 0.3672643550538599),
    (1.8657070833564235, -1.23145450650965, 0.36574763500927243),
)

Box = Tuple[int, int, int, int]  # rows [r0, r1) x cols [c0, c1) of a matrix
Index = Tuple[slice, slice]  # a box as an index into a tensor


@dataclass
class LocalBlocks:
    """A rank's local blocks of one matrix: `count` blocks of `height` rows and `cols` columns, rows
    row0 : row0 + count * height of its shard."""
    row0: int
    count: int
    height: int
    cols: int


@dataclass
class MatrixPlan:
    """How one Muon matrix steps on this rank. The slice lists follow the block order, so a
    sender's list matches its receiver's."""
    send: List[List[Index]]  # [r]: my slices of the remote blocks rank r owns, as boxes of my shard
    recv: List[List[Tuple[int, Index]]]  # [r]: rank r's slices of the remote blocks I own, as (index in owned, box of the block)
    owned: List[int] = field(default_factory=list)  # the remote blocks I own (indices into Muon.blocks)
    local: Optional[LocalBlocks] = None  # my local blocks of the matrix, if it has any here


def split_muon_params(model: nn.Module):
    """Registered weights go to Muon, everything else to AdamW. Call it after wrapping: the weights
    are looked up by name. Returns the Muon params, their specs and names, and the AdamW params."""
    registered = {id(getattr(m, name)): (spec, _pattern(f"{fqn}.{name}".lstrip(".")))
                  for fqn, m in model.named_modules() for name, spec in vars(m).get("muon_specs", {}).items()}
    muon_params = [p for p in model.parameters() if id(p) in registered]
    adamw_params = [p for p in model.parameters() if id(p) not in registered]
    specs = [registered[id(p)][0] for p in muon_params]
    names = [registered[id(p)][1] for p in muon_params]
    return muon_params, specs, names, adamw_params


def _local(t: Tensor) -> Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _overlap(a: Box, b: Box):
    """The box both a and b cover, or None."""
    box = (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))
    return box if box[0] < box[1] and box[2] < box[3] else None


def _local_box(box: Box, origin: Box) -> Tuple[slice, slice]:
    """`box` as an index into the tensor whose top-left corner is `origin`'s."""
    return slice(box[0] - origin[0], box[1] - origin[0]), slice(box[2] - origin[2], box[3] - origin[2])


def _shape(index: Tuple[slice, slice]) -> Tuple[int, int]:
    rows, cols = index
    return rows.stop - rows.start, cols.stop - cols.start


def _split(sizes: List[int], limit: int) -> List[Tuple[int, int]]:
    """Cut range(len(sizes)) into consecutive [lo, hi), each closed once its sizes add up to `limit`
    (the last one may fall short); a single range if limit is 0."""
    bounds, lo, total = [], 0, 0
    for i, size in enumerate(sizes):
        total += size
        if limit and total >= limit:
            bounds.append((lo, i + 1))
            lo, total = i + 1, 0
    if lo < len(sizes) or not bounds:
        bounds.append((lo, len(sizes)))
    return bounds


def newton_schulz(X: Tensor, eps: float) -> Tensor:
    """U S V^T -> U V^T for a stack of matrices [B, m, n], in fp16, in X's layout.

    - The steps run on the smaller Gram matrix: A = X X^T and X <- a X + B X, or for a tall X, A = X^T X and
      X <- a X + X B (B = b A + c A^2 is symmetric), so a tall X is never transposed into a copy.
    - X is scaled by s = ||A^2||_F^(1/4) >= s_max, 5% more for rounding: within k^(1/8) of s_max where ||X||_F
      is within k^(1/2) (k = min(m, n)), so the small singular values start higher. s comes from the first
      step's A^2 and folds into that step: p(X / s) = (a X + (b / s^2 A + c / s^4 A^2) X) / s.
    - The first step runs in bf16, whose range holds any X; the others in fp16, whose 3 more mantissa bits
      bring the singular values to 1 within ~1e-3. The sums run in cuBLAS epilogues (baddbmm's alpha / beta),
      accumulated in fp32."""
    tall = X.shape[-2] > X.shape[-1]

    def gram(X):
        return X.mT @ X if tall else X @ X.mT

    def step(X, B, a):  # a X + X B, or a X + B X
        return torch.baddbmm(X, X, B, beta=a) if tall else torch.baddbmm(X, B, X, beta=a)

    X = X.to(torch.bfloat16)
    A = gram(X)
    AA = A @ A
    s = AA.float().norm(dim=(-2, -1), keepdim=True).pow(0.25) * 1.05 + eps
    (a, b, c), *rest = NS_COEFFS
    B = (AA.float() * (c / s**4) + A.float() * (b / s**2)).bfloat16()
    X = torch.div(step(X, B, a), s, out=torch.empty_like(X, dtype=torch.half))
    for a, b, c in rest:
        A = gram(X)
        X = step(X, torch.baddbmm(A, A, A, beta=b, alpha=c), a)  # B = b A + c A^2
    return X


def orthogonalize_(X: Tensor, eps: float):
    """Newton-Schulz in place on a stack of matrices [M, R, C]."""
    X.copy_(newton_schulz(X, eps))


def _rows_as_blocks(x: Tensor, local: LocalBlocks) -> Tensor:
    """The local blocks of x [rows, cols] as [count, height, cols], a view."""
    return x[local.row0:local.row0 + local.count * local.height].view(local.count, local.height, x.shape[1])


def _copy_(dst: List[List[Tensor]], src: List[List[Tensor]]):
    """dst[r][i].copy_(src[r][i]) for every rank r and i, in one launch."""
    xllm_ops.batched_copy_([t for ts in dst for t in ts], [t for ts in src for t in ts])


def _views(buf: Tensor, shapes: List[List[Tuple[int, int]]]) -> List[List[Tensor]]:
    """buf cut into one view of each shape, rank after rank."""
    parts = iter(buf.split([m * n for ss in shapes for m, n in ss]))
    return [[next(parts).view(shape) for shape in ss] for ss in shapes]


def _packed(blocks: List[Tuple[int, int, int]]) -> Tuple[List[Tensor], List[Tensor]]:
    """One bf16 [B, R, C] buffer per shape of block, for blocks given as (count, rows, cols). Returns each
    entry's blocks as a [count, rows, cols] view of its buffer, and the buffers: Newton-Schulz runs on a
    buffer in place."""
    sizes = Counter()
    for m, r, c in blocks:
        sizes[r, c] += m
    buffers = {s: torch.empty(n, *s, dtype=torch.bfloat16, device="cuda") for s, n in sizes.items()}
    taken = Counter()
    views = []
    for m, r, c in blocks:
        views.append(buffers[r, c][taken[r, c]:taken[r, c] + m])
        taken[r, c] += m
    return views, list(buffers.values())


def group_hparams(cfg: OptimConf, algorithm: str) -> Dict:
    """Param-group hyperparameters that OptimConf sets for "muon" or "adamw" (single source of truth)."""
    if algorithm == "muon":
        return dict(momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov, adjust_lr=cfg.muon_adjust_lr,
                    block_scale=cfg.muon_block_scale,
                    weight_decay=cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay)
    return dict(betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)


class Muon(Optimizer):
    """Muon for FSDP2 parameters sharded on dim 0, with AdamW for the rest.

    Each step:
    1. remote matrices: update the momentum on the local shard; gather: an asynchronous all-to-all sends
       every slice of a remote block to the block's owner;
    2. meanwhile the local matrices, one batch after another (cfg.muon_batch_numel): momentum, Newton-Schulz
       in place on the batch, update; no communication;
    3. remote matrices: Newton-Schulz on their local blocks, then on the remote blocks I own; scatter: an
       all-to-all sends the slices back; each rank applies its update.
    Owners are balanced over all the blocks, so the step waits once for the slowest rank, at the scatter.
    FSDP2 already balances the local blocks: every rank stores as many experts / heads.

    With tensor parallelism each rank holds its TP-local shard of a weight, and FSDP2 shards that over
    its dp group. TP is just one more way the weight is sharded: Blocks are defined on the whole weight,
    each rank's region is a box of it, and the same two all-to-alls run over the dp x TP plane. A Block
    may sit inside one TP rank (whole heads) or span several, and is orthogonalized once, by its owner.
    """

    def __init__(
        self,
        muon_params: List[nn.Parameter],
        specs: List[MuonSpec],
        adamw_params: List[nn.Parameter],
        mesh: DeviceMesh,
        cfg: OptimConf,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        """specs[i]: how muon_params[i] maps to Newton-Schulz matrices (register_muon / muon_spec).
        tp_group: the tensor-parallel group, if any: the params are then each rank's TP-local shard. Muon
        works over the plane of the dp mesh x TP group, TP being one more way the weight is sharded."""
        defaults = dict(eps=cfg.epsilon)
        super().__init__(
            [
                dict(params=muon_params, lr=cfg.muon_lr, algorithm="muon", **group_hparams(cfg, "muon")),
                dict(params=adamw_params, lr=cfg.lr, algorithm="adamw", **group_hparams(cfg, "adamw")),
            ],
            defaults,
        )
        self._plan([tuple(p.shape) for p in muon_params], specs, mesh, tp_group, cfg.muon_batch_numel)

    # ---- plan: built once from the global shapes, identical on every rank ----

    def _plan(self, shapes: List[Tuple[int, int]], specs: List[MuonSpec], mesh: DeviceMesh,
              tp_group: Optional[dist.ProcessGroup], batch_numel: int):
        # Ranks: the plane of the dp mesh x TP group that shards a weight. self.coords[r] is the
        # (dp index, tp index) of its rank r, in the process group's order.
        self.tp = 1 if tp_group is None else dist.get_world_size(tp_group)
        self.dp = mesh.size()
        if tp_group is None:
            self.process_group, self.rank = mesh.get_group(), mesh.get_local_rank()
            self.coords = [(r, 0) for r in range(self.dp)]
        else:
            mine = (dist.get_process_group_ranks(mesh.get_group()), dist.get_process_group_ranks(tp_group))
            every = [None] * dist.get_world_size()
            dist.all_gather_object(every, mine)
            plane = sorted(g for d in mine[0] for g in every[d][1])
            planes = [None] * len(every)
            dist.all_gather_object(planes, plane)
            for ranks in sorted({tuple(q) for q in planes}):  # every rank creates every group, in the same order
                group = dist.new_group(list(ranks))
                if dist.get_rank() in ranks:
                    self.process_group = group
            self.rank = plane.index(dist.get_rank())
            self.coords = [(every[g][0].index(g), every[g][1].index(g)) for g in plane]
        self.world = len(self.coords)

        # Geometry is fixed; scale policy comes from the current param group,
        # including after a checkpoint restore or an explicit config override.
        # `shapes` are this rank's TP-local shapes; Blocks, scales and regions are over the whole weight
        shapes = [spec.global_shape(shape, self.tp) for spec, shape in zip(specs, shapes)]
        self._fans = {
            mode: [spec.fans(shape, per_block=mode == "block") for spec, shape in zip(specs, shapes)]
            for mode in ("block", "matrix")
        }
        self.blocks: List[Block] = []
        for i, (spec, shape) in enumerate(zip(specs, shapes)):
            for b in spec.blocks(shape):
                b.param = i
                self.blocks.append(b)

        # regions[i][r]: the box of matrix i that rank r stores
        regions = [self._regions(spec, shape) for spec, shape in zip(specs, shapes)]
        # slices[k]: (rank, box in global coordinates) for every rank that stores part of block k
        slices = []
        for b in self.blocks:
            block = (b.r0, b.r1, b.c0, b.c1)
            overlaps = [(r, _overlap(block, region)) for r, region in enumerate(regions[b.param])]
            slices.append([(r, box) for r, box in overlaps if box is not None])

        self._assign_owners(slices)
        local, mine = self._local_blocks(slices, regions)
        # plans[i]: matrix i on this rank
        self.plans = [MatrixPlan([[] for _ in range(self.world)], [[] for _ in range(self.world)], local=mine.get(i))
                      for i in range(len(shapes))]
        for k in range(len(self.blocks)):
            if k not in local:
                self._plan_block(k, slices[k], regions)
        # has_remote: some rank has a remote block, so every rank joins the all-to-alls
        self.has_remote = len(local) < len(self.blocks)
        # remote: the matrices this rank steps around the all-to-alls (it sends or owns a remote block, or stores
        # none of the matrix); batches: the others, whose shard is exactly their local blocks
        self.remote = [i for i, plan in enumerate(self.plans) if plan.local is None or any(plan.send)]
        # a rank that owns a remote block stores part of it, so it has something to send (to itself at least)
        assert all(plan.local is None or any(plan.send) or not plan.owned for plan in self.plans)
        self.batches = self._batches(batch_numel)
        # owned_slices[r]: rank r's slices of the remote blocks I own, as (index among the owned blocks of
        # the remote matrices, box of the block)
        first = list(accumulate((len(self.plans[i].owned) for i in self.remote), initial=0))
        self.owned_slices = [[(j0 + j, box) for j0, i in zip(first, self.remote) for j, box in self.plans[i].recv[r]]
                             for r in range(self.world)]

    def _assign_owners(self, slices: List[List[Tuple[int, Box]]]):
        """owner: the rank that stores most of the block (no transfer for that part), then the least loaded;
        blocks are placed by decreasing Newton-Schulz cost m * n * min(m, n), all of them together: the step
        waits for the most loaded rank once, at the scatter."""
        load = [0] * self.world
        for k in sorted(range(len(self.blocks)), key=lambda k: -self._ns_cost(self.blocks[k])):
            stored = dict.fromkeys(range(self.world), 0)
            for r, (r0, r1, c0, c1) in slices[k]:
                stored[r] = (r1 - r0) * (c1 - c0)
            b = self.blocks[k]
            b.owner = min(range(self.world), key=lambda r: (-stored[r], load[r]))
            load[b.owner] += self._ns_cost(b)

    def _local_blocks(self, slices: List[List[Tuple[int, Box]]], regions: List[List[Box]]) -> Tuple[set, Dict[int, LocalBlocks]]:
        """The blocks that every rank orthogonalizes in place: for each matrix and rank, the blocks within
        the rank's region and across its columns, if they follow each other with one height (one view).
        Returns them (indices into self.blocks) and, per matrix, this rank's."""
        runs = defaultdict(list)
        for k in range(len(self.blocks)):
            b, r = self.blocks[k], slices[k][0][0]
            if len(slices[k]) == 1 and (b.c0, b.c1) == regions[b.param][r][2:]:
                runs[b.param, r].append(k)
        local, mine = set(), {}
        for (i, r), run in runs.items():
            blocks = sorted((self.blocks[k] for k in run), key=lambda b: b.r0)
            if all(a.r1 == b.r0 and a.shape == b.shape for a, b in zip(blocks, blocks[1:])):
                local.update(run)
                if r == self.rank:
                    mine[i] = LocalBlocks(blocks[0].r0 - regions[i][r][0], len(blocks), blocks[0].r1 - blocks[0].r0,
                                          blocks[0].c1 - blocks[0].c0)
        return local, mine

    def _plan_block(self, k: int, slices: List[Tuple[int, Box]], regions: List[List[Box]]):
        """Add remote block k to its matrix's plan on this rank."""
        b = self.blocks[k]
        plan = self.plans[b.param]
        if b.owner == self.rank:
            plan.owned.append(k)
        for r, box in slices:
            if r == self.rank:
                plan.send[b.owner].append(_local_box(box, regions[b.param][r]))
            if b.owner == self.rank:
                plan.recv[r].append((len(plan.owned) - 1, _local_box(box, (b.r0, b.r1, b.c0, b.c1))))

    def _batches(self, limit: int) -> List[List[int]]:
        """The local matrices in batches: by the shape of their blocks (one buffer per batch), consecutive ones
        until a batch holds `limit` elements of this rank (0: one batch per shape)."""
        remote = set(self.remote)
        by_shape = defaultdict(list)
        for i, plan in enumerate(self.plans):
            if i not in remote:
                b = plan.local
                by_shape[b.height, b.cols].append((i, b.count * b.height * b.cols))
        return [[i for i, _ in matrices[lo:hi]]
                for matrices in by_shape.values() for lo, hi in _split([n for _, n in matrices], limit)]

    def _regions(self, spec: MuonSpec, shape: Tuple[int, int]) -> List[Box]:
        """The box of a (global) matrix that each rank stores. TP gives rank (i, t) the t-th chunk of the
        weight along spec.tp_dim (every rank holds all of it if the weight is not split); FSDP2 Shard(0)
        then cuts that TP-local tensor by torch.chunk (contiguous rows, all columns; trailing ranks may get none)."""
        local = list(shape)
        if self.tp > 1 and spec.tp_dim is not None:
            local[spec.tp_dim] //= self.tp
        step = math.ceil(local[0] / self.dp)
        boxes = []
        for i, t in self.coords:
            r0, r1 = min(i * step, local[0]), min((i + 1) * step, local[0])
            row_off = t * local[0] if self.tp > 1 and spec.tp_dim == 0 else 0
            col_off = t * local[1] if self.tp > 1 and spec.tp_dim == 1 else 0
            boxes.append((row_off + r0, row_off + r1, col_off, col_off + local[1]))
        return boxes

    @staticmethod
    def _ns_cost(b: Block) -> int:
        m, n = b.shape
        return m * n * min(m, n)

    @staticmethod
    def _scale(fan_out: int, fan_in: int, adjust_lr: Optional[str]) -> float:
        """Update scale of a fan_out x fan_in matrix (one Block's, or with muon_block_scale=matrix the weight's)."""
        if adjust_lr == "spectral_norm":
            return math.sqrt(fan_out / fan_in)
        if adjust_lr == "rms_norm":
            return 0.2 * math.sqrt(max(fan_out, fan_in))
        return 1.0

    # ---- step ----

    def _exchange(self, send: List[List[Tensor]],
                  recv_shapes: List[List[Tuple[int, int]]]) -> Callable[[], List[List[Tensor]]]:
        """Launch an all-to-all of bf16 tensors, send[r] going to rank r. Returns a function that waits
        for it and returns the tensors received from each rank."""
        send_shapes = [[tuple(t.shape) for t in ts] for ts in send]
        send_sizes = [sum(m * n for m, n in shapes) for shapes in send_shapes]
        recv_sizes = [sum(m * n for m, n in shapes) for shapes in recv_shapes]
        send_buf = torch.empty(sum(send_sizes), dtype=torch.bfloat16, device="cuda")
        _copy_(_views(send_buf, send_shapes), send)
        recv_buf = torch.empty(sum(recv_sizes), dtype=torch.bfloat16, device="cuda")
        work = dist.all_to_all_single(recv_buf, send_buf, recv_sizes, send_sizes, group=self.process_group,
                                      async_op=True)

        def wait():
            work.wait()
            return _views(recv_buf, recv_shapes)
        return wait

    def _momentum(self, params: List[Tensor], group: Dict, out: List[Tensor]):
        """Momentum update on the local shards (one batched call); writes the bf16 update directions into `out`,
        contiguous tensors of the shards' sizes in any shape, zeros for the params without a gradient."""
        # Keep the fixed communication plan, but do not create or decay state for
        # a parameter without a gradient (the same semantics as AdamW).
        mu = group["momentum"]
        live = []
        for i, p in enumerate(params):
            if p.grad is None:
                out[i].zero_()
                continue
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
            live.append(i)
        xllm_ops.muon_momentum_([_local(self.state[params[i]]["momentum"]) for i in live],
                                [_local(params[i].grad) for i in live], [out[i] for i in live], mu, group["nesterov"])

    def _gather(self, mine: List[List[Tensor]], plans: List[MatrixPlan]) -> Callable[[], Tuple[List[Tensor], List[Tensor]]]:
        """Launch sending my slices of remote blocks (mine[r]: those rank r owns) to their owners. Returns a
        function that waits and returns the remote blocks I own, in plan order, and the [M, R, C] buffers that
        hold them."""
        recv = self._exchange(mine, [[_shape(box) for _, box in slices] for slices in self.owned_slices])

        def wait():
            views, buffers = _packed([(1, *self.blocks[k].shape) for plan in plans for k in plan.owned])
            out = [v[0] for v in views]
            _copy_([[out[j][box] for j, box in slices] for slices in self.owned_slices], recv())
            return out, buffers
        return wait

    def _scatter(self, blocks: List[Tensor], mine: List[List[Tensor]]):
        """Send each rank its slices of the remote blocks I own; write the slices I receive into `mine`."""
        send = [[blocks[j][box] for j, box in slices] for slices in self.owned_slices]
        recv = self._exchange(send, [[t.shape for t in ts] for ts in mine])
        _copy_(mine, recv())

    @torch.no_grad()
    def _muon_step(self, group: Dict):
        eps = group["eps"]
        scales = [self._scale(*fans, group["adjust_lr"]) for fans in self._fans[group["block_scale"]]]
        params = [group["params"][i] for i in self.remote]
        plans = [self.plans[i] for i in self.remote]
        updates = [torch.empty_like(_local(p), dtype=torch.bfloat16) for p in params]
        self._momentum(params, group, updates)                                  # 1. remote matrices: momentum,
        # mine[r]: my slices of the remote blocks rank r owns, views of the updates: sent, then overwritten
        mine = [[u[box] for u, plan in zip(updates, plans) for box in plan.send[r]] for r in range(self.world)]
        gathered = self._gather(mine, plans) if self.has_remote else None       #    remote blocks on the wire
        for batch in self.batches:                                              # 2. meanwhile the local matrices
            self._step_local([group["params"][i] for i in batch], [self.plans[i] for i in batch],
                             [scales[i] for i in batch], group)
        for u, plan in zip(updates, plans):                                     # 3. remote matrices: local blocks,
            if plan.local:
                orthogonalize_(_rows_as_blocks(u, plan.local), eps)
        if self.has_remote:
            blocks, buffers = gathered()                                        #    the remote blocks I own,
            del gathered                                                        #    (frees the receive buffer)
            for X in buffers:
                orthogonalize_(X, eps)
            self._scatter(blocks, mine)                                         #    back
        self._update(params, updates, [scales[i] for i in self.remote], group["lr"], group["weight_decay"])

    def _step_local(self, params: List[Tensor], plans: List[MatrixPlan], scales: List[float], group: Dict):
        """One batch of local matrices: momentum into one buffer per block shape, Newton-Schulz on it in place, update."""
        updates, buffers = _packed([(plan.local.count, plan.local.height, plan.local.cols) for plan in plans])
        self._momentum(params, group, updates)
        for X in buffers:
            orthogonalize_(X, group["eps"])
        self._update(params, updates, scales, group["lr"], group["weight_decay"])

    @staticmethod
    def _update(params: List[Tensor], updates: List[Tensor], scales: List[float], lr: float, wd: float):
        """W <- W - lr (wd W + scale update), in fp32, for the params with a gradient; updates[i] holds the
        shard's elements, contiguous, in any shape (its blocks)."""
        live = [i for i, p in enumerate(params) if p.grad is not None]
        xllm_ops.muon_update_([_local(params[i]) for i in live], [updates[i] for i in live],
                              [scales[i] for i in live], lr, wd)

    @torch.no_grad()
    def _adamw_step(self, group: Dict):
        """torch.optim.AdamW's update (foreach) on the group's params."""
        params = [p for p in group["params"] if p.grad is not None]
        states = [self.state[p] for p in params]
        for p, state in zip(params, states):
            if "step" not in state:
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        adamw(params, [p.grad for p in params], [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states],
              [], [s["step"] for s in states], foreach=True, amsgrad=False, beta1=group["betas"][0],
              beta2=group["betas"][1], lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
              maximize=False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss


def _pattern(fqn: str) -> str:
    """layers.3.attention.wq.weight -> layers.*.attention.wq.weight"""
    return re.sub(r"(?<![^.])\d+(?=\.|$)", "*", fqn)


def _warn_unregistered(model: nn.Module, muon_ids: set):
    """2-D weights of tensor-parallel layers in the blocks that are not registered use AdamW: say which."""
    from xllm.modules.model_parallel import ColumnParallelLinear, RowParallelLinear
    missed = defaultdict(int)
    for fqn, m in model.named_modules():
        if "layers." in fqn and isinstance(m, (ColumnParallelLinear, RowParallelLinear)) and id(m.weight) not in muon_ids:
            missed[_pattern(f"{fqn}.weight")] += 1
    if missed:
        logger.warning(f"Muon: unregistered weights use AdamW: {dict(missed)}")


def build_muon(model: nn.Module, cfg: OptimConf) -> Optimizer:
    """The model must be wrapped with FSDP2; Muon works over the shard dimension of the mesh its parameters live on."""
    from xllm.distributed import get_model_parallel_group, get_model_parallel_world_size
    tp = get_model_parallel_world_size()
    muon_params, specs, names, adamw_params = split_muon_params(model)
    lines = []
    for name, spec, p in zip(names, specs, muon_params):
        fan_out, fan_in = spec.fans(spec.global_shape(tuple(p.shape), tp), per_block=True)
        lines.append(f"Muon {name} {list(p.shape)} {str(spec)!r}: {len(spec.blocks(tuple(p.shape)))} blocks of "
                     f"fan_out {fan_out} x fan_in {fan_in}, tp_dim {spec.tp_dim}")
    for line in dict.fromkeys(lines):  # one line per kind of weight, e.g. layers.*.attention.wq.weight
        logger.info(line)
    _warn_unregistered(model, {id(p) for p in muon_params})
    if not muon_params or not hasattr(muon_params[0], "device_mesh"):
        raise ValueError("Muon requires a model wrapped with FSDP2 and at least one registered weight.")
    mesh = muon_params[0].device_mesh
    shard_mesh = mesh if mesh.ndim == 1 else mesh["dp"]
    tp_group = get_model_parallel_group() if tp > 1 else None
    optimizer = Muon(muon_params, specs, adamw_params, shard_mesh, cfg, tp_group=tp_group)
    logger.info(f"Muon: {len(muon_params)} matrices / {len(optimizer.blocks)} blocks (scale={cfg.muon_block_scale}) / "
                f"{sum(p.numel() for p in muon_params)} params; AdamW: {len(adamw_params)} tensors / "
                f"{sum(p.numel() for p in adamw_params)} params")
    return optimizer
