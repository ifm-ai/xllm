"""Muon on FSDP2 with per-head orthogonalization.

Which weights use Muon: a model marks a linear layer with a `muon` attribute, and that
layer's weight goes to Muon; everything unmarked goes to AdamW.

    linear.muon = n    Muon; with per-head, its rows are cut into n equal groups, each
                       orthogonalized on its own: 1 = whole matrix, the number of heads =
                       one group per head, in between = groups of adjacent heads.
                       Without per-head every matrix is one group.

The mark sits on the layer, not on the weight tensor: FSDP2 and Module._apply (.cuda(), .to())
replace or swap DTensor parameters and drop attributes set on them.

Terminology, for one Muon matrix W of shape [rows, cols]:

- shard: FSDP2 Shard(0) stores W as contiguous row ranges, one per rank (torch.chunk).
- Block: a row range of W that Newton-Schulz orthogonalizes on its own, chosen from the
  model, not from the sharding: one of the matrix's groups (the whole matrix, or one or
  more heads of wq / wk / wv / wr).
- owner: the rank a Block is assigned to; it gathers the block and runs NS on it.
- Slice: the rows of one Block stored on one rank (Block intersected with
  that rank's shard). Concatenating the slices in rank order gives the whole block.

Example: wk [8, 4] with 2 heads of 4 rows on 3 ranks (torch.chunk: rows 0-2, 3-5, 6-7):

    rank 0  rows 0-2  a          block 0 = rows 0-3 = cat(a, b1)
    rank 1  rows 3-5  b1 | b2    block 1 = rows 4-7 = cat(b2, c)
    rank 2  rows 6-7  c          a is block 0's Slice on rank 0; b1, b2 are rank 1's two Slices
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from xllm.config import OptimConf


logger = getLogger()

# Newton-Schulz 5 steps with per-step coefficients (same as microsoft/dion)
NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

@dataclass
class Block:
    """Rows [r0, r1) of Muon matrix `param` (one group of heads, or the whole matrix), all
    `cols` columns; Newton-Schulz orthogonalizes it on its own."""
    param: int
    r0: int
    r1: int
    cols: int
    owner: int = 0  # rank that gathers the block and runs Newton-Schulz on it

    @property
    def shape(self) -> Tuple[int, int]:
        return self.r1 - self.r0, self.cols


@dataclass
class Slice:
    """The rows of one Block that one rank stores: `rows` indexes that rank's local shard,
    and `block_row` is the first row of the slice inside the block. May be empty."""
    rows: slice
    block_row: int

    @property
    def num_rows(self) -> int:
        return self.rows.stop - self.rows.start


def split_muon_params(model: nn.Module, per_head: bool):
    """Weights of layers marked with a `muon` attribute go to Muon; everything else goes to AdamW.

    Returns the Muon params, the number of row groups of each (1 without per_head), and
    the AdamW params.
    """
    marks = {id(m.weight): m.muon for m in model.modules() if hasattr(m, "muon")}
    muon, groups, adamw = [], [], []
    for p in model.parameters():
        mark = marks.get(id(p))
        if mark is None:
            adamw.append(p)
            continue
        muon.append(p)
        groups.append(mark if per_head else 1)
    return muon, groups, adamw


def _local(t: Tensor) -> Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


@torch.compile(fullgraph=True)
def newton_schulz5(X: Tensor, eps: float) -> Tensor:
    """Orthogonalize a stack of matrices [B, m, n] with m <= n."""
    X = X.to(torch.bfloat16)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in NS_COEFFS:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X


def orthogonalize(mats: List[Tensor], eps: float) -> List[Tensor]:
    """Newton-Schulz on each matrix, batched over matrices of the same shape.
    Tall matrices are transposed (NS(X^T) = NS(X)^T), so [m, n] and [n, m] share a batch."""
    by_shape = defaultdict(list)
    for k, x in enumerate(mats):
        by_shape[(min(x.shape), max(x.shape))].append(k)
    out = list(mats)
    for idx in by_shape.values():
        tall = [mats[k].shape[0] > mats[k].shape[1] for k in idx]
        X = newton_schulz5(torch.stack([mats[k].mT if t else mats[k] for k, t in zip(idx, tall)]), eps)
        for j, (k, t) in enumerate(zip(idx, tall)):
            out[k] = X[j].mT if t else X[j]
    return out


class Muon(Optimizer):
    """Muon for FSDP2 parameters sharded on dim 0, with AdamW for the rest.

    Each step:
    1. update the momentum on the local shard;
    2. gather: one all-to-all sends every Slice to its block's owner;
    3. each owner runs Newton-Schulz on its blocks;
    4. scatter: one all-to-all sends the slices back; each rank applies its update.
    """

    def __init__(
        self,
        muon_params: List[nn.Parameter],
        groups: List[int],
        adamw_params: List[nn.Parameter],
        mesh: DeviceMesh,
        cfg: OptimConf,
    ):
        defaults = dict(weight_decay=cfg.weight_decay, eps=cfg.epsilon)
        super().__init__(
            [
                dict(params=muon_params, lr=cfg.muon_lr, algorithm="muon"),
                dict(params=adamw_params, lr=cfg.lr, algorithm="adamw"),
            ],
            defaults,
        )
        self.momentum = cfg.muon_momentum
        self.nesterov = cfg.muon_nesterov
        self.adjust_lr = cfg.muon_adjust_lr
        self.block_scale = cfg.muon_block_scale
        self.betas = (cfg.beta1, cfg.beta2)
        self.group = mesh.get_group()
        self.world = mesh.size()
        self.rank = mesh.get_local_rank()
        self._plan([tuple(p.shape) for p in muon_params], groups)

    # ---- plan: built once from the global shapes, identical on every rank ----

    def _plan(self, shapes: List[Tuple[int, int]], groups: List[int]):
        self.shapes = shapes
        self.blocks: List[Block] = []
        for i, ((rows, cols), n) in enumerate(zip(shapes, groups)):
            assert rows % n == 0, (i, rows, n)
            size = rows // n
            self.blocks += [Block(i, s, min(s + size, rows), cols) for s in range(0, rows, size)]

        # balance owners by size, largest blocks first
        load = [0] * self.world
        for b in sorted(self.blocks, key=lambda b: -math.prod(b.shape)):
            b.owner = min(range(self.world), key=lambda r: load[r])
            load[b.owner] += math.prod(b.shape)

        # slices[k][r]: the rows of block k stored on rank r
        shard_rows = [[self._shard_rows(rows, r) for r in range(self.world)] for rows, _ in shapes]
        self.slices = [[self._slice_on_shard(b, *shard_rows[b.param][r]) for r in range(self.world)]
                       for b in self.blocks]
        self.owned = [k for k, b in enumerate(self.blocks) if b.owner == self.rank]
        self.owned_by = [[k for k, b in enumerate(self.blocks) if b.owner == r] for r in range(self.world)]

    def _shard_rows(self, n: int, rank: int) -> Tuple[int, int]:
        """Rows of an n-row matrix stored on `rank` (FSDP2 Shard(0) follows torch.chunk)."""
        chunk = math.ceil(n / self.world)
        return min(rank * chunk, n), min((rank + 1) * chunk, n)

    @staticmethod
    def _slice_on_shard(b: Block, s0: int, s1: int) -> Slice:
        """The rows of block b inside the shard that holds global rows [s0, s1)."""
        lo, hi = max(b.r0, s0), min(b.r1, s1)
        hi = max(lo, hi)
        return Slice(rows=slice(lo - s0, hi - s0), block_row=lo - b.r0)

    def _scale(self, b: Block) -> float:
        """Update scale of block b, from its own shape or from its whole matrix's shape."""
        fan_out, fan_in = b.shape if self.block_scale == "block" else self.shapes[b.param]
        if self.adjust_lr == "spectral_norm":
            return math.sqrt(fan_out / fan_in)
        if self.adjust_lr == "rms_norm":
            return 0.2 * math.sqrt(max(fan_out, fan_in))
        return 1.0

    # ---- step ----

    def _exchange(self, send: List[List[Tensor]], recv_shapes: List[List[Tuple[int, int]]]) -> List[List[Tensor]]:
        """all-to-all of bf16 tensors: send[r] goes to rank r; returns the tensors received from each rank."""
        send_sizes = [sum(t.numel() for t in ts) for ts in send]
        recv_sizes = [sum(m * n for m, n in shapes) for shapes in recv_shapes]
        flat = [t.reshape(-1) for ts in send for t in ts]
        send_buf = torch.cat(flat) if flat else torch.empty(0, dtype=torch.bfloat16, device="cuda")
        recv_buf = torch.empty(sum(recv_sizes), dtype=torch.bfloat16, device="cuda")
        dist.all_to_all_single(recv_buf, send_buf, recv_sizes, send_sizes, group=self.group)
        out = []
        for chunk, shapes in zip(recv_buf.split(recv_sizes), recv_shapes):
            out.append([t.view(shape) for t, shape in zip(chunk.split([m * n for m, n in shapes]), shapes)])
        return out

    def _momentum(self, p: Tensor) -> Tensor:
        """Momentum update on the local shard; returns the bf16 update direction."""
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
        m = _local(state["momentum"])
        # every rank must take part in the exchange, so a missing grad counts as zero
        g = _local(p.grad) if p.grad is not None else torch.zeros_like(m)
        m.mul_(self.momentum).add_(g)
        u = g + self.momentum * m if self.nesterov else m
        return u.to(torch.bfloat16)

    def _gather(self, updates: List[Tensor]) -> List[Tensor]:
        """Send every shard slice to its block's owner; return the whole blocks this rank owns."""
        send = [[] for _ in range(self.world)]
        for b, slices in zip(self.blocks, self.slices):
            mine = slices[self.rank]
            send[b.owner].append(updates[b.param][mine.rows])
        recv_shapes = [[(self.slices[k][src].num_rows, self.blocks[k].shape[1]) for k in self.owned]
                       for src in range(self.world)]
        recv = self._exchange(send, recv_shapes)
        # slices arrive in rank order, which is the row order inside the block
        return [torch.cat([recv[src][j] for src in range(self.world)]) for j in range(len(self.owned))]

    def _scatter(self, results: List[Tensor], updates: List[Tensor]):
        """Send each rank its rows of every owned block; write the received rows into `updates`."""
        send = [[] for _ in range(self.world)]
        for k, out in zip(self.owned, results):
            for dst in range(self.world):
                theirs = self.slices[k][dst]
                send[dst].append(out[theirs.block_row:theirs.block_row + theirs.num_rows])
        recv_shapes = [[(self.slices[k][self.rank].num_rows, self.blocks[k].shape[1]) for k in self.owned_by[src]]
                       for src in range(self.world)]
        recv = self._exchange(send, recv_shapes)
        for src in range(self.world):
            for k, rows in zip(self.owned_by[src], recv[src]):
                mine = self.slices[k][self.rank]
                updates[self.blocks[k].param][mine.rows] = rows

    @torch.no_grad()
    def _muon_step(self, group: Dict):
        params, lr, wd = group["params"], group["lr"], group["weight_decay"]
        updates = [self._momentum(p) for p in params]                      # 1. local shards
        gathered = self._gather(updates)                                   # 2. whole owned blocks
        results = [o * self._scale(self.blocks[k])                         # 3. Newton-Schulz + scale
                   for k, o in zip(self.owned, orthogonalize(gathered, group["eps"]))]
        self._scatter(results, updates)                                    # 4. back to the shards
        for p, u in zip(params, updates):
            x = _local(p)
            x.mul_(1 - lr * wd).add_(u.to(x.dtype), alpha=-lr)

    @torch.no_grad()
    def _adamw_step(self, group: Dict):
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = self.betas
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
            step = state["step"].item()
            x, g = _local(p), _local(p.grad).float()
            m, v = _local(state["exp_avg"]), _local(state["exp_avg_sq"])
            m.lerp_(g, 1 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            denom = (v / (1 - beta2 ** step)).sqrt_().add_(eps)
            x.mul_(1 - lr * wd).addcdiv_(m.to(x.dtype), denom.to(x.dtype), value=-lr / (1 - beta1 ** step))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)


def build_muon(model: nn.Module, mesh: DeviceMesh, cfg: OptimConf) -> Optimizer:
    """`mesh` is the FSDP mesh; Muon works over its shard dimension."""
    muon, groups, adamw = split_muon_params(model, cfg.muon_per_head)
    shard_mesh = mesh if mesh.ndim == 1 else mesh["dp"]
    optimizer = Muon(muon, groups, adamw, shard_mesh, cfg)
    logger.info(f"Muon: {len(muon)} matrices / {len(optimizer.blocks)} blocks (scale={cfg.muon_block_scale}) / "
                f"{sum(p.numel() for p in muon)} params; AdamW: {len(adamw)} tensors / "
                f"{sum(p.numel() for p in adamw)} params")
    return optimizer
