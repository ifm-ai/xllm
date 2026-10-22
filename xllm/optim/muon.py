"""Muon on FSDP2 with per-head orthogonalization.

Which weights use Muon: a model marks a linear layer with a `muon` attribute, and that
layer's weight goes to Muon; everything unmarked goes to AdamW.

    linear.muon = n    Muon; its rows are cut into n equal blocks, each orthogonalized on
                       its own: 1 = whole matrix, the number of heads = one block per head,
                       in between = blocks of adjacent heads.

The mark sits on the layer, not on the weight tensor: FSDP2 and Module._apply (.cuda(), .to())
replace or swap DTensor parameters and drop attributes set on them.

Terminology, for one Muon matrix W of shape [rows, cols]:

- shard: FSDP2 Shard(0) stores W as contiguous row ranges, one per rank (torch.chunk).
- Block: one of the n row ranges of W from its mark, orthogonalized by Newton-Schulz on
  its own; chosen from the model, not from the sharding (the whole matrix, or one or
  more heads of wq / wk / wv / wr).
- owner: the rank that gathers a Block and runs NS on it: the rank that stores most of
  its rows, then the least loaded one.
- slice: the rows of a Block that one rank stores (the Block's overlap with that rank's
  shard). A Block is the concatenation of its slices in rank order.

Example: wk [8, 4] with wk.muon = 2 on 3 ranks (torch.chunk: rows 0-2, 3-5, 6-7):

    rank 0  rows 0-2  a          block 0 = rows 0-3 = cat(a, b1), owner rank 0 (stores 3 rows)
    rank 1  rows 3-5  b1 | b2    block 1 = rows 4-7 = cat(b2, c), owner rank 1 (tie with rank 2)
    rank 2  rows 6-7  c          b1 and b2 are rank 1's slices of blocks 0 and 1
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
from torch.optim.adamw import adamw

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
    """Rows [r0, r1) of Muon matrix `param` (one or more heads, or the whole matrix), all
    `cols` columns; Newton-Schulz orthogonalizes it on its own."""
    param: int
    r0: int
    r1: int
    cols: int
    owner: int = 0  # rank that gathers the block and runs Newton-Schulz on it

    @property
    def shape(self) -> Tuple[int, int]:
        return self.r1 - self.r0, self.cols


def split_muon_params(model: nn.Module):
    """Weights of layers marked with a `muon` attribute go to Muon; everything else goes to AdamW.

    Returns the Muon params, the number of row blocks of each (its mark), and the AdamW params.
    """
    marks = {id(m.weight): m.muon for m in model.modules() if hasattr(m, "muon")}
    muon_params, n_blocks, adamw_params = [], [], []
    for p in model.parameters():
        mark = marks.get(id(p))
        if mark is None:
            adamw_params.append(p)
            continue
        muon_params.append(p)
        n_blocks.append(mark)
    return muon_params, n_blocks, adamw_params


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
    2. gather: one all-to-all sends every slice to its block's owner;
    3. each owner runs Newton-Schulz on its blocks;
    4. scatter: one all-to-all sends the slices back; each rank applies its update.
    """

    def __init__(
        self,
        muon_params: List[nn.Parameter],
        n_blocks: List[int],
        adamw_params: List[nn.Parameter],
        mesh: DeviceMesh,
        cfg: OptimConf,
    ):
        defaults = dict(weight_decay=cfg.weight_decay, eps=cfg.epsilon)
        super().__init__(
            [
                dict(params=muon_params, lr=cfg.muon_lr, algorithm="muon",
                     weight_decay=cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay),
                dict(params=adamw_params, lr=cfg.lr, algorithm="adamw"),
            ],
            defaults,
        )
        self.momentum = cfg.muon_momentum
        self.nesterov = cfg.muon_nesterov
        self.adjust_lr = cfg.muon_adjust_lr
        self.block_scale = cfg.muon_block_scale
        self.betas = (cfg.beta1, cfg.beta2)
        self.process_group = mesh.get_group()
        self.world = mesh.size()
        self.rank = mesh.get_local_rank()
        self._plan([tuple(p.shape) for p in muon_params], n_blocks)

    # ---- plan: built once from the global shapes, identical on every rank ----

    def _plan(self, shapes: List[Tuple[int, int]], n_blocks: List[int]):
        self.shapes = shapes
        self.blocks: List[Block] = []
        for i, ((rows, cols), n) in enumerate(zip(shapes, n_blocks)):
            assert rows % n == 0, (i, rows, n)
            size = rows // n
            self.blocks += [Block(i, s, s + size, cols) for s in range(0, rows, size)]

        # shard_start[i][r]: first global row of matrix i on rank r (FSDP2 Shard(0) follows torch.chunk)
        shard_start = [[min(r * math.ceil(rows / self.world), rows) for r in range(self.world + 1)]
                       for rows, _ in shapes]
        # slices[k]: (rank, global rows lo, hi) for every rank that stores rows of block k
        slices = []
        for b in self.blocks:
            start = shard_start[b.param]
            overlap = [(r, max(b.r0, start[r]), min(b.r1, start[r + 1])) for r in range(self.world)]
            slices.append([(r, lo, hi) for r, lo, hi in overlap if lo < hi])

        # owner: the rank that stores most of the block (no transfer for those rows), then the least
        # loaded; blocks are placed by decreasing Newton-Schulz cost m * n * min(m, n)
        load = [0] * self.world
        for k in sorted(range(len(self.blocks)), key=lambda k: -self._ns_cost(self.blocks[k])):
            stored = dict.fromkeys(range(self.world), 0)
            for r, lo, hi in slices[k]:
                stored[r] = hi - lo
            b = self.blocks[k]
            b.owner = min(range(self.world), key=lambda r: (-stored[r], load[r]))
            load[b.owner] += self._ns_cost(b)

        # my_slices[r]: my slices of the blocks rank r owns, as (matrix, rows of my shard)
        # owned_slices[r]: rank r's slices of the blocks I own, as (index in self.owned, rows of the block)
        # Both lists follow the block order, so a sender's list matches its receiver's.
        self.owned = [k for k, b in enumerate(self.blocks) if b.owner == self.rank]
        index = {k: j for j, k in enumerate(self.owned)}
        self.my_slices = [[] for _ in range(self.world)]
        self.owned_slices = [[] for _ in range(self.world)]
        for k, b in enumerate(self.blocks):
            for r, lo, hi in slices[k]:
                if r == self.rank:
                    s0 = shard_start[b.param][r]
                    self.my_slices[b.owner].append((b.param, slice(lo - s0, hi - s0)))
                if b.owner == self.rank:
                    self.owned_slices[r].append((index[k], slice(lo - b.r0, hi - b.r0)))

    @staticmethod
    def _ns_cost(b: Block) -> int:
        m, n = b.shape
        return m * n * min(m, n)

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
        dist.all_to_all_single(recv_buf, send_buf, recv_sizes, send_sizes, group=self.process_group)
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
        """Send my slices to their blocks' owners; return the whole blocks I own."""
        send = [[updates[i][rows] for i, rows in self.my_slices[r]] for r in range(self.world)]
        recv = self._exchange(send, [[(rows.stop - rows.start, self.blocks[self.owned[j]].cols)
                                      for j, rows in self.owned_slices[r]] for r in range(self.world)])
        blocks = [torch.empty(self.blocks[k].shape, dtype=torch.bfloat16, device="cuda") for k in self.owned]
        for r in range(self.world):
            for (j, rows), t in zip(self.owned_slices[r], recv[r]):
                blocks[j][rows] = t
        return blocks

    def _scatter(self, results: List[Tensor], updates: List[Tensor]):
        """Send each rank its slices of the blocks I own; write the slices I receive into `updates`."""
        send = [[results[j][rows] for j, rows in self.owned_slices[r]] for r in range(self.world)]
        recv = self._exchange(send, [[(rows.stop - rows.start, self.shapes[i][1])
                                      for i, rows in self.my_slices[r]] for r in range(self.world)])
        for r in range(self.world):
            for (i, rows), t in zip(self.my_slices[r], recv[r]):
                updates[i][rows] = t

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
        """torch.optim.AdamW's update (foreach) on the group's params."""
        params = [p for p in group["params"] if p.grad is not None]
        states = [self.state[p] for p in params]
        for p, state in zip(params, states):
            if "step" not in state:
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        adamw(params, [p.grad for p in params], [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states],
              [], [s["step"] for s in states], foreach=True, amsgrad=False, beta1=self.betas[0],
              beta2=self.betas[1], lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
              maximize=False)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)


def build_muon(model: nn.Module, mesh: DeviceMesh, cfg: OptimConf) -> Optimizer:
    """`mesh` is the FSDP mesh; Muon works over its shard dimension."""
    muon_params, n_blocks, adamw_params = split_muon_params(model)
    shard_mesh = mesh if mesh.ndim == 1 else mesh["dp"]
    optimizer = Muon(muon_params, n_blocks, adamw_params, shard_mesh, cfg)
    logger.info(f"Muon: {len(muon_params)} matrices / {len(optimizer.blocks)} blocks (scale={cfg.muon_block_scale}) / "
                f"{sum(p.numel() for p in muon_params)} params; AdamW: {len(adamw_params)} tensors / "
                f"{sum(p.numel() for p in adamw_params)} params")
    return optimizer
