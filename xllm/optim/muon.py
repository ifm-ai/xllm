"""Muon on FSDP2 with per-head orthogonalization.

Terminology, for one Muon matrix W of shape [rows, cols]:

- shard: FSDP2 Shard(0) stores W as contiguous row ranges, one per rank (torch.chunk).
- Block: a submatrix that Newton-Schulz orthogonalizes on its own, chosen from the
  model, not from the sharding. Whole matrix by default; with per-head, one head: a row
  range of wq / wk / wv / wr, or a column range of wo.
- owner: the rank a Block is assigned to; it gathers the block and runs NS on it.
- Slice: the rows of one Block stored on one rank (Block intersected with
  that rank's shard). Concatenating the slices in rank order gives the whole block.

Example: wo [8, 8] with 2 heads of 4 columns on 4 ranks (2 rows each):

            block 0     block 1
    rank 0  a a a a  |  e e e e      block 0 = cat(a, b, c, d), owner rank 0
    rank 1  b b b b  |  f f f f      block 1 = cat(e, f, g, h), owner rank 1
    rank 2  c c c c  |  g g g g      a is block 0's Slice on rank 0
    rank 3  d d d d  |  h h h h
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from xllm.config import OptimConf
from xllm.modules.gated_delta_attention import GatedDeltaAttention
from xllm.modules.model_parallel.layers import ColumnParallelLinear, RowParallelLinear
from xllm.modules.multihead_attention import MultiheadAttention


logger = getLogger()

# Newton-Schulz 5 steps with per-step coefficients (same as microsoft/dion)
NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

# How a Muon matrix is cut into blocks: None = whole matrix, (0, n) = row blocks of
# n rows, (1, n) = column blocks of n columns.
BlockSpec = Optional[Tuple[int, int]]


@dataclass
class Block:
    """A submatrix that Newton-Schulz orthogonalizes on its own (one attention head, or a whole
    matrix): rows [r0, r1) x columns [c0, c1) of Muon matrix `param`, in global coordinates."""
    param: int
    r0: int
    r1: int
    c0: int
    c1: int
    owner: int = 0  # rank that gathers the block and runs Newton-Schulz on it

    @property
    def shape(self) -> Tuple[int, int]:
        return self.r1 - self.r0, self.c1 - self.c0


@dataclass
class Slice:
    """The rows of one Block that one rank stores: `rows`, `cols` index that rank's local
    shard, and `block_row` is the first row of the slice inside the block. May be empty."""
    rows: slice
    cols: slice
    block_row: int

    @property
    def num_rows(self) -> int:
        return self.rows.stop - self.rows.start


def split_muon_params(model: nn.Module, per_head: bool):
    """Linear weights inside the blocks go to Muon; everything else
    (embedding, output layer, norms, biases, convolutions) goes to AdamW.

    Returns the Muon params, one BlockSpec per Muon param, and the AdamW params.
    With per_head, the attention projections are cut into heads: rows of wq / wk / wv / wr,
    columns of wo.
    """
    heads = {}
    if per_head:
        for attn in model.layers.modules():
            if isinstance(attn, (GatedDeltaAttention, MultiheadAttention)):
                v_dim = getattr(attn, "v_head_dim", attn.head_dim)
                for name, dim, size in (("wq", 0, attn.head_dim), ("wk", 0, attn.head_dim),
                                        ("wv", 0, v_dim), ("wr", 0, v_dim), ("wo", 1, v_dim)):
                    linear = getattr(attn, name, None)
                    if linear is not None:
                        heads[id(linear.weight)] = (dim, size)

    muon = [
        m.weight for m in model.layers.modules()
        if isinstance(m, (ColumnParallelLinear, RowParallelLinear))
    ]
    specs = [heads.get(id(p)) for p in muon]
    muon_ids = {id(p) for p in muon}
    adamw = [p for p in model.parameters() if id(p) not in muon_ids]
    return muon, specs, adamw


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
        block_specs: List[BlockSpec],
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
        self.betas = (cfg.beta1, cfg.beta2)
        self.group = mesh.get_group()
        self.world = mesh.size()
        self.rank = mesh.get_local_rank()
        self._plan([tuple(p.shape) for p in muon_params], block_specs)

    # ---- plan: built once from the global shapes, identical on every rank ----

    def _plan(self, shapes: List[Tuple[int, int]], specs: List[BlockSpec]):
        self.blocks: List[Block] = []
        for i, ((rows, cols), spec) in enumerate(zip(shapes, specs)):
            if spec is None:
                self.blocks.append(Block(i, 0, rows, 0, cols))
            elif spec[0] == 0:
                self.blocks += [Block(i, s, min(s + spec[1], rows), 0, cols) for s in range(0, rows, spec[1])]
            else:
                self.blocks += [Block(i, 0, rows, s, min(s + spec[1], cols)) for s in range(0, cols, spec[1])]

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
        """The rows of block b inside the shard that holds global rows [s0, s1) and all columns."""
        lo, hi = max(b.r0, s0), min(b.r1, s1)
        hi = max(lo, hi)
        return Slice(rows=slice(lo - s0, hi - s0), cols=slice(b.c0, b.c1), block_row=lo - b.r0)

    def _scale(self, shape: Tuple[int, int]) -> float:
        fan_out, fan_in = shape
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
            send[b.owner].append(updates[b.param][mine.rows, mine.cols])
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
                updates[self.blocks[k].param][mine.rows, mine.cols] = rows

    @torch.no_grad()
    def _muon_step(self, group: Dict):
        params, lr, wd = group["params"], group["lr"], group["weight_decay"]
        updates = [self._momentum(p) for p in params]                      # 1. local shards
        gathered = self._gather(updates)                                   # 2. whole owned blocks
        results = [o * self._scale(self.blocks[k].shape)                   # 3. Newton-Schulz + scale
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
    muon, specs, adamw = split_muon_params(model, cfg.muon_per_head)
    shard_mesh = mesh if mesh.ndim == 1 else mesh["dp"]
    optimizer = Muon(muon, specs, adamw, shard_mesh, cfg)
    logger.info(f"Muon: {len(muon)} matrices / {len(optimizer.blocks)} blocks / "
                f"{sum(p.numel() for p in muon)} params; AdamW: {len(adamw)} tensors / "
                f"{sum(p.numel() for p in adamw)} params")
    return optimizer
