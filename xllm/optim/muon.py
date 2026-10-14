import math
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Tuple

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


@dataclass
class Block:
    """A slice of a Muon matrix that is orthogonalized on its own:
    rows [start, start + size) if dim == 0, columns if dim == 1."""
    param: int
    dim: int
    start: int
    size: int
    owner: int = 0


def split_muon_params(model: nn.Module, per_head: bool):
    """Linear weights inside the blocks go to Muon; everything else
    (embedding, output layer, norms, biases, convolutions) goes to AdamW.

    Returns the Muon params, their block size along (rows, columns), and the AdamW params.
    A matrix is one block unless per_head splits the attention projections into heads:
    rows of wq / wk / wv / wr, columns of wo.
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


class Muon(Optimizer):
    """Muon for FSDP2 parameters sharded on dim 0, with AdamW for the rest.

    Every Muon matrix is a list of blocks (whole matrix, or attention heads). Each step:
    1. update the momentum on the local shard;
    2. all-to-all: every block goes, piece by piece, to the rank that owns it;
    3. the owner stacks its blocks by shape and runs one batched Newton-Schulz per shape;
    4. all-to-all back, then each rank applies the update to its shard.
    """

    def __init__(
        self,
        muon_params: List[nn.Parameter],
        block_sizes: List,
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
        self.blocks = self._plan(muon_params, block_sizes)

    def _rows(self, n: int, rank: int) -> Tuple[int, int]:
        """Rows of an n-row matrix held by `rank` (FSDP2 Shard(0) uses torch.chunk)."""
        chunk = math.ceil(n / self.world)
        return min(rank * chunk, n), min((rank + 1) * chunk, n)

    def _plan(self, params, block_sizes) -> List[Block]:
        self.shapes = [tuple(p.shape) for p in params]
        blocks = []
        for i, ((rows, cols), spec) in enumerate(zip(self.shapes, block_sizes)):
            dim, size = spec if spec is not None else (0, rows)
            n = (rows, cols)[dim]
            blocks += [Block(i, dim, s, min(size, n - s)) for s in range(0, n, size)]
        # balance owners by work, largest blocks first
        load = [0] * self.world
        for b in sorted(blocks, key=lambda b: -self._numel(b)):
            b.owner = min(range(self.world), key=lambda r: load[r])
            load[b.owner] += self._numel(b)
        return blocks

    def _numel(self, b: Block) -> int:
        rows, cols = self.shapes[b.param]
        return b.size * (cols if b.dim == 0 else rows)

    def _piece(self, b: Block, rank: int) -> Tuple[slice, slice, int]:
        """The part of block b stored on `rank`: (local rows, columns, offset of these rows in the block)."""
        rows, cols = self.shapes[b.param]
        r0, r1 = self._rows(rows, rank)
        if b.dim == 1:
            return slice(0, r1 - r0), slice(b.start, b.start + b.size), r0
        lo, hi = max(b.start, r0), min(b.start + b.size, r1)
        hi = max(lo, hi)
        return slice(lo - r0, hi - r0), slice(None), lo - b.start

    def _block_shape(self, b: Block) -> Tuple[int, int]:
        rows, cols = self.shapes[b.param]
        return (b.size, cols) if b.dim == 0 else (rows, b.size)

    def _scale(self, shape: Tuple[int, int]) -> float:
        fan_out, fan_in = shape
        if self.adjust_lr == "spectral_norm":
            return math.sqrt(fan_out / fan_in)
        if self.adjust_lr == "rms_norm":
            return 0.2 * math.sqrt(max(fan_out, fan_in))
        return 1.0

    def _piece_shape(self, b: Block, rank: int) -> Tuple[int, int]:
        rs, _, _ = self._piece(b, rank)
        rows, cols = self._block_shape(b)
        return rs.stop - rs.start, cols

    def _exchange(self, send: List[List[Tensor]], recv_shapes: List[List[Tuple[int, int]]]) -> List[List[Tensor]]:
        """all-to-all of bf16 pieces: send[r] goes to rank r; returns the pieces received from each rank."""
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

    @torch.no_grad()
    def _muon_step(self, group: Dict):
        params = group["params"]
        lr, wd = group["lr"], group["weight_decay"]
        mine = [b for b in self.blocks if b.owner == self.rank]

        # 1. momentum on the local shard -> bf16 update direction
        updates = []
        for p in params:
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
            m = _local(state["momentum"])
            # every rank must take part in the exchange, so a missing grad counts as zero
            g = _local(p.grad) if p.grad is not None else torch.zeros_like(m)
            m.mul_(self.momentum).add_(g)
            u = g + self.momentum * m if self.nesterov else m
            updates.append(u.to(torch.bfloat16))

        # 2. every block goes, piece by piece, to its owner
        send = [[] for _ in range(self.world)]
        for b in self.blocks:
            rs, cs, _ = self._piece(b, self.rank)
            send[b.owner].append(updates[b.param][rs, cs])
        recv = self._exchange(send, [[self._piece_shape(b, src) for b in mine] for src in range(self.world)])
        full = [torch.cat([recv[src][k] for src in range(self.world)]) for k in range(len(mine))]

        # 3. one batched Newton-Schulz per block shape (tall blocks are transposed)
        by_shape: Dict[Tuple[int, int], List[int]] = {}
        for k, x in enumerate(full):
            by_shape.setdefault((min(x.shape), max(x.shape)), []).append(k)
        for shape, idx in by_shape.items():
            tall = [full[k].shape != shape for k in idx]
            X = newton_schulz5(torch.stack([full[k].mT if t else full[k] for k, t in zip(idx, tall)]), group["eps"])
            for j, (k, t) in enumerate(zip(idx, tall)):
                full[k] = (X[j].mT if t else X[j]) * self._scale(tuple(full[k].shape))

        # 4. back to the shards, then apply
        send = [[] for _ in range(self.world)]
        for b, o in zip(mine, full):
            for dst in range(self.world):
                rs, _, off = self._piece(b, dst)
                send[dst].append(o[off:off + rs.stop - rs.start])
        recv = self._exchange(send, [[self._piece_shape(b, self.rank) for b in self.blocks if b.owner == src]
                                     for src in range(self.world)])
        received = [iter(r) for r in recv]
        for b in self.blocks:
            rs, cs, _ = self._piece(b, self.rank)
            updates[b.param][rs, cs] = next(received[b.owner])
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
