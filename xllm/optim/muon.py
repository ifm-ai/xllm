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

Example: wk [8, 4] in 2 row Blocks of 4 rows on 3 ranks (torch.chunk: rows 0-2, 3-5, 6-7):

    rank 0  rows 0-2  a          block 0 = rows 0-3 = cat(a, b1), owner rank 0 (stores 3 rows)
    rank 1  rows 3-5  b1 | b2    block 1 = rows 4-7 = cat(b2, c), owner rank 1 (tie with rank 2)
    rank 2  rows 6-7  c          b1 and b2 are rank 1's slices of blocks 0 and 1
"""
import math
from collections import defaultdict
from logging import getLogger
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.adamw import adamw

from xllm.config import OptimConf
from .muon_spec import Block, MuonSpec, muon_spec, register_muon


logger = getLogger()

# Newton-Schulz 5 steps with per-step coefficients (microsoft/dion's zeropower_via_newtonschulz5; dion's default is Polar Express)
NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

Box = Tuple[int, int, int, int]  # rows [r0, r1) x cols [c0, c1) of a matrix



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
        specs: List[MuonSpec],
        adamw_params: List[nn.Parameter],
        mesh: DeviceMesh,
        cfg: OptimConf,
    ):
        """specs[i]: how muon_params[i] maps to Newton-Schulz matrices (register_muon / muon_spec)."""
        defaults = dict(weight_decay=cfg.weight_decay, eps=cfg.epsilon)
        super().__init__(
            [
                dict(params=muon_params, lr=cfg.muon_lr, algorithm="muon",
                     momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
                     adjust_lr=cfg.muon_adjust_lr, block_scale=cfg.muon_block_scale,
                     weight_decay=cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay),
                dict(params=adamw_params, lr=cfg.lr, algorithm="adamw", betas=(cfg.beta1, cfg.beta2)),
            ],
            defaults,
        )
        self.process_group = mesh.get_group()
        self.world = mesh.size()
        self.rank = mesh.get_local_rank()
        self._plan([tuple(p.shape) for p in muon_params], specs)

    def load_state_dict(self, state_dict):
        """Restore checkpoint hyperparameters; fill metadata absent from legacy checkpoints."""
        migrated = dict(state_dict)
        migrated["param_groups"] = []
        for saved in state_dict["param_groups"]:
            group = dict(saved)
            # Older Muon checkpoints did not serialize these hyperparameters.
            configured = next(g for g in self.param_groups if g["algorithm"] == group["algorithm"])
            keys = ("momentum", "nesterov", "adjust_lr", "block_scale") if group["algorithm"] == "muon" else ("betas",)
            missing = [key for key in keys if key not in group]
            if missing:
                logger.warning(f"Legacy Muon {group['algorithm']} checkpoint lacks {missing}; using configured values.")
            for key in missing:
                group[key] = configured[key]
            migrated["param_groups"].append(group)
        return super().load_state_dict(migrated)

    # ---- plan: built once from the global shapes, identical on every rank ----

    def _plan(self, shapes: List[Tuple[int, int]], specs: List[MuonSpec]):
        # Geometry is fixed; scale policy comes from the current param group,
        # including after a checkpoint restore or an explicit config override.
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
        regions = [self._regions(shape) for shape in shapes]
        # slices[k]: (rank, box in global coordinates) for every rank that stores part of block k
        slices = []
        for b in self.blocks:
            block = (b.r0, b.r1, b.c0, b.c1)
            overlaps = [(r, _overlap(block, region)) for r, region in enumerate(regions[b.param])]
            slices.append([(r, box) for r, box in overlaps if box is not None])

        # owner: the rank that stores most of the block (no transfer for that part), then the least
        # loaded; blocks are placed by decreasing Newton-Schulz cost m * n * min(m, n)
        load = [0] * self.world
        for k in sorted(range(len(self.blocks)), key=lambda k: -self._ns_cost(self.blocks[k])):
            stored = dict.fromkeys(range(self.world), 0)
            for r, (r0, r1, c0, c1) in slices[k]:
                stored[r] = (r1 - r0) * (c1 - c0)
            b = self.blocks[k]
            b.owner = min(range(self.world), key=lambda r: (-stored[r], load[r]))
            load[b.owner] += self._ns_cost(b)

        # my_slices[r]: my slices of the blocks rank r owns, as (matrix, box of my region)
        # owned_slices[r]: rank r's slices of the blocks I own, as (index in self.owned, box of the block)
        # Both lists follow the block order, so a sender's list matches its receiver's.
        self.owned = [k for k, b in enumerate(self.blocks) if b.owner == self.rank]
        index = {k: j for j, k in enumerate(self.owned)}
        self.my_slices = [[] for _ in range(self.world)]
        self.owned_slices = [[] for _ in range(self.world)]
        for k, b in enumerate(self.blocks):
            for r, box in slices[k]:
                if r == self.rank:
                    self.my_slices[b.owner].append((b.param, _local_box(box, regions[b.param][r])))
                if b.owner == self.rank:
                    self.owned_slices[r].append((index[k], _local_box(box, (b.r0, b.r1, b.c0, b.c1))))

    def _regions(self, shape: Tuple[int, int]) -> List[Box]:
        """The box of a matrix that each rank stores: FSDP2 Shard(0) follows torch.chunk
        (contiguous rows, all columns; trailing ranks may get none)."""
        rows, cols = shape
        step = math.ceil(rows / self.world)
        return [(min(r * step, rows), min((r + 1) * step, rows), 0, cols) for r in range(self.world)]

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

    def _momentum(self, p: Tensor, group: Dict) -> Tensor:
        """Momentum update on the local shard; returns the bf16 update direction."""
        # Keep the fixed communication plan, but do not create or decay state for
        # a parameter without a gradient (the same semantics as AdamW).
        if p.grad is None:
            return torch.zeros_like(_local(p), dtype=torch.bfloat16)
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
        m = _local(state["momentum"])
        g = _local(p.grad)
        m.mul_(group["momentum"]).add_(g)
        u = g + group["momentum"] * m if group["nesterov"] else m
        return u.to(torch.bfloat16)

    def _gather(self, updates: List[Tensor]) -> List[Tensor]:
        """Send my slices to their blocks' owners; return the whole blocks I own."""
        send = [[updates[i][box] for i, box in self.my_slices[r]] for r in range(self.world)]
        recv = self._exchange(send, [[_shape(box) for _, box in self.owned_slices[r]] for r in range(self.world)])
        blocks = [torch.empty(self.blocks[k].shape, dtype=torch.bfloat16, device="cuda") for k in self.owned]
        for r in range(self.world):
            for (j, box), t in zip(self.owned_slices[r], recv[r]):
                blocks[j][box] = t
        return blocks

    def _scatter(self, results: List[Tensor], updates: List[Tensor]):
        """Send each rank its slices of the blocks I own; write the slices I receive into `updates`."""
        send = [[results[j][box] for j, box in self.owned_slices[r]] for r in range(self.world)]
        recv = self._exchange(send, [[_shape(box) for _, box in self.my_slices[r]] for r in range(self.world)])
        for r in range(self.world):
            for (i, box), t in zip(self.my_slices[r], recv[r]):
                updates[i][box] = t

    @torch.no_grad()
    def _muon_step(self, group: Dict):
        params, lr, wd = group["params"], group["lr"], group["weight_decay"]
        updates = [self._momentum(p, group) for p in params]                      # 1. local shards
        gathered = self._gather(updates)                                   # 2. whole owned blocks
        self._scatter(orthogonalize(gathered, group["eps"]), updates)      # 3. Newton-Schulz, 4. back
        scales = [self._scale(*fans, group["adjust_lr"]) for fans in self._fans[group["block_scale"]]]
        for p, u, scale in zip(params, updates, scales):              # 5. scaled update in fp32
            if p.grad is None:
                continue
            x = _local(p)
            x.mul_(1 - lr * wd).add_(u.to(x.dtype), alpha=-lr * scale)

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
    missed = defaultdict(int)
    for fqn, m in model.named_modules():
        for name in getattr(m, "tp_dims", None) or {}:
            p = getattr(m, name, None)
            if "layers." in fqn and isinstance(p, nn.Parameter) and p.dim() == 2 and id(p) not in muon_ids:
                missed[_pattern(f"{fqn}.{name}")] += 1
    if missed:
        logger.warning(f"Muon: unregistered weights use AdamW: {dict(missed)}")


def build_muon(model: nn.Module, mesh: DeviceMesh, cfg: OptimConf) -> Optimizer:
    """`mesh` is the FSDP mesh; Muon works over its shard dimension."""
    muon_params, specs, names, adamw_params = split_muon_params(model)
    lines = []
    for name, spec, p in zip(names, specs, muon_params):
        fan_out, fan_in = spec.fans(tuple(p.shape), per_block=True)
        lines.append(f"Muon {name} {list(p.shape)} {str(spec)!r}: {len(spec.blocks(tuple(p.shape)))} blocks of "
                     f"fan_out {fan_out} x fan_in {fan_in}, tp_dim {spec.tp_dim}")
    for line in dict.fromkeys(lines):  # one line per kind of weight, e.g. layers.*.attention.wq.weight
        logger.info(line)
    _warn_unregistered(model, {id(p) for p in muon_params})
    shard_mesh = mesh if mesh.ndim == 1 else mesh["dp"]
    optimizer = Muon(muon_params, specs, adamw_params, shard_mesh, cfg)
    logger.info(f"Muon: {len(muon_params)} matrices / {len(optimizer.blocks)} blocks (scale={cfg.muon_block_scale}) / "
                f"{sum(p.numel() for p in muon_params)} params; AdamW: {len(adamw_params)} tensors / "
                f"{sum(p.numel() for p in adamw_params)} params")
    return optimizer
