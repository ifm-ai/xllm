"""Muon on FSDP2 with per-head orthogonalization; AdamW for the parameters it does not take.

Modules register their Muon weights in __init__ (muon_spec.register_muon); build_muon hands them to Muon.
For one Muon weight W [rows, cols]:

- Block: a box of W that Newton-Schulz orthogonalizes on its own (the whole weight, one head, one expert);
  a weight's Blocks tile it. From the model (register_muon), not from the sharding.
- region: the box of W that one rank stores (FSDP2 Shard(0) of its TP-local chunk: contiguous rows); the
  all-to-alls run over the dp x TP plane.
- slice: the overlap of a Block with one rank's region.
- owner: the rank that assembles a Block from its slices and runs Newton-Schulz on it: the rank storing
  most of it, then the least loaded. A Block whose only slice is its owner's is local: no communication.

Example: wk [8, 4] in 2 row Blocks of 4 rows on 3 ranks (torch.chunk: rows 0-2, 3-5, 6-7):

    rank 0  rows 0-2  a          block 0 = rows 0-3 = cat(a, b1), owner rank 0 (stores 3 rows)
    rank 1  rows 3-5  b1 | b2    block 1 = rows 4-7 = cat(b2, c), owner rank 1 (tie with rank 2)
    rank 2  rows 6-7  c          b1 and b2 are rank 1's slices of blocks 0 and 1
"""
import math
from collections import Counter, defaultdict
from logging import getLogger

import torch
import torch.distributed as dist
import xllm_extension.ops as xllm_ops
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.adamw import adamw

from xllm.config import OptimConf

from .muon_spec import Block, MuonSpec, box_index, name_pattern, split_muon_params

logger = getLogger()

# Newton-Schulz steps X <- a X + b (X X^T) X + c (X X^T)^2 X: each applies p(s) = a s + b s^3 + c s^5 to the singular
# values. Polar Express (arXiv 2505.16932): each p is the minimax fit of 1 on the range the previous steps leave,
# from [1e-3, 1.01]; fitted above a cushion of 0.05 of the range's top (so no p dips near 0 inside it, where rounding
# would flip a sign) and 1% above it. Remez in quad precision; in exact arithmetic s in [1e-3, 1] ends within 3e-7 of 1.
NS_COEFFS = (
    (6.6855200767111675, -17.811269767700782, 12.619541303795451),
    (4.016121039752612, -3.861087728177999, 0.9871918227003456),
    (4.016121039752612, -3.861087728177999, 0.9871918227003456),
    (3.795262478265412, -3.5559258715900817, 0.9017288960572022),
    (2.3593913208894888, -1.696737173296388, 0.4096643411881119),
    (1.8910694529512306, -1.2577697543810034, 0.3672643550538599),
    (1.8657070833564235, -1.23145450650965, 0.36574763500927243),
)


def newton_schulz(X: Tensor, eps: float) -> Tensor:
    """U S V^T -> U V^T for a stack of matrices [B, m, n], in fp16, in X's layout.

    - The steps run on the smaller Gram matrix: A = X X^T and X <- a X + B X, or for a tall X, A = X^T X and
      X <- a X + X B (B = b A + c A^2 is symmetric), so a tall X is never transposed into a copy.
    - X is scaled by s = ||A^2||_F^(1/4) >= s_max, 5% more for rounding: within k^(1/8) of s_max where ||X||_F is
      within k^(1/2) (k = min(m, n)), so the small singular values start higher. s comes from the first step's A^2
      and folds into that step: p(X / s) = (a X + (b / s^2 A + c / s^4 A^2) X) / s.
    - The first step runs in bf16, whose range holds any X; the others in fp16, whose 3 more mantissa bits bring the
      singular values to 1 within ~1e-3. The sums run in cuBLAS epilogues (baddbmm's alpha / beta), in fp32."""
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


def group_hparams(cfg: OptimConf, algorithm: str) -> dict:
    """Param-group hyperparameters that OptimConf sets for "muon" or "adamw" (single source of truth)."""
    if algorithm == "muon":
        wd = cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay
        return {"momentum": cfg.muon_momentum, "nesterov": cfg.muon_nesterov, "adjust_lr": cfg.muon_adjust_lr,
                "block_scale": cfg.muon_block_scale, "weight_decay": wd}
    return {"betas": (cfg.beta1, cfg.beta2), "weight_decay": cfg.weight_decay}


class Muon(Optimizer):
    """Muon for FSDP2 parameters sharded on dim 0, with AdamW for the rest. Each step: momentum on every shard into
    bf16 update directions u; an all-to-all sends the remote blocks' slices to their owners while the local blocks
    are orthogonalized (one buffer per shape at a time); the owners orthogonalize the remote blocks and an
    all-to-all sends the slices back into u; each rank applies its update."""

    def __init__(self, muon_params: list[nn.Parameter], specs: list[MuonSpec], adamw_params: list[nn.Parameter],
                 mesh: DeviceMesh, cfg: OptimConf, tp_group: dist.ProcessGroup | None = None):
        """specs[i]: muon_params[i]'s Blocks (register_muon); tp_group: the TP group, if any (params are then TP-local)."""
        groups = [dict(params=muon_params, lr=cfg.muon_lr, algorithm="muon", **group_hparams(cfg, "muon")),
                  dict(params=adamw_params, lr=cfg.lr, algorithm="adamw", **group_hparams(cfg, "adamw"))]
        super().__init__(groups, {"eps": cfg.epsilon})
        self._ranks(mesh, tp_group)
        self.u = [torch.empty_like(_local(p), dtype=torch.bfloat16) for p in muon_params]  # the shards' update directions
        self._plan([spec.global_shape(tuple(p.shape), self.tp) for spec, p in zip(specs, muon_params)], specs)

    def _ranks(self, mesh: DeviceMesh, tp_group: dist.ProcessGroup | None):
        """The process group over the dp x TP plane that shards a weight, my rank in it, each rank's (dp index, tp index)."""
        self.tp = 1 if tp_group is None else dist.get_world_size(tp_group)
        self.dp = mesh.size()
        if tp_group is None:
            self.group, self.rank = mesh.get_group(), mesh.get_local_rank()
            self.coords = [(r, 0) for r in range(self.dp)]
        else:
            every = [None] * dist.get_world_size()  # [g]: the dp ranks and the tp ranks of global rank g
            dist.all_gather_object(every, (dist.get_process_group_ranks(mesh.get_group()),
                                           dist.get_process_group_ranks(tp_group)))
            plane = lambda g: sorted(h for d in every[g][0] for h in every[d][1])
            for ranks in sorted({tuple(plane(g)) for g in range(len(every))}):  # every rank creates every group
                group = dist.new_group(list(ranks))
                if dist.get_rank() in ranks:
                    self.group = group
            mine = plane(dist.get_rank())
            self.rank = mine.index(dist.get_rank())
            self.coords = [(every[g][0].index(g), every[g][1].index(g)) for g in mine]
        self.world = len(self.coords)

    def _plan(self, shapes: list[tuple[int, int]], specs: list[MuonSpec]):
        """From the whole weights' shapes, identical on every rank: blocks, with param and owner; fans[mode][i],
        (fan_out, fan_in) of weight i's blocks or of the weight. Then this rank's views of u and buffers: local, the
        local blocks I own by shape; remote, the blocks I own with other ranks' slices, in buffers by shape; mine,
        my slices of the remote blocks, rank by rank, packed into mine_buf for the all-to-all; theirs, the ranks'
        slices of the blocks I own, unpacked from theirs_buf the same way."""
        self.fans = {mode: [spec.fans(shape, per_block=mode == "block") for spec, shape in zip(specs, shapes)]
                     for mode in ("block", "matrix")}
        self.blocks: list[Block] = [b for i, (spec, shape) in enumerate(zip(specs, shapes)) for b in spec.blocks(shape, i)]
        regions = [spec.regions(shape, self.coords, self.dp, self.tp) for spec, shape in zip(specs, shapes)]
        slices = [[(r, box) for r, region in enumerate(regions[b.param]) if (box := b.overlap(region))]
                  for b in self.blocks]  # [k]: (rank, box) for the ranks that store part of block k
        load = [0] * self.world
        for b, held in sorted(zip(self.blocks, slices), key=lambda bh: -bh[0].cost):  # the costly blocks first
            stored = {r: (r1 - r0) * (c1 - c0) for r, (r0, r1, c0, c1) in held}
            b.owner = min(stored, key=lambda r: (-stored[r], load[r]))
            load[b.owner] += b.cost
            b.local = len(held) == 1
        self.has_remote = not all(b.local for b in self.blocks)  # then every rank joins the all-to-alls
        owned = [b for b in self.blocks if b.owner == self.rank and not b.local]
        views, self.remote = _pack([b.shape for b in owned])
        slot = {id(b): view for b, view in zip(owned, views)}
        local = defaultdict(list)
        mine = [[] for _ in range(self.world)]
        theirs = [[] for _ in range(self.world)]
        for b, held in zip(self.blocks, slices):
            for r, box in held:
                if b.local and r == self.rank:
                    local[b.shape].append(self.u[b.param][box_index(box, regions[b.param][r])])
                elif not b.local:
                    if r == self.rank:
                        mine[b.owner].append(self.u[b.param][box_index(box, regions[b.param][r])])
                    if b.owner == self.rank:
                        theirs[r].append(slot[id(b)][box_index(box, b.box)])
        self.local = list(local.items())
        self.mine, self.mine_buf, self.mine_packed, self.mine_sizes = _buffer(mine)
        self.theirs, self.theirs_buf, self.theirs_packed, self.theirs_sizes = _buffer(theirs)

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

    def _muon_step(self, group: dict):
        eps = group["eps"]
        self._momentum(group)
        gather = None
        if self.has_remote:  # my slices of the remote blocks go to their owners while the local blocks are done
            xllm_ops.batched_copy_(self.mine_packed, self.mine)
            gather = dist.all_to_all_single(self.theirs_buf, self.mine_buf, self.theirs_sizes, self.mine_sizes,
                                            group=self.group, async_op=True)
        for shape, views in self.local:  # one buffer per shape
            X = torch.empty(len(views), *shape, dtype=torch.bfloat16, device="cuda")
            xllm_ops.batched_copy_(list(X), views)
            X.copy_(newton_schulz(X, eps))
            xllm_ops.batched_copy_(views, list(X))
        if gather is not None:
            gather.wait()  # the stream waits for the collective; the CPU goes on
            xllm_ops.batched_copy_(self.theirs, self.theirs_packed)  # the remote blocks I own, assembled
            for X in self.remote:
                X.copy_(newton_schulz(X, eps))
            xllm_ops.batched_copy_(self.theirs_packed, self.theirs)  # and back the way they came
            dist.all_to_all_single(self.mine_buf, self.theirs_buf, self.mine_sizes, self.theirs_sizes, group=self.group)
            xllm_ops.batched_copy_(self.mine, self.mine_packed)
        self._update(group)

    def _momentum(self, group: dict):
        """m <- mu m + g on the shards (fp32); the bf16 directions (Nesterov: g + mu m) into u. No gradient: zeros,
        no state, no decay (as with AdamW); the plan stays fixed."""
        params = group["params"]
        for i, p in enumerate(params):
            if p.grad is None:
                self.u[i].zero_()
            elif "momentum" not in self.state[p]:
                self.state[p]["momentum"] = torch.zeros_like(p, dtype=torch.float32)
        live = [i for i, p in enumerate(params) if p.grad is not None]
        xllm_ops.muon_momentum_([_local(self.state[params[i]]["momentum"]) for i in live],
                                [_local(params[i].grad) for i in live], [self.u[i] for i in live],
                                group["momentum"], group["nesterov"])

    def _update(self, group: dict):
        """W <- W - lr (wd W + scale u) for the params with a gradient, in fp32. The scale multiplies the update, not
        the decay: from the fans of the block, or of the whole weight (block_scale)."""
        params, fans = group["params"], self.fans[group["block_scale"]]
        live = [i for i, p in enumerate(params) if p.grad is not None]
        xllm_ops.muon_update_([_local(params[i]) for i in live], [self.u[i] for i in live],
                              [_scale(*fans[i], group["adjust_lr"]) for i in live], group["lr"], group["weight_decay"])

    def _adamw_step(self, group: dict):
        """torch.optim.AdamW's update (foreach) on the group's params."""
        params = [p for p in group["params"] if p.grad is not None]
        states = [self.state[p] for p in params]
        for p, state in zip(params, states):
            if "step" not in state:
                state.update(step=torch.zeros((), dtype=torch.float32), exp_avg=torch.zeros_like(p, dtype=torch.float32),
                             exp_avg_sq=torch.zeros_like(p, dtype=torch.float32))
        adamw(params, [p.grad for p in params], [s["exp_avg"] for s in states], [s["exp_avg_sq"] for s in states],
              [], [s["step"] for s in states], foreach=True, amsgrad=False, beta1=group["betas"][0],
              beta2=group["betas"][1], lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
              maximize=False)


def _scale(fan_out: int, fan_in: int, adjust_lr: str | None) -> float:
    """sqrt(fan_out / fan_in) for adjust_lr spectral_norm, 0.2 sqrt(max(fan_out, fan_in)) for rms_norm, else 1."""
    if adjust_lr == "spectral_norm":
        return math.sqrt(fan_out / fan_in)
    return 0.2 * math.sqrt(max(fan_out, fan_in)) if adjust_lr == "rms_norm" else 1.0


def _local(t: Tensor) -> Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _pack(shapes: list[tuple[int, int]]) -> tuple[list[Tensor], list[Tensor]]:
    """One bf16 [B, R, C] buffer per shape; returns a [R, C] view per entry, and the buffers."""
    buffers = {shape: torch.empty(n, *shape, dtype=torch.bfloat16, device="cuda") for shape, n in Counter(shapes).items()}
    parts = {shape: iter(buffer.unbind(0)) for shape, buffer in buffers.items()}
    return [next(parts[shape]) for shape in shapes], list(buffers.values())


def _buffer(views: list[list[Tensor]]) -> tuple[list[Tensor], Tensor, list[Tensor], list[int]]:
    """views[r]: the views exchanged with rank r. Returns them flat; one bf16 buffer for them all, rank after rank; a
    view of it per view; and the per-rank sizes, for all_to_all_single."""
    flat = [t for ts in views for t in ts]
    buf = torch.empty(sum(t.numel() for t in flat), dtype=torch.bfloat16, device="cuda")
    packed = [part.view(t.shape) for part, t in zip(buf.split([t.numel() for t in flat]), flat)]
    return flat, buf, packed, [sum(t.numel() for t in ts) for ts in views]


def build_muon(model: nn.Module, cfg: OptimConf) -> Optimizer:
    """The model must be wrapped with FSDP2; Muon works over the shard dimension of the mesh its parameters live on."""
    from xllm.distributed import get_model_parallel_group, get_model_parallel_world_size
    from xllm.modules.model_parallel import ColumnParallelLinear, RowParallelLinear
    tp = get_model_parallel_world_size()
    muon_params, specs, names, adamw_params = split_muon_params(model)
    if not muon_params or not hasattr(muon_params[0], "device_mesh"):
        raise ValueError("Muon requires a model wrapped with FSDP2 and at least one registered weight.")
    for name, (spec, p) in dict(zip(names, zip(specs, muon_params))).items():  # one line per kind of weight
        fan_out, fan_in = spec.fans(spec.global_shape(tuple(p.shape), tp), per_block=True)
        logger.info(f"Muon {name} {list(p.shape)} {str(spec)!r}: {len(spec.blocks(tuple(p.shape)))} blocks of "
                    f"fan_out {fan_out} x fan_in {fan_in}, tp_dim {spec.tp_dim}")
    muon_ids = {id(p) for p in muon_params}  # 2-D weights of parallel layers in the blocks left to AdamW: say which
    missed = Counter(name_pattern(f"{fqn}.weight") for fqn, m in model.named_modules() if "layers." in fqn
                     and isinstance(m, (ColumnParallelLinear, RowParallelLinear)) and id(m.weight) not in muon_ids)
    if missed:
        logger.warning(f"Muon: unregistered weights use AdamW: {dict(missed)}")
    mesh = muon_params[0].device_mesh
    optimizer = Muon(muon_params, specs, adamw_params, mesh if mesh.ndim == 1 else mesh["dp"], cfg,
                     tp_group=get_model_parallel_group() if tp > 1 else None)
    numel = lambda ps: sum(p.numel() for p in ps)
    logger.info(f"Muon: {len(muon_params)} matrices / {len(optimizer.blocks)} blocks / {numel(muon_params)} params; "
                f"AdamW: {len(adamw_params)} tensors / {numel(adamw_params)} params")
    return optimizer
