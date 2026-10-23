"""
In-memory copies of the training state for in-allocation recovery (ft).

Every k steps each rank packs its local training state on the GPU: parameters,
optimizer state and persistent buffers, byte for byte. It keeps one copy in a
RAM-backed slot on its own node and sends one to the rank G places ahead (the
next node), which holds it for this rank. Each kind of copy alternates between
two slots, A and B. A slot's metadata file is removed before the slot is
overwritten and written last, so a copy cut off halfway never hides the
previous one. The copy runs on its own CUDA stream, communicator and thread;
when any rank is still writing the previous copy, all ranks skip this one.
"""

import logging
import math
import os
import pickle
import threading
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from xllm.ft.sizing import GpuPeakRecorder

logger = logging.getLogger()


def run_epoch() -> Tuple[str, str]:
    """Identifies one run of the job: slots of another epoch (an earlier run before a requeue) are never used."""
    return os.environ.get("SLURM_JOB_ID", "local"), os.environ.get("SLURM_RESTART_COUNT", "0")


def slot_path(slot_dir: str, kind: str, rank: int, slot: int) -> Path:
    """kind is "own" (this rank's state) or "peer" (the state held for another rank); rank is the state's owner."""
    job, _ = run_epoch()
    return Path(slot_dir) / f"ft_{job}_{kind}_r{rank:06d}_s{slot}"


def meta_path(path: Path) -> Path:
    return path.with_suffix(".meta")


def local_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def state_tensors(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> List[torch.Tensor]:
    """The local tensors that make up a rank's training state, in a fixed order."""
    out = []
    for p in model.parameters():
        out.append(local_tensor(p))
        state = optimizer.state.get(p, {})
        out += [local_tensor(state[k]) for k in sorted(state) if isinstance(state[k], torch.Tensor)]
    for module in model.modules():
        for name, buf in module.named_buffers(recurse=False):
            if buf is not None and name not in module._non_persistent_buffers_set:
                out.append(local_tensor(buf))
    return out


def pack(tensors: List[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
    # some state lives on the CPU (e.g. AdamW's per-parameter step); restore copies it back in place
    flat = [t.detach().to(out.device).contiguous().reshape(-1).view(torch.uint8) for t in tensors]
    return torch.cat(flat, out=out[: sum(f.numel() for f in flat)])


class Replica:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        slot_dir: str,
        k: int,
        mtbf_sys: float,
        gpus_per_node: int,
        gpu_peak: GpuPeakRecorder,
    ):
        self.model, self.optimizer = model, optimizer
        self.slot_dir = slot_dir
        self.k, self.mtbf_sys, self.gpu_peak = k, mtbf_sys, gpu_peak
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        sizes = torch.tensor([gpus_per_node, -gpus_per_node], device="cuda")
        dist.all_reduce(sizes, op=dist.ReduceOp.MAX)
        if sizes[0].item() != -sizes[1].item():
            raise RuntimeError(f"ft: ranks disagree on the GPUs per node ({-sizes[1].item()}..{sizes[0].item()})")
        multi_node = self.world > gpus_per_node
        self.gpus_per_node = gpus_per_node
        self.peer_of = (self.rank + gpus_per_node) % self.world if multi_node else None
        self.peer_src = (self.rank - gpus_per_node) % self.world if multi_node else None
        self.group = dist.new_group(list(range(self.world)))  # never queued behind training collectives
        self.stream = torch.cuda.Stream()
        self.busy = False
        self.force_copy = False  # set by a memory restore: copy at the first allowed step
        self.restored_slot: Optional[int] = None  # the forced copy goes to the other slot
        self.thread: Optional[threading.Thread] = None
        self.gpu_mine = self.gpu_recv = self.host_mine = self.host_recv = None
        self.recv_nbytes = 0
        self.probe: Optional[Dict[str, float]] = None  # auto k: timings of the first copy
        Path(slot_dir).mkdir(parents=True, exist_ok=True)

    def maybe_copy(self, step: int, training_state: Callable[[], Dict[str, Any]]):
        """Call after the optimizer and scheduler steps. Collective over all ranks."""
        if not self.gpu_peak.copies_allowed:
            return
        forced = self.force_copy
        if forced:
            self.force_copy = False
        elif self.k == 0 and not self._set_auto_k(step):
            return
        elif self.k > 0 and step % self.k != 0:
            return
        busy = torch.tensor([int(self.busy)], device="cuda")
        dist.all_reduce(busy, op=dist.ReduceOp.MAX, group=self.group)
        if busy.item():
            logger.warning(f"ft: copy at step {step} skipped, the previous copy is still being written")
            return
        self._start_copy(step, training_state(), forced)

    def _set_auto_k(self, step: int) -> bool:
        """Auto k: copy once at the first allowed step, then k* = sqrt(2 C MTBF) / t from its measured cost."""
        if self.probe is None:
            self.probe = {"step": step, "t0": timer(), "copy_s": -1.0}
            return True
        busy = torch.tensor([int(self.busy)], device="cuda")  # every rank must take the same branch
        dist.all_reduce(busy, op=dist.ReduceOp.MAX, group=self.group)
        if busy.item():
            return False
        t = (timer() - self.probe["t0"]) / max(1, step - self.probe["step"])
        cost = torch.tensor([self.probe["copy_s"], t], device="cuda", dtype=torch.float64)
        dist.all_reduce(cost, op=dist.ReduceOp.MAX, group=self.group)
        c, t = cost.tolist()
        if c < 0:  # the probe copy failed on some rank: probe again
            self.probe = None
            return False
        self.k = max(1, int(math.sqrt(2 * c * self.mtbf_sys) / t))
        logger.info(f"ft: copy cost {c:.2f} s, step {t:.3f} s, MTBF {self.mtbf_sys:.0f} s -> copy every {self.k} steps")
        return False

    def prepare(self, nbytes: int):
        """Allocates the copy buffers and learns the size of the state this rank holds. Collective (pairs)."""
        if self.gpu_mine is None:
            self.gpu_mine = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            self.host_mine = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            if self.peer_of is not None:
                # shards can differ in size between ranks: learn the size of the state this rank will hold
                sizes = torch.tensor([nbytes, 0], device="cuda")
                dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, sizes[:1], self.peer_of, group=self.group),
                        dist.P2POp(dist.irecv, sizes[1:], self.peer_src, group=self.group),
                    ]
                )[-1].wait()
                torch.cuda.synchronize()
                self.recv_nbytes = int(sizes[1].item())
                self.gpu_recv = torch.empty(self.recv_nbytes, dtype=torch.uint8, device="cuda")
                self.host_recv = torch.empty(self.recv_nbytes, dtype=torch.uint8, pin_memory=True)

    def _start_copy(self, step: int, training_state: Dict[str, Any], forced: bool = False):
        if forced and self.restored_slot is not None:
            slot = 1 - self.restored_slot  # keep the restored copy until the new one is complete
        else:
            slot = (step // self.k) % 2 if self.k > 0 else 0
        tensors = state_tensors(self.model, self.optimizer)
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        self.prepare(nbytes)
        assert nbytes == self.gpu_mine.numel(), "the training state changed size"
        meta = pickle.dumps({"step": step, "epoch": run_epoch(), "rank": self.rank, "nbytes": nbytes,
                             "training_state": training_state})
        pack(tensors, self.gpu_mine)
        ready = torch.cuda.Event()
        ready.record()
        works, meta_recv = [], None
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            if self.peer_of is not None:
                # the metadata size changes between copies: exchange it first (before the payload is queued,
                # so reading it back does not wait for the device-to-host copy)
                meta_t = torch.frombuffer(bytearray(meta), dtype=torch.uint8).cuda()
                lens = torch.tensor([meta_t.numel(), 0], device="cuda")
                dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, lens[:1], self.peer_of, group=self.group),
                        dist.P2POp(dist.irecv, lens[1:], self.peer_src, group=self.group),
                    ]
                )[-1].wait()
                meta_recv = torch.empty(int(lens[1].item()), dtype=torch.uint8, device="cuda")
            self.host_mine.copy_(self.gpu_mine, non_blocking=True)
            if self.peer_of is not None:
                works = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, self.gpu_mine, self.peer_of, group=self.group),
                        dist.P2POp(dist.irecv, self.gpu_recv, self.peer_src, group=self.group),
                        dist.P2POp(dist.isend, meta_t, self.peer_of, group=self.group),
                        dist.P2POp(dist.irecv, meta_recv, self.peer_src, group=self.group),
                    ]
                )
            sent = torch.cuda.Event()
            sent.record(self.stream)
        self.busy = True
        self.thread = threading.Thread(
            target=self._write, args=(slot, meta, works, meta_recv, sent, timer()), daemon=True
        )
        self.thread.start()

    def _write(self, slot: int, meta: bytes, works, meta_recv, sent, t0: float):
        try:
            self._write_slots(slot, meta, works, meta_recv, sent, t0)
        except Exception as e:  # e.g. /dev/shm full: this copy is lost (its meta was removed first), the next one retries
            logger.error(f"ft: copy of step {pickle.loads(meta)['step']} failed: {e!r}")
        finally:
            self.busy = False

    def _write_slots(self, slot: int, meta: bytes, works, meta_recv, sent, t0: float):
        with torch.cuda.stream(self.stream):
            for w in works:
                w.wait()
            if self.peer_of is not None:
                self.host_recv.copy_(self.gpu_recv, non_blocking=True)
                meta_host = meta_recv.cpu()
            done = torch.cuda.Event()
            done.record(self.stream)
        sent.synchronize()
        done.synchronize()
        write_slot(slot_path(self.slot_dir, "own", self.rank, slot), self.host_mine, meta)
        if self.peer_of is not None:
            write_slot(slot_path(self.slot_dir, "peer", self.peer_src, slot), self.host_recv, bytes(meta_host.numpy()))
        if self.probe is not None and self.probe["copy_s"] < 0:
            self.probe["copy_s"] = timer() - t0
        if self.rank == 0:
            logger.info(f"ft: copy of step {pickle.loads(meta)['step']} written in {timer() - t0:.2f} s")

    def close(self):
        if self.thread is not None:
            self.thread.join()


def write_slot(path: Path, data: torch.Tensor, meta: bytes):
    mp = meta_path(path)
    mp.unlink(missing_ok=True)  # the slot is invalid from here until the new metadata is in place
    if not path.exists() or path.stat().st_size != data.numel():
        with open(path, "wb") as f:
            # reserve the space: on tmpfs a truncate does not, and a full /dev/shm would then kill the trainer with
            # SIGBUS in the copy below instead of failing here (the caller loses this copy and the next one retries)
            os.posix_fallocate(f.fileno(), 0, data.numel())
    torch.from_file(str(path), shared=True, size=data.numel(), dtype=torch.uint8).copy_(data)
    tmp = mp.with_suffix(".tmp")
    tmp.write_bytes(meta)
    os.replace(tmp, mp)


def read_meta(path: Path) -> Optional[Dict[str, Any]]:
    mp = meta_path(path)
    if not mp.exists():
        return None
    return pickle.loads(mp.read_bytes())


NO_COMMON_STEP_EXIT = 76  # a memory restore found no step that every rank holds: the agent restarts from disk


def replaced_nodes() -> List[int]:
    """Logical nodes whose state was lost this generation (set by the agent; their ranks restore from the holder)."""
    return [int(n) for n in os.environ.get("FT_REPLACED", "").split(",") if n]


def valid_steps(path_of_slot: Callable[[int], Path], rank: int) -> Dict[int, int]:
    """step -> slot index, for the slots whose metadata is complete and belongs to this run and this rank."""
    out = {}
    for slot in (0, 1):
        meta = read_meta(path_of_slot(slot))
        if meta is not None and meta["epoch"] == run_epoch() and meta["rank"] == rank:
            out[meta["step"]] = slot
    return out


@torch.no_grad()
def init_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Creates the optimizer state with its real shapes, sharding and devices (it is overwritten right after).
    Only trainable parameters get a gradient, so the state set matches the one copied."""
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def unpack(tensors: List[torch.Tensor], buf: torch.Tensor):
    off = 0
    for t in tensors:
        n = t.numel() * t.element_size()
        src = buf[off:off + n].to(t.device)
        if t.is_contiguous():
            t.view(-1).view(torch.uint8).copy_(src)
        else:  # through a contiguous buffer, in the order pack() wrote it
            t.copy_(src.view(t.dtype).view(t.shape))
        off += n
    assert off == buf.numel(), "the copy does not match the training state"


ANY_STEP = 2**62  # offered by a rank whose state comes from its holder: it takes whatever step the others agree on


def agree_step(candidates: Optional[Set[int]], reduce_min: Callable[[int], int]) -> int:
    """
    The newest step every rank holds, or -1. candidates: this rank's steps (None: any step). reduce_min: an
    all-reduce MIN over all ranks. A failed copy write can leave ranks with different steps, so a step counts only
    once every rank confirmed it holds it; else the ranks agree again below it. Collective.
    """
    bound = ANY_STEP + 1
    while True:
        mine = None if candidates is None else [s for s in candidates if s < bound]
        step = reduce_min(ANY_STEP if mine is None else max(mine, default=-1))
        if step < 0 or step == ANY_STEP:
            return -1
        if reduce_min(int(candidates is None or step in candidates)) == 1:
            return step
        bound = step


def restore(replica: Replica) -> Optional[Dict[str, Any]]:
    """
    After the process groups are rebuilt: agree on the newest step every rank holds, then load it into the
    model, optimizer and buffers. Returns this rank's non-tensor training state, or None when no common
    step exists (the caller exits with NO_COMMON_STEP_EXIT). Collective over all ranks.
    """
    rank, gpus = replica.rank, replica.gpus_per_node
    node_replaced = (rank // gpus) in replaced_nodes()
    own = valid_steps(lambda s: slot_path(replica.slot_dir, "own", rank, s), rank)
    candidates = set(own) if not node_replaced else None  # None: this rank's state comes from its holder
    holder_of = replica.peer_src is not None and (replica.peer_src // gpus) in replaced_nodes()
    if holder_of:  # this rank holds the copy of a rank whose node was replaced
        held = valid_steps(lambda s: slot_path(replica.slot_dir, "peer", replica.peer_src, s), replica.peer_src)
        candidates = set(held) if candidates is None else candidates & set(held)

    def reduce_min(value: int) -> int:
        t = torch.tensor([value], device="cuda", dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())

    step = agree_step(candidates, reduce_min)
    if step < 0:
        logger.error(f"ft: no in-memory step held by every rank (this rank holds {sorted(candidates or [])})")
        return None

    init_optimizer_state(replica.model, replica.optimizer)
    tensors = state_tensors(replica.model, replica.optimizer)
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    replica.prepare(nbytes)
    if holder_of:  # send the held copy and its metadata to the rank that now owns that state
        path = slot_path(replica.slot_dir, "peer", replica.peer_src, held[step])
        payload = torch.from_file(str(path), shared=False, size=path.stat().st_size, dtype=torch.uint8).cuda()
        meta = torch.frombuffer(bytearray(meta_path(path).read_bytes()), dtype=torch.uint8).cuda()
        sizes = torch.tensor([payload.numel(), meta.numel()], device="cuda")
        dist.send(sizes, replica.peer_src, group=replica.group)
        dist.send(payload, replica.peer_src, group=replica.group)
        dist.send(meta, replica.peer_src, group=replica.group)
    if node_replaced:
        sizes = torch.empty(2, dtype=torch.int64, device="cuda")
        dist.recv(sizes, replica.peer_of, group=replica.group)
        payload = torch.empty(int(sizes[0].item()), dtype=torch.uint8, device="cuda")
        meta_t = torch.empty(int(sizes[1].item()), dtype=torch.uint8, device="cuda")
        dist.recv(payload, replica.peer_of, group=replica.group)
        dist.recv(meta_t, replica.peer_of, group=replica.group)
        meta = pickle.loads(bytes(meta_t.cpu().numpy()))
    else:
        path = slot_path(replica.slot_dir, "own", rank, own[step])
        payload = torch.from_file(str(path), shared=False, size=path.stat().st_size, dtype=torch.uint8).cuda()
        meta = read_meta(path)
    assert meta["step"] == step and meta["rank"] == rank and meta["nbytes"] == nbytes == payload.numel()
    unpack(tensors, payload)
    replica.restored_slot = None if node_replaced else own[step]
    replica.force_copy = True  # after the first good step, copy at once: fresh copies everywhere
    logger.info(f"ft: restored step {step} from memory{' (state received from its holder)' if node_replaced else ''}")
    return meta["training_state"]
