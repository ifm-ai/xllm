"""AdamW with fp32 master weights and moment states in host memory (the ZeRO-Offload layout), for FSDP2 models
whose sharded params and grads are stored in bf16 on the GPU.

After each backward, accumulate() copies the bf16 grad shards to pinned host memory and adds them into fp32
accumulators, so accumulation over micro-batches is exact and the GPU holds one micro-batch of grads. The copy runs
on a side stream and the add on a host thread; with several micro-batches per step both overlap the next one.
Per optimizer step: clip_grad_norm_() takes the norm of the host accumulators (one all-reduce), step() runs the
fused CPU AdamW kernel on the fp32 masters and copies the updated shards back to the GPU in bf16 once.
Host memory per param: 4 (master) + 8 (moments) + 4 (accumulator) + 2 (pinned bf16 staging) bytes.
The host AdamW uses torch's intra-op threads: torchrun sets OMP_NUM_THREADS=1 unless it is set.
"""
import logging
import threading

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xllm.distributed.initialize import get_model_parallel_group, get_model_parallel_world_size

logger = logging.getLogger()


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class CPUMasterAdamW(torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        self.gpu_params = [p for p in params if p.requires_grad]
        locals_ = [_local(p) for p in self.gpu_params]
        assert all(l.dtype == torch.bfloat16 for l in locals_), "optimizer offload expects bf16 sharded params"
        masters = [l.detach().to("cpu", torch.float32) for l in locals_]
        super().__init__(masters, fused=True, **kwargs)
        self.masters = masters
        self.acc = [torch.zeros_like(m) for m in masters]
        # one pinned buffer: per-tensor pinned allocations are rounded up to powers of two by the host allocator
        flat = torch.empty(sum(l.numel() for l in locals_), dtype=torch.bfloat16, pin_memory=True)
        self.stage = [st.view(l.shape) for st, l in zip(torch.split(flat, [l.numel() for l in locals_]), locals_)]
        self.got_grad = [False] * len(masters)
        self.copy_stream = torch.cuda.Stream()
        self.adder, self.adder_error = None, None
        dts = [p for p in self.gpu_params if isinstance(p, DTensor)]
        if dts:
            mesh, placements = dts[0].device_mesh, dts[0].placements
            assert all(p.device_mesh == mesh and p.placements == placements for p in dts), \
                "optimizer offload reduces the grad norm over one mesh; params are sharded on several"
            self.shard_groups = [mesh.get_group(d) for d, pl in enumerate(placements) if pl.is_shard()]
        else:
            self.shard_groups = []
        if torch.get_num_threads() == 1:
            logger.warning("optimizer offload: torch uses 1 CPU thread (torchrun sets OMP_NUM_THREADS=1); "
                           "set OMP_NUM_THREADS to the cores per rank or the host AdamW runs on one core")

    @torch.no_grad()
    def accumulate(self):
        """Move this backward's GPU grads into the fp32 host accumulators and free them on the GPU.

        Returns at once: the copies run on a side stream and the fp32 adds on a host thread. The previous adds
        finish first, since they read the same staging buffers.
        """
        self._drain()
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        live = []
        with torch.cuda.stream(self.copy_stream):
            for i, (p, st, a) in enumerate(zip(self.gpu_params, self.stage, self.acc)):
                if p.grad is not None:
                    g = _local(p.grad)
                    st.copy_(g, non_blocking=True)
                    g.record_stream(self.copy_stream)  # the allocator keeps g until the copy is done
                    p.grad = None
                    self.got_grad[i] = True
                    live.append((st, a))
        done = torch.cuda.Event()
        done.record(self.copy_stream)

        def add():
            try:
                done.synchronize()
                for st, a in live:
                    a.add_(st)
            except BaseException as e:  # re-raised by _drain() on the training thread
                self.adder_error = e

        self.adder = threading.Thread(target=add)
        self.adder.start()

    def _drain(self):
        if self.adder is not None:
            self.adder.join()
            self.adder = None
        if self.adder_error is not None:
            e, self.adder_error = self.adder_error, None
            raise RuntimeError("optimizer offload: grad accumulation on the host failed") from e

    @torch.no_grad()
    def _push_masters(self):
        # fp32 masters -> bf16 staging -> GPU shards; the sync keeps the staging buffers free for accumulate()
        for m, st, p in zip(self.masters, self.stage, self.gpu_params):
            st.copy_(m)
            _local(p).copy_(st, non_blocking=True)
        torch.cuda.current_stream().synchronize()

    @torch.no_grad()
    def sync_masters_from_params(self):
        """Take the masters from the GPU params, after the model's weights were loaded from a checkpoint."""
        self._drain()
        for m, p in zip(self.masters, self.gpu_params):
            m.copy_(_local(p))

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> float:
        assert norm_type == 2.0, "optimizer offload clips by the 2-norm only"
        self._drain()
        sq = sum(torch.linalg.vector_norm(a).double() ** 2 for a in self.acc)
        total = torch.as_tensor(sq, dtype=torch.float64).to(torch.cuda.current_device())
        for g in self.shard_groups:
            dist.all_reduce(total, group=g)
        if get_model_parallel_world_size() > 1:
            dist.all_reduce(total, group=get_model_parallel_group())
        total = total.sqrt().item()
        coef = float(max_norm) / (total + 1e-6)
        if coef < 1.0:
            torch._foreach_mul_(self.acc, coef)
        return total

    @torch.no_grad()
    def step(self, closure=None):
        self._drain()
        # params that got no grad since the last step are skipped, as AdamW skips params whose grad is None
        for m, a, got in zip(self.masters, self.acc, self.got_grad):
            m.grad = a if got else None
        loss = super().step(closure)
        self._push_masters()
        torch._foreach_zero_(self.acc)
        self.got_grad = [False] * len(self.masters)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        # the host accumulators are zeroed by step(); the GPU grads are freed by accumulate()
        for p in self.gpu_params:
            p.grad = None

    def state_dict(self):
        self._drain()
        sd = super().state_dict()
        sd["masters"] = self.masters
        return sd

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """Restore a state saved by this class, or a plain AdamW state of the same params (its moments are
        DTensors on the GPU; the masters then come from sync_masters_from_params(), i.e. the bf16 weights)."""
        self._drain()
        state_dict = dict(state_dict)
        masters = state_dict.pop("masters", None)
        state_dict["state"] = {k: {n: _local(v) for n, v in s.items()} for k, s in state_dict["state"].items()}
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group["fused"] = True  # load_state_dict takes the saved groups; a plain AdamW saved fused=None
        if masters is not None:
            for m, s in zip(self.masters, masters):
                m.copy_(s)
            self._push_masters()
