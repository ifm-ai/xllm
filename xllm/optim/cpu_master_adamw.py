"""AdamW with fp32 master weights and moment states in host memory (the ZeRO-Offload layout), for FSDP2 models
whose sharded params and grads stay in bf16 on the GPU.

Per micro-batch, accumulate() copies the bf16 grad shards to pinned host buffers and adds them into fp32
accumulators, so accumulation over micro-batches is exact and the GPU only ever holds one micro-batch of grads.
The copy (side stream) and the add (host thread) overlap the next micro-batch's forward and backward.
Per optimizer step: clip_grad_norm_() takes the norm of the host accumulators (one NCCL all-reduce), step() runs the
fused CPU AdamW kernel on the fp32 masters and copies the updated shards back to the GPU in bf16 once.
Host memory per param: 4 (master) + 8 (moments) + 4 (accumulator) + 2 (pinned bf16 staging) bytes.
"""
import threading

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xllm.distributed.initialize import get_model_parallel_group, get_model_parallel_world_size


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
        self.stage = [torch.empty(l.shape, dtype=l.dtype, pin_memory=True) for l in locals_]
        for m, a in zip(masters, self.acc):
            m.grad = a
        self.copy_stream = torch.cuda.Stream()
        self.adder = None
        dt = next((p for p in self.gpu_params if isinstance(p, DTensor)), None)
        self.shard_groups = [] if dt is None else [
            dt.device_mesh.get_group(d) for d, pl in enumerate(dt.placements) if pl.is_shard()
        ]

    @torch.no_grad()
    def accumulate(self):
        """Move this micro-batch's GPU grads into the fp32 host accumulators and free them on the GPU.

        Returns at once: the copies run on a side stream and the fp32 adds on a host thread, both overlapping the
        next micro-batch. The previous micro-batch's adds finish first, since they read the same staging buffers.
        """
        self._drain()
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        live = []
        with torch.cuda.stream(self.copy_stream):
            for p, st, a in zip(self.gpu_params, self.stage, self.acc):
                if p.grad is not None:
                    g = _local(p.grad)
                    st.copy_(g, non_blocking=True)
                    g.record_stream(self.copy_stream)  # the allocator keeps g until the copy is done
                    p.grad = None
                    live.append((st, a))
        done = torch.cuda.Event()
        done.record(self.copy_stream)

        def add():
            done.synchronize()
            for st, a in live:
                a.add_(st)

        self.adder = threading.Thread(target=add)
        self.adder.start()

    def _drain(self):
        if self.adder is not None:
            self.adder.join()
            self.adder = None

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
        loss = super().step(closure)
        self._push_masters()
        torch._foreach_zero_(self.acc)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        # host accumulators are zeroed by step(); the GPU grads are freed by accumulate()
        for p in self.gpu_params:
            p.grad = None

    def state_dict(self):
        self._drain()
        sd = super().state_dict()
        sd["masters"] = self.masters
        return sd

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        masters = state_dict.pop("masters", None)
        # a plain AdamW state from a run without offload holds DTensor moments; the host optimizer keeps local shards
        state_dict["state"] = {
            k: {n: _local(v) for n, v in s.items()} for k, s in state_dict["state"].items()
        }
        super().load_state_dict(state_dict)
        for m, a in zip(self.masters, self.acc):
            m.grad = a
        if masters is not None:
            for m, s in zip(self.masters, masters):
                m.copy_(s)
            self._push_masters()
