"""CUDA-graph replay of the fused Gekko block (``ModelConf.cuda_graph_block``).

The block's forward and backward are captured once per layer (after ``warmup`` eager calls, which also compile the
kernels) and replayed afterwards. A launch-bound preset (gekko-4B: ~300 ms of host issue per step against ~250 ms
of kernels) then issues two graph launches per layer instead of ~150 kernel launches.

Every tensor input is copied into a static buffer before the replay (one ``_foreach_copy_`` per layer): the block
input, the data tensors (freqs_cis, bos_mask, segment_idx, prev_segment_count, the timestep-norm priors) and the
parameters. The parameters are copied too because under FSDP2 the unsharded parameter storage is reallocated every
step (also with ``reshard_after_forward=False``: it is freed after the backward), so a graph recorded against it would
have to be recaptured every step; the copies cost one read + write of the layer's parameters per step (gekko-4B:
~8 GB per step, ~1 ms). The output and the gradients are static buffers of the graph pool, returned detached (a consumer
must not modify them in place - the model does not); the gradients flow into the real parameters through autograd.

Requirements: no dropout (RNG state is not refreshed inside a replay), no context parallelism (the block would receive /
send on other streams), fixed shapes (one capture per layer; a shape change recaptures).

Measured (gekko-4B B1 L8192, B200, 2026-10-25, same batches): 1 GPU 258 / 252 / 249 / 311 -> 247 / 246 / 245 / 263 ms
per step (3-6%); 8 GPUs with recompute_conv_qk=False, fsdp_layers_per_group=4, reshard_after_forward=False: 262 -> 267
(no gain). The compute-stream gaps do disappear (1 GPU: 63 -> 2 ms per step), but the GPU runs at its 1000 W power cap
in both modes and a denser kernel stream only lowers the SM clock (1965 -> ~1700 MHz; every kernel 4-9% slower inside
the replay), so half of the host time saved comes back as kernel time. Peak memory +21-28 GB (the static parameter and
gradient buffers of 36 layers, 2 x 7.7 GB, the input copies and the graph pool). Losses match eager (atomics noise).
Default off: worth it only where the host is the bottleneck and the GPU is not power-bound.
"""
import os
import warnings
from typing import Any, List, Optional, Tuple

import torch

from .gekko import GekkoBlockFunction


# All layers' graphs share one memory pool: they are captured lazily in the order they are replayed every step (the
# forwards in layer order, then the backwards in reverse), which is the condition for sharing a pool, so the temporaries
# of a layer's graphs are reused by the next layer's instead of being held 36 times. A shape change bumps the generation:
# every layer recaptures into a fresh pool (the old pool is released with the old graphs).
_shared = {"gen": 0, "pool": None}


def _bump_generation():
    _shared["gen"] += 1
    _shared["pool"] = None


class _LayerGraphs:
    """The captured forward / backward graphs of one layer and their static tensors."""

    def __init__(self):
        self.fwd: Optional[torch.cuda.CUDAGraph] = None
        self.bwd: Optional[torch.cuda.CUDAGraph] = None
        self.pool = None
        self.static_args: List[Any] = []
        self.tensor_idx: List[int] = []  # positions of all tensor args (the outer Function's inputs, all copied)
        self.static_tensors: List[torch.Tensor] = []  # their static buffers, in that order
        self.out: Optional[torch.Tensor] = None
        self.gout: Optional[torch.Tensor] = None
        self.grads: dict = {}  # arg position -> static gradient buffer (None: no grad)
        self.shapes: Tuple = ()
        self.gen: int = -1


def _tensor_signature(args) -> Tuple:
    return tuple((i, a.shape, a.dtype, a.requires_grad) for i, a in enumerate(args) if isinstance(a, torch.Tensor))


class GraphedGekkoBlock:
    """Callable replacing ``GekkoBlockFunction.apply`` for one layer."""

    def __init__(self, layer_id: int, warmup: int = 2):
        self.layer_id = layer_id
        self.warmup = max(1, warmup)
        self.calls = 0
        self.g: Optional[_LayerGraphs] = None
        self.captures = 0

    def __call__(self, *args):
        self.calls += 1
        if self.calls <= self.warmup:
            return GekkoBlockFunction.apply(*args)
        g = self.g
        if g is not None and g.shapes != _tensor_signature(args):
            if self.captures >= 2 and self.captures % 8 == 2:
                warnings.warn(f"GraphedGekkoBlock layer {self.layer_id}: recapture #{self.captures} (the input shapes change "
                              "between steps; the CUDA graphs are captured per shape)")
            _bump_generation()
        if g is None or g.gen != _shared["gen"]:
            self._capture(args)
            g = self.g
        tensors = [args[i] for i in g.tensor_idx]
        out = _Replay.apply(g, *tensors)
        return out, None, None

    def _capture(self, args):
        self.g = None  # release the previous graphs and their pool first
        g = _LayerGraphs()
        g.shapes = _tensor_signature(args)
        static = list(args)
        for i, a in enumerate(args):
            if not isinstance(a, torch.Tensor):
                continue
            g.tensor_idx.append(i)
            static[i] = a.detach().clone().requires_grad_(a.requires_grad)
            g.static_tensors.append(static[i])
        g.static_args = static
        grad_pos = [i for i in g.tensor_idx if static[i].requires_grad]

        g.fwd = torch.cuda.CUDAGraph()
        g.bwd = torch.cuda.CUDAGraph()
        if _shared["pool"] is None:
            _shared["pool"] = torch.cuda.graph_pool_handle()
        g.pool = _shared["pool"]
        g.gen = _shared["gen"]
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            # one eager iteration on the capture stream (allocator / cuBLAS workspace of that stream)
            out, _, _ = GekkoBlockFunction.apply(*static)
            torch.autograd.grad((out,), [static[i] for i in grad_pos], (torch.zeros_like(out),), allow_unused=True)
            del out
        torch.cuda.current_stream().wait_stream(capture_stream)
        with torch.cuda.graph(g.fwd, pool=g.pool, stream=capture_stream):
            out, _, _ = GekkoBlockFunction.apply(*static)
        g.out = out
        g.gout = torch.empty_like(out)
        with torch.cuda.graph(g.bwd, pool=g.pool, stream=capture_stream):
            grads = torch.autograd.grad((out,), [static[i] for i in grad_pos], (g.gout,), allow_unused=True, retain_graph=True)
        g.grads = {i: t for i, t in zip(grad_pos, grads)}
        self.g = g
        self.captures += 1
        if os.environ.get("XLLM_CG_DEBUG"):
            print(f"[cuda graph] layer {self.layer_id}: capture #{self.captures} (call {self.calls}), {len(g.tensor_idx)} tensors", flush=True)


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g: _LayerGraphs, *tensors):
        with torch.no_grad():
            torch._foreach_copy_(g.static_tensors, list(tensors))
        g.fwd.replay()
        ctx.g = g
        return g.out.detach()

    @staticmethod
    def backward(ctx, gout):
        g = ctx.g
        g.gout.copy_(gout)
        g.bwd.replay()
        # the static gradient buffers are overwritten by the next replay: hand autograd fresh tensors (which the
        # parameters' AccumulateGrad nodes take over without another copy)
        return (None,) + tuple(None if g.grads.get(i) is None else g.grads[i].clone() for i in g.tensor_idx)
