"""fp8 all-gather of the fused Gekko block's linear weights under FSDP2 (Hopper fp8 GEMMs; opt-in XLLM_FP8_AG=1).

With ``fp8_gemm`` every rank otherwise all-gathers the bf16 weights of a layer, copies them out of FSDP2's flat
buffer and quantises all of them (the same work on every rank) before the fp8 GEMMs. Here the sharded weight is an
``Fp8ShardParam`` (a wrapper around the bf16 shard that behaves like it for every op) whose FSDP2 all-gather extension
(``fsdp_pre_all_gather`` / ``fsdp_post_all_gather``) quantises the local shard into 128 x 128 tiles (fp8_sm90) and
all-gathers the e4m3 data and the tile scales; the unsharded parameter is an ``Fp8Weight`` holding them, which
``MxLinearSet`` takes as its rowwise operands (the colwise ones are built in the backward by a transpose, as for the
quantised bf16 weights). Shards are row blocks of the (N, K) weight, so a multiple of 128 rows per shard keeps the
tiles of the full weight (checked by ``eligible``). Half the all-gather / copy-out bytes, 1 / world of the weight
quantisation, and no fp8 weight copies kept from the forward (FSDP2 frees and re-gathers the unsharded tensors; the
backward reads the re-gathered ones). Other ops on an ``Fp8Weight`` (a bf16 fallback path) dequantise it.

Measured (8 x H100, alternating A/B runs): peak memory -6 GiB on 7.1B, -2.6 on 4B, kernel time -35 ms per 4B step,
but no step-time gain (7.1B fp8 613 / 615 / 612 vs 619 / 617 / 619 ms; runs with it vary more, 565 - 609 for one
config): use it for the memory (7.1B fp8 + recompute_conv_qk=False fits with it).
"""
import os

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

aten = torch.ops.aten

# ops FSDP2 applies to the sharded parameter (sharding, padding, copies, state-dict loading): results stay wrappers
_PRESERVE = {aten.empty_like, aten.new_zeros, aten.new_empty, aten.slice, aten.narrow, aten.copy_, aten.view, aten.as_strided,
             aten._to_copy, aten._pin_memory, aten.split, aten.split_with_sizes, aten.chunk, aten.clone, aten.detach,
             aten.alias, aten.zero_, aten.fill_}


_STEP = [0]  # optimizer steps taken (any torch.optim optimizer): invalidates the quantised shards
_STEP_HOOK = []


def _register_step_hook():
    if not _STEP_HOOK:
        def bump(*_):
            _STEP[0] += 1
        from torch.optim.optimizer import register_optimizer_step_post_hook
        _STEP_HOOK.append(register_optimizer_step_post_hook(bump))


class Fp8ShardParam(torch.Tensor):
    """A bf16 (or fp32) weight shard with the FSDP2 fp8 all-gather extension; any op runs on the wrapped tensor."""

    @staticmethod
    def __new__(cls, t: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(cls, t.size(), strides=t.stride(), storage_offset=t.storage_offset(),
                                                   dtype=t.dtype, layout=t.layout, device=t.device,
                                                   requires_grad=t.requires_grad)

    def __init__(self, t: torch.Tensor):
        self._t = t
        self._q = None  # (version of _t, quantised shard): the backward's all-gather reuses the forward's

    def __repr__(self):
        return f"Fp8ShardParam({self._t!r})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unwrap = lambda x: x._t if isinstance(x, cls) else x
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        if func.overloadpacket not in _PRESERVE:
            return out
        if func._schema.is_mutable:  # in place (copy_, zero_, ...): the wrapper that was written to
            return args[0]
        return tree_map(lambda x: cls(x) if isinstance(x, torch.Tensor) and not isinstance(x, cls) else x, out)

    def __tensor_flatten__(self):
        return ["_t"], None

    @staticmethod
    def __tensor_unflatten__(inner, meta, outer_size, outer_stride):
        return Fp8ShardParam(inner["_t"])

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        # the shard changes in place: in an optimizer step (torch._fused_adamw_ does not bump version counters, so
        # every optimizer step bumps _STEP through a global step hook) or through ordinary ops (state-dict load, ...)
        key = (_STEP[0], self._t._version)
        if self._q is not None and self._q[0] == key:
            return self._q[1], None
        from xllm.modules.fused_ops import fp8_sm90
        row, _ = fp8_sm90.quantize(self._t, colwise=False, kind="weight")  # fp32 shards: rounded to bf16 in the kernel
        self._q = (key, (row.data, row.scale))
        return self._q[1], None

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        data, scale = all_gather_outputs
        if out is not None:  # the all-gather outputs are the unsharded tensor's storage (re-filled in place)
            return
        return Fp8Weight(data, scale, param_dtype), (data, scale)


class Fp8Weight(torch.Tensor):
    """An unsharded (N, K) weight as e4m3 data and 128 x 128 tile scales (reports the parameter dtype)."""

    @staticmethod
    def __new__(cls, data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), strides=data.stride(), dtype=dtype,
                                                   device=data.device, requires_grad=False)

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        self._data = data
        self._scale = scale

    def __repr__(self):
        return f"Fp8Weight(shape={tuple(self.shape)}, dtype={self.dtype})"

    def fp8_row(self):
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.Fp8Mat(self._data, self._scale, True)

    def dequantize(self) -> torch.Tensor:
        from xllm.modules.fused_ops import fp8_sm90
        return fp8_sm90.dequantize(self.fp8_row()).to(self.dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (aten.detach.default, aten.alias.default) or (
                func is aten.as_strided.default and tuple(args[1]) == tuple(args[0].shape)):
            w = args[0]
            return cls(w._data, w._scale, w.dtype)
        unwrap = lambda x: x.dequantize() if isinstance(x, cls) else x
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

    def __tensor_flatten__(self):
        return ["_data", "_scale"], self.dtype

    @staticmethod
    def __tensor_unflatten__(inner, dtype, outer_size, outer_stride):
        return Fp8Weight(inner["_data"], inner["_scale"], dtype)


def enabled() -> bool:
    return os.environ.get("XLLM_FP8_AG", "0") == "1"


def eligible(w: torch.Tensor, world_size: int) -> bool:
    """A weight whose FSDP2 row shards are whole 128-row tiles and whose dimensions the fp8 GEMMs take."""
    from xllm.modules.fused_ops import fp8_sm90
    # world_size 1: FSDP2 skips the all-gather (and with it the extension)
    return (world_size > 1 and w.dim() == 2 and w.is_cuda and torch.cuda.get_device_capability(w.device)[0] == 9
            and w.shape[0] % (128 * world_size) == 0 and fp8_sm90.supported(*w.shape))


class _CachedInspect:
    """FSDP2 calls inspect.signature(fsdp_pre_all_gather) on every all-gather of every parameter (~40 us each): the
    signature of the underlying function, cached."""

    def __init__(self, mod):
        self._mod = mod
        self._cache = {}

    def signature(self, f):
        key = getattr(f, "__func__", f)
        sig = self._cache.get(key)
        if sig is None:
            sig = self._cache[key] = self._mod.signature(f)
        return sig

    def __getattr__(self, name):
        return getattr(self._mod, name)


def _patch_fsdp_inspect():
    import torch.distributed.fsdp._fully_shard._fsdp_param as fp
    if not isinstance(getattr(fp, "inspect", None), _CachedInspect) and hasattr(fp, "inspect"):
        fp.inspect = _CachedInspect(fp.inspect)


def convert(modules, world_size: int) -> int:
    """Replace the ``weight`` of each module (before ``fully_shard``) with an Fp8ShardParam; returns the count."""
    _patch_fsdp_inspect()
    _register_step_hook()
    n = 0
    for m in modules:
        if m is None or not isinstance(getattr(m, "weight", None), nn.Parameter) or not eligible(m.weight, world_size):
            continue
        p = m.weight
        m.weight = nn.Parameter(Fp8ShardParam(p.data), requires_grad=p.requires_grad)
        n += 1
    return n
