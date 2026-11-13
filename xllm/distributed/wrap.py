import contextlib
from typing import Any, Generator, Optional, Sequence
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method


@contextlib.contextmanager
def enable_wrap(
    expert_mesh: Optional[DeviceMesh] = None,
    **wrapper_kwargs: Any
) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

    Usage::

        with enable_wrap(wrapper_cls, **params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        wrapper_cls:
            Class that `wrap` annotation will `wrap` modules with, such as
            `FullyShardedDataParallel`.
        expert_mesh:
            Mesh of the routed experts for ``wrap_experts`` (None: the mesh
            in ``wrapper_kwargs``).
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    with _ConfigAutoWrap(expert_mesh, **wrapper_kwargs):
        yield


def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
    if _ConfigAutoWrap.in_autowrap_context:
        wrap_overrides = {**_ConfigAutoWrap.kwargs, **wrap_overrides}
        return fully_shard(module, **wrap_overrides)
    return module


def wrap_experts(module: nn.Module) -> nn.Module:
    """
    Shard the routed experts as their own unit, on the expert mesh.

    Call it before wrapping the layer that holds them. Their run_fused method
    is registered as a forward method, so the fused blocks that run through it
    gather and reduce the expert weights.
    """
    if not _ConfigAutoWrap.in_autowrap_context:
        return module
    overrides = {} if _ConfigAutoWrap.expert_mesh is None else {"mesh": _ConfigAutoWrap.expert_mesh}
    module = wrap(module, **overrides)
    register_fsdp_forward_method(module, "run_fused")
    return module


def prefetch_experts_in_backward(units: Sequence[nn.Module]) -> None:
    """
    Make each unit gather in backward the unit before it together with its experts.

    Args:
        units: the units in forward order, e.g. the layers then the output layer.

    By default, FSDP2 gathers in backward the unit whose forward ended just
    before. A layer and its experts are two units whose forwards end one
    right after the other, so one of them gathers the other, already
    gathered, and a part of the layer below would only be gathered once its
    backward starts.
    """
    for unit, prev in zip(units[1:], units[:-1]):
        experts = getattr(getattr(prev, "moe", None), "experts", None)
        if isinstance(unit, FSDPModule) and isinstance(prev, FSDPModule) and isinstance(experts, FSDPModule):
            unit.set_modules_to_backward_prefetch([prev, experts])


class _ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    kwargs: dict[str, Any] = {}  # Wrapper's args
    expert_mesh: Optional[DeviceMesh] = None  # Mesh of the routed experts

    def __init__(self, expert_mesh: Optional[DeviceMesh] = None, **kwargs: dict[str, Any]):
        self.expert_mesh = expert_mesh
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any, expert_mesh: Optional[DeviceMesh] = None) -> None:
        if _ConfigAutoWrap.in_autowrap_context:
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        _ConfigAutoWrap.in_autowrap_context = True
        # Save the rest.
        _ConfigAutoWrap.kwargs = kwargs
        _ConfigAutoWrap.expert_mesh = expert_mesh

    @staticmethod
    def disable_autowrap_context() -> None:
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.kwargs = {}
        _ConfigAutoWrap.expert_mesh = None

    def __enter__(self) -> None:
        self.enable_autowrap_context(self.kwargs, self.expert_mesh)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disable_autowrap_context()
