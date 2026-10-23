"""Optimizer entry points, loaded lazily so model metadata stays lightweight."""
from importlib import import_module


__all__ = ["rescale_grads", "build_optimizer", "build_lr_scheduler", "restore_optimizer_config", "MuonLoadPlanner"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".lr_scheduler" if name == "build_lr_scheduler" else ".optimizer"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
