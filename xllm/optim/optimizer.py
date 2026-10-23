from logging import getLogger
from typing import Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.optim import AdamW

from xllm.config import OptimConf
from .lr_scheduler import build_lr_scheduler


logger = getLogger()


@torch.no_grad()
def rescale_grads(model: nn.Module, scale: float):
    if scale == 1.0:
        return
    p_with_grads = (p for p in model.parameters() if p.grad is not None)
    for p in p_with_grads:
        assert p.grad is not None  # mypy
        p.grad /= scale


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str,
                    mesh: Optional[DeviceMesh] = None):
    logger.info("Starting build of optimizer...")
    if cfg.optimizer == "muon":
        if mesh is None:
            raise ValueError("Muon requires the FSDP2 device mesh used to build the model.")
        from .muon import build_muon
        optimizer = build_muon(model, mesh, cfg)
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
            eps=cfg.epsilon,
        )

    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler


def restore_optimizer_config(optimizer, cfg: OptimConf):
    """Training resume policy: explicitly apply the configured Muon/AdamW hyperparameters.

    Loading the optimizer alone restores checkpoint values. Training calls this
    after loading state; the scheduler separately determines the current LRs.
    The standalone AdamW path retains upstream's existing resume behavior.
    """
    if cfg.optimizer != "muon":
        return
    for group in optimizer.param_groups:
        if group["algorithm"] == "muon":
            values = dict(momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
                          adjust_lr=cfg.muon_adjust_lr, block_scale=cfg.muon_block_scale,
                          weight_decay=cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay,
                          eps=cfg.epsilon, initial_lr=cfg.muon_lr)
        else:
            values = dict(betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
                          eps=cfg.epsilon, initial_lr=cfg.lr)
        changed = {key: (group.get(key), value) for key, value in values.items() if group.get(key) != value}
        if changed:
            logger.info(f"Resume {group['algorithm']}: configured values replace checkpoint values (old, new): {changed}")
        group.update(values)


class MuonLoadPlanner(DefaultLoadPlanner):
    """Read legacy Muon DCP metadata without permitting missing optimizer tensors."""

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        super().set_up_planner(state_dict, metadata, is_coordinator)
        if metadata is None:
            return
        legacy_fields = {"momentum", "nesterov", "adjust_lr", "block_scale", "betas"}
        for key, path in self.mappings.items():
            if (len(path) >= 3 and path[0] == "param_groups" and path[2] in legacy_fields
                    and key not in metadata.state_dict_metadata):
                # The template's configured value remains in original_state_dict.
                # All state tensors and pre-existing metadata remain strict.
                logger.warning(f"Legacy Muon checkpoint lacks {key}; using configured value.")
                self.state_dict.pop(key, None)
