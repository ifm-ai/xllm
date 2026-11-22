from logging import getLogger

import torch
from torch import nn
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


def embed_readout_params(model: nn.Module) -> list[nn.Parameter]:
    """The embedding and the readout weight: the tensors optim.no_weight_decay_embed_readout exempts.

    Both are AdamW-stepped under either optimizer, since neither is ever register_muon'd. Looked up by attribute
    (model.embed, model.output.output), which transformer, gekko and hybrid all have, and which survives FSDP2's
    wrap because it shards modules in place. A model class that no longer has them raises here rather than quietly
    exempting nothing: this flag's failure mode is a run that looks fine while the embedding is still decaying.
    """
    return [p for module in (model.embed, model.output.output) for p in module.parameters()]


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str):
    from .muon import group_hparams
    logger.info("Starting build of optimizer...")
    no_decay = embed_readout_params(model) if cfg.no_weight_decay_embed_readout else []
    if no_decay:
        logger.info(f"optim.no_weight_decay_embed_readout: {len(no_decay)} tensors "
                    f"({sum(p.numel() for p in no_decay)} params) step with weight_decay 0")
    if cfg.optimizer == "muon":
        from .muon import build_muon
        optimizer = build_muon(model, cfg, no_decay)
    else:
        exempt = {id(p) for p in no_decay}
        groups = [dict(params=[p for p in model.parameters() if id(p) not in exempt],
                       **group_hparams(cfg, "adamw"))]
        if no_decay:
            groups.append(dict(params=no_decay, **group_hparams(cfg, "adamw_no_wd")))
        optimizer = AdamW(groups, eps=cfg.adamw.epsilon)

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
    from .muon import group_hparams
    for group in optimizer.param_groups:
        values = group_hparams(cfg, group["algorithm"])
        values.update(eps=cfg.adamw.epsilon, initial_lr=values.pop("lr"))  # the scheduler sets the current lr
        changed = {key: (group.get(key), value) for key, value in values.items() if group.get(key) != value}
        if changed:
            logger.info(f"Resume {group['algorithm']}: configured values replace checkpoint values (old, new): {changed}")
        group.update(values)
