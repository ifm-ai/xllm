from logging import getLogger

import torch
from torch import nn
from torch.distributed.tensor import DTensor
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


def _local(t):
    t = t._local_tensor if isinstance(t, DTensor) else t
    # a wrapper shard (fp8_fsdp.Fp8ShardParam, XLLM_FP8_AG): the plain tensor it wraps (updated in place, so the
    # wrapper sees the update), without a Python dispatch per optimizer step
    return getattr(t, "_t", t)


class LocalFusedAdamW(AdamW):
    """AdamW(fused=True) for FSDP2 (DTensor) parameters: the fused kernel runs on the local shards of the params,
    grads and moments directly. Dispatching one ``_fused_adamw_`` over N DTensors costs ~10 ms of CPU per step
    (DTensor spec construction per tensor); the states stay DTensors, so checkpoints are unchanged."""

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._lists = {}  # the state tensors were replaced

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        self._lists = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if "_lists" not in self.__dict__:
            self._lists = {}
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for gi, group in enumerate(self.param_groups):
            assert group["fused"] and not group["amsgrad"] and not group["maximize"] and not group["differentiable"]
            cached = self._lists.get(gi)
            if cached is not None and all(p.grad is not None for p in cached[0]):
                # the local tensors of the params and moments and the step counters do not change between steps;
                # only the gradients are new tensors each step (~3 ms of CPU per step saved on 600 params)
                params, local = cached
                lists = [local[0], [_local(p.grad) for p in params], local[1], local[2]]
                state_steps = local[3]
            else:
                params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], [], [], []
                self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
                if not params:
                    continue
                lists = [[_local(t) for t in ts] for ts in (params, grads, exp_avgs, exp_avg_sqs)]
                keep = [i for i, t in enumerate(lists[0]) if t.numel() > 0]
                if len(keep) < len(params):
                    params = [params[i] for i in keep]
                    lists = [[ts[i] for i in keep] for ts in lists]
                    state_steps = [state_steps[i] for i in keep]
                self._lists[gi] = (params, (lists[0], lists[2], lists[3], state_steps))
            torch._foreach_add_(state_steps, 1)
            torch._fused_adamw_(
                *lists, [], state_steps, amsgrad=False, lr=group["lr"], beta1=group["betas"][0], beta2=group["betas"][1],
                weight_decay=group["weight_decay"], eps=group["eps"], maximize=False, grad_scale=None, found_inf=None,
            )
        return loss


def build_optimizer(model: nn.Module, cfg: OptimConf, total_steps: int, param_dtype: str):
    logger.info("Starting build of optimizer...")
    params = list(model.parameters())
    cls = LocalFusedAdamW if cfg.fused and any(isinstance(p, DTensor) for p in params) else AdamW
    optimizer = cls(
        params,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.epsilon,
        fused=cfg.fused and torch.cuda.is_available(),
    )

    # scheduler
    scheduler = build_lr_scheduler(optimizer, cfg, total_steps)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
