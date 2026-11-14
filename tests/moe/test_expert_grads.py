"""Check the gradients of the routed experts on their FSDP mesh, for several (TP, CP, EP, fully sharded size).

A toy model has a weight on the model's FSDP mesh (distinct on each TP rank, as a TP-split weight) and real
routed experts (Expert) on the expert mesh, sharded with the same wrap calls as the model. For each config:

- gradient scaling: GPU r has the loss a_r * sum(weight) + A_r * sum_e (e + 1) * sum(W_e), with a_r = r + 1 and
  A_r the sum of a over the E_out replicas of r's expert group, as each expert gets the tokens of these
  replicas. After the FSDP reduction both gradients must be the mean over the DP x CP GPUs of the TP rank: the
  weight's gradient is mean(a) and expert e's is (e + 1) * mean(a). Without the experts' divide factor the
  latter would be E_out times larger.
- gradient norm: synthetic gradients drawn from seeds (per TP rank for the weight, per global expert for the
  experts) are set on both meshes. XLLModel.clip_grad_norm_ must return the 2-norm and the inf-norm of the
  full gradients (every TP rank's weight and every expert once) and clip every gradient by the same factor;
  the health metrics' squared norms must add up to the same values; an AdamW step on the clipped gradients
  must match one on the full tensors.

The gradient scaling is checked again with a bf16 mixed precision policy (bf16 compute and reduction), where FSDP
must not divide by the experts' factor in the reduction itself (NCCL's bf16 PreMulSum returns 0 in our build).

Every value must be finite. Launch one process per rank; it exits with 1 if any check fails, e.g.:

    torchrun --nproc_per_node 8 tests/moe/test_expert_grads.py
    torchrun --nproc_per_node 8 tests/moe/test_expert_grads.py --configs "[(2, 1, 4, 2)]"
    torchrun --nproc_per_node 8 tests/moe/test_expert_grads.py --bug outer   # must fail

--bug injects a known error to check that the test catches it: 'divide' (no expert gradient divide factor),
'premul' (the divide factor applied in the reduction, as a PreMulSum),
'outer' (expert norm not added up over the expert group), 'replicas' (norms added up over the whole FSDP mesh,
counting replicas several times).
"""
import math
import os
import sys

import fire
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor

import xllm.distributed.initialize as parallel_state
import xllm.models.xllm as xllm_model
from xllm.distributed import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_expert_outer_group,
    get_model_parallel_rank,
)
from xllm.distributed.utils import get_device_mesh, get_expert_device_mesh
from xllm.distributed.wrap import enable_wrap, wrap, wrap_experts
from xllm.health_metrics import _dtensor_squared_l2_contribution
from xllm.models.xllm import XLLModel
from xllm.modules.moe.expert import Expert, routed_expert_parameters

# (TP, CP, EP, fully sharded size) on 8 GPUs; EP None means EP = TP
CONFIGS = [
    (2, 1, None, 4), (2, 1, None, 2), (1, 2, None, 2),
    (1, 1, 2, 8), (1, 1, 2, 2), (2, 1, 4, 4), (2, 1, 4, 2), (1, 2, 2, 4), (1, 2, 2, 2), (2, 2, 4, 2),
    (1, 1, 4, 8), (1, 1, 4, 2), (2, 1, 8, 4), (1, 2, 4, 4),
    (1, 1, 8, 8),
]
N_EXPERTS, MODEL_DIM, INTER_DIM, DENSE_SHAPE = 16, 16, 8, (24, 16)
WEIGHT_NAMES = ('weight1', 'weight2', 'weight3')


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(DENSE_SHAPE[1], DENSE_SHAPE[0], bias=False)
        self.experts = Expert(MODEL_DIM, INTER_DIM, N_EXPERTS)

    def forward(self, a: torch.Tensor, x: torch.Tensor, coefs: dict) -> torch.Tensor:
        expert_loss = self.experts.run_fused(
            lambda x, *weights: x * sum((coefs[n][:, None] * w).sum() for n, w in zip(WEIGHT_NAMES, weights)), x
        )
        return a * self.dense.weight.sum() + expert_loss


def expert_shape(name: str) -> tuple:
    """Shape of one expert's block of an expert weight."""
    return (MODEL_DIM, INTER_DIM) if name == 'weight2' else (INTER_DIM, MODEL_DIM)


def synthetic_grad(name: str, index: int) -> torch.Tensor:
    """Gradient of the weight of TP rank `index` ('dense'), or of the global expert `index`, from its own seed."""
    seed = 1000 * (WEIGHT_NAMES.index(name) + 1 if name in WEIGHT_NAMES else 0) + index
    shape = DENSE_SHAPE if name == 'dense' else expert_shape(name)
    return torch.randn(shape, generator=torch.Generator(device='cuda').manual_seed(seed), device='cuda') * (1 + index % 3)


def full_grads(tp: int, layout) -> tuple:
    """This GPU's full (FSDP-unsharded) synthetic gradients, and the full gradients of the whole model."""
    local = {'dense.weight': synthetic_grad('dense', get_model_parallel_rank())}
    for name in WEIGHT_NAMES:
        local[f'experts.{name}'] = torch.cat([synthetic_grad(name, e) for e in range(layout.start, layout.end)])
    everything = [synthetic_grad('dense', k) for k in range(tp)]
    everything += [synthetic_grad(name, e) for name in WEIGHT_NAMES for e in range(N_EXPERTS)]
    return local, everything


def set_grads(model: nn.Module, local: dict):
    for name, p in model.named_parameters():
        p.grad = distribute_tensor(local[name].clone(), p.device_mesh, p.placements)


def close(x, y, rtol=1e-5, atol=1e-6) -> bool:
    x, y = (torch.as_tensor(t).detach().to('cpu', torch.float64) for t in (x, y))
    return bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.allclose(x, y, rtol=rtol, atol=atol))


def build_meshes(tp: int, cp: int, ep: int, fs: int) -> tuple:
    """The meshes of build_model_fsdp2: the model's and the experts'."""
    dp = dist.get_world_size() // (tp * cp)
    mesh = get_device_mesh(fs, dp // fs * cp, tp)
    return mesh, mesh if ep == tp else get_expert_device_mesh(fs, cp, tp, ep // tp)


def check(tp: int, cp: int, ep: int, fs: int, meshes: tuple, reduce: str = 'fp32') -> list:
    """Every rule that does not hold on this rank. With reduce='bf16', FSDP computes and reduces in bf16 and only
    the gradient scaling is checked."""
    world_size, rank = dist.get_world_size(), dist.get_rank()
    dp = world_size // (tp * cp)
    e_out = ep // tp
    errors = []

    # the meshes and wrap calls of build_model_fsdp2 and Transformer
    mesh, expert_mesh = meshes
    torch.manual_seed(0)
    wrap_kwargs = {}
    if reduce == 'bf16':
        wrap_kwargs['mp_policy'] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    # bf16 holds the per-GPU gradients exactly, not their mean
    rtol = 2e-2 if reduce == 'bf16' else 1e-5
    with torch.device('cuda'), enable_wrap(expert_mesh=expert_mesh, mesh=mesh, **wrap_kwargs):
        model = Toy()
        wrap_experts(model.experts)
        model = wrap(model)
    layout = model.experts.layout
    if [id(p) for p in routed_expert_parameters(model)] != [id(getattr(model.experts, n)) for n in WEIGHT_NAMES]:
        errors.append("routed_expert_parameters does not return the expert weights")
    for p, m in ((model.experts.weight1, expert_mesh), (model.dense.weight, mesh)):
        if not torch.equal(p.device_mesh.mesh, m.mesh):
            errors.append(f"parameter on mesh {p.device_mesh.mesh.tolist()} instead of {m.mesh.tolist()}")

    # gradient scaling through the FSDP reduction
    a = float(rank + 1)
    group_a = torch.tensor(a, device='cuda')
    if e_out > 1:
        dist.all_reduce(group_a, group=get_expert_outer_group())
    x = group_a.clone().requires_grad_(True)
    coefs = {}
    for name in WEIGHT_NAMES:
        coefs[name] = torch.arange(layout.start + 1, layout.end + 1, device='cuda', dtype=torch.float32)
        coefs[name] = coefs[name].repeat_interleave(expert_shape(name)[0])
    model(torch.tensor(a, device='cuda'), x, coefs).backward()
    k = get_model_parallel_rank()
    mean_a = sum(r + 1 for r in range(world_size) if r % tp == k) / (dp * cp)
    if not close(model.dense.weight.grad.full_tensor(), torch.full(DENSE_SHAPE, mean_a), rtol=rtol):
        errors.append(f"weight gradient is not mean(a) = {mean_a}")
    for name in WEIGHT_NAMES:
        grad = model.experts.get_parameter(name).grad.full_tensor()
        if not close(grad, mean_a * coefs[name][:, None].expand_as(grad), rtol=rtol):
            ratio = (grad[0, 0] / (mean_a * coefs[name][0])).item()
            errors.append(f"expert {name} gradient is {ratio:.4g} x (e + 1) * mean(a)")
    model.zero_grad(set_to_none=True)
    if reduce == 'bf16':
        return errors

    # gradient norms and clipping on synthetic gradients
    local, everything = full_grads(tp, layout)
    ref_l2 = torch.sqrt(sum(g.double().square().sum() for g in everything)).item()
    ref_inf = max(g.abs().max().item() for g in everything)

    set_grads(model, local)
    # health metrics: the squared norms of the GPUs' parameter gradients add up to every TP rank's / expert's once
    health = torch.stack([_dtensor_squared_l2_contribution(p.grad) for _, p in model.named_parameters()]).double()
    dist.all_reduce(health)
    ref_health = [sum(g.double().square().sum() for g in everything[:tp])]
    ref_health += [sum(g.double().square().sum() for g in everything[tp + i * N_EXPERTS: tp + (i + 1) * N_EXPERTS])
                   for i in range(len(WEIGHT_NAMES))]
    if not close(health, torch.stack(ref_health)):
        errors.append(f"health squared norms {health.tolist()} != {[v.item() for v in ref_health]}")

    for norm_type, ref in ((math.inf, ref_inf), (2.0, ref_l2)):
        set_grads(model, local)
        max_norm = 0.5 * ref
        norm = XLLModel.clip_grad_norm_(model, max_norm, norm_type)  # the model's method, on the toy
        if not close(norm, ref):
            errors.append(f"{norm_type}-norm {norm} != {ref} (ratio {norm / ref:.6g})")
        coef = min(max_norm / (norm + 1e-6), 1.0)
        for name, p in model.named_parameters():
            if not close(p.grad.full_tensor(), local[name] * coef):
                errors.append(f"{norm_type}-norm: {name} gradient is not clipped by {coef:.6g}")

    # an AdamW step (default implementation) on both meshes against one on the full tensors
    named = list(model.named_parameters())
    full_params = [p.full_tensor().detach().clone().requires_grad_(True) for _, p in named]
    for (_, p), q in zip(named, full_params):
        q.grad = p.grad.full_tensor()
    for params, kwargs in ((model.parameters(), {}), (full_params, {'foreach': False})):
        torch.optim.AdamW(params, lr=1e-2, weight_decay=0.1, **kwargs).step()
    for (name, p), q in zip(named, full_params):
        if not close(p.full_tensor(), q.detach(), rtol=1e-6, atol=1e-7):
            errors.append(f"AdamW step on {name} differs from the full tensors")
    return errors


def inject(bug: str):
    if bug == 'divide':
        FSDPModule.set_gradient_divide_factor = lambda self, factor: None
    elif bug == 'premul':
        FSDPModule.set_force_sum_reduction_for_comms = lambda self, enable: None
    elif bug == 'outer':
        all_reduce_norm = xllm_model._all_reduce_norm

        def skip_outer(total_norm, norm_type, group):
            return total_norm if group is get_expert_outer_group() else all_reduce_norm(total_norm, norm_type, group)
        xllm_model._all_reduce_norm = skip_outer
    elif bug == 'replicas':
        def over_whole_mesh(grads, norm_type):
            assert norm_type == 2.0 or norm_type == math.inf
            mesh = grads[0].device_mesh
            local = [g.to_local().float() for g in grads]
            if norm_type == math.inf:
                total = torch.stack([t.abs().max() for t in local]).max()
            else:
                total = sum(t.square().sum() for t in local)
            for dim in range(mesh.ndim):
                dist.all_reduce(total, op=dist.ReduceOp.MAX if norm_type == math.inf else dist.ReduceOp.SUM,
                                group=mesh.get_group(dim))
            return total if norm_type == math.inf else total.sqrt()
        xllm_model._grads_norm = over_whole_mesh
    else:
        assert bug is None, f"unknown bug {bug}"


def main(configs=None, bug=None):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    inject(bug)

    failed = False
    for tp, cp, ep, fs in CONFIGS if configs is None else configs:
        initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=ep)
        ep_ = tp if ep is None else ep
        meshes = build_meshes(tp, cp, ep_, fs)
        errors = []
        for reduce in ('fp32', 'bf16'):
            try:
                errors += [f"{reduce} reduce: {line}" for line in check(tp, cp, ep_, fs, meshes, reduce)]
            except Exception as e:  # report and fail on every rank rather than hang the others
                errors.append(f"{reduce} reduce: {type(e).__name__}: {e}")
        groups = {id(group): group for name in dir(parallel_state) if name.endswith('_GROUP')
                  for group in [getattr(parallel_state, name)] if group is not None}
        for group in groups.values():
            dist.destroy_process_group(group)
        destroy_model_parallel()

        bad = torch.tensor(len(errors), device='cuda')
        dist.all_reduce(bad)
        failed |= bad.item() > 0
        for line in errors:
            print(f"rank {dist.get_rank()}: tp={tp} cp={cp} ep={ep} fs={fs}: {line}", flush=True)
        if dist.get_rank() == 0:
            print(f"tp={tp} cp={cp} ep={ep} fs={fs}: {'ok' if bad.item() == 0 else 'FAIL'}", flush=True)

    if dist.get_rank() == 0:
        print('FAILED' if failed else 'passed', flush=True)
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    fire.Fire(main)
