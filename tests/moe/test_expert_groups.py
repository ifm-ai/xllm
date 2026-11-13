"""Check the expert process groups against the EP = TP x E_out rules, for several (TP, CP, EP) on one world.

E_out = EP / TP consecutive data parallel replicas form an expert group (per CP rank). The GPU at TP rank k in
replica o of expert group g has:
- expert outer group: the E_out GPUs at TP rank k in expert group g, with rank o,
and owns the experts given by ExpertLayout. Launch one process per rank, e.g.:

    torchrun --nproc_per_node 8 tests/moe/test_expert_groups.py
    torchrun --nproc_per_node 8 tests/moe/test_expert_groups.py --configs "[(2, 1, 8)]"
"""
import os

import fire
import torch
import torch.distributed as dist

import xllm.distributed.initialize as parallel_state
from xllm.distributed import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_model_parallel_group,
    get_expert_outer_group,
    get_expert_outer_rank,
    get_expert_outer_world_size,
)
from xllm.modules.moe.layout import ExpertLayout
from xllm.modules.moe.moe import NormalizedMoE

# (TP, CP, EP) on 8 GPUs; EP None means EP = TP
CONFIGS = [
    (1, 1, None), (8, 1, None), (4, 1, None), (2, 1, None), (2, 2, None), (1, 2, None),
    (1, 1, 2), (1, 1, 8), (4, 1, 8), (2, 1, 4), (2, 1, 8), (2, 2, 4), (1, 2, 4),
]
# EP not a multiple of TP, or EP / TP not dividing DP
BAD_CONFIGS = [(4, 1, 6), (2, 1, 6), (4, 1, 12), (2, 2, 8)]
N_EXPERTS = 64


def group_ranks(group):
    return [dist.get_rank()] if group is None else dist.get_process_group_ranks(group)


def check(tp: int, cp: int, ep: int) -> list:
    """Every rule that does not hold on this rank."""
    rank = dist.get_rank()
    e_out = ep // tp
    # rank grid (DP, CP, TP), TP fastest
    k, j, i = rank % tp, rank // tp % cp, rank // (tp * cp)
    g, o = divmod(i, e_out)

    def global_rank(i_, j_, k_):
        return (i_ * cp + j_) * tp + k_

    errors = []
    expected = sorted(global_rank(g * e_out + o_, j, k) for o_ in range(e_out))
    if group_ranks(get_expert_outer_group()) != expected:
        errors.append(f"expert outer group {group_ranks(get_expert_outer_group())} != {expected}")
    if (get_expert_outer_world_size(), get_expert_outer_rank()) != (e_out, o):
        errors.append(f"outer size and rank {get_expert_outer_world_size()}, {get_expert_outer_rank()}")
    if e_out == 1 and get_expert_outer_group() is not None:
        # nothing new is created
        errors.append("expert outer group is not None")

    layout = ExpertLayout.from_process_groups(N_EXPERTS)
    n_local = N_EXPERTS // ep
    if (layout.tp, layout.e_out, layout.tp_rank, layout.outer_rank) != (tp, e_out, k, o):
        errors.append(f"layout {layout}")
    if (layout.start, layout.end) != (k * N_EXPERTS // tp + o * n_local, k * N_EXPERTS // tp + (o + 1) * n_local):
        errors.append(f"owned experts [{layout.start}, {layout.end})")

    # through the collectives: the outer group owns its TP rank's share, so each expert group (the outer
    # groups of a replica's TP ranks) owns every expert once
    owned = torch.zeros(N_EXPERTS, device='cuda')
    owned[layout.start:layout.end] = 1
    counts = {'expert outer': owned.clone()}
    if get_expert_outer_group() is not None:
        dist.all_reduce(counts['expert outer'], group=get_expert_outer_group())
    counts['expert group'] = counts['expert outer'].clone()
    if get_model_parallel_group() is not None:
        dist.all_reduce(counts['expert group'], group=get_model_parallel_group())
    tp_share = torch.zeros(N_EXPERTS, device='cuda')
    tp_share[k * N_EXPERTS // tp:(k + 1) * N_EXPERTS // tp] = 1
    if not torch.equal(counts['expert group'], torch.ones_like(owned)):
        errors.append("expert group does not own every expert once")
    if not torch.equal(counts['expert outer'], tp_share):
        errors.append("outer group does not own its TP rank's share")

    # every layout builds its MoE layer
    NormalizedMoE(0, 64, 16, N_EXPERTS, 2, 0)
    return errors


def main(configs=None):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()

    failed = False
    for tp, cp, ep in CONFIGS if configs is None else configs:
        initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=ep)
        errors = check(tp, cp, tp if ep is None else ep)
        # free the communicators before the next config: there are too many to keep them all
        groups = {id(group): group for name in dir(parallel_state) if name.endswith('_GROUP')
                  for group in [getattr(parallel_state, name)] if group is not None}
        for group in groups.values():
            dist.destroy_process_group(group)
        destroy_model_parallel()
        names = [n for n in dir(parallel_state) if n.startswith('_EXPERT_')]
        if any(getattr(parallel_state, n) is not None for n in names):
            errors.append("destroy_model_parallel left expert state behind")

        bad = torch.tensor(len(errors), device='cuda')
        dist.all_reduce(bad)
        failed |= bad.item() > 0
        for line in errors:
            print(f"rank {dist.get_rank()}: tp={tp} cp={cp} ep={ep}: {line}", flush=True)
        if dist.get_rank() == 0:
            dp = world_size // (tp * cp)
            print(f"tp={tp} cp={cp} dp={dp} ep={ep}: {'ok' if bad.item() == 0 else 'FAIL'}", flush=True)

    if configs is None:
        for tp, cp, ep in BAD_CONFIGS:
            try:
                initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=ep)
                ok = False
            except AssertionError:
                ok = True
            destroy_model_parallel()
            failed |= not ok
            if dist.get_rank() == 0:
                print(f"tp={tp} cp={cp} ep={ep}: {'rejected' if ok else 'FAIL: accepted'}", flush=True)

    dist.destroy_process_group()
    assert not failed, "expert group checks failed"


if __name__ == '__main__':
    fire.Fire(main)
