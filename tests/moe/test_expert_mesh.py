"""Check the FSDP mesh of the routed experts, for several (TP, CP, EP, fully sharded size) on one world.

The mesh holds the expert data parallel GPUs (the GPUs owning the same experts) as (replicate, shard),
with as many shards as the largest divisor of DP / E_out up to the fully sharded size (min(fully sharded
size, DP / E_out) when it divides DP / E_out). With EP = TP its ranks are those of the model's FSDP mesh.
Launch one process per rank, on 8 or 6 GPUs (where DP / E_out can be 3), e.g.:

    torchrun --nproc_per_node 8 tests/moe/test_expert_mesh.py
    torchrun --nproc_per_node 6 tests/moe/test_expert_mesh.py
    torchrun --nproc_per_node 8 tests/moe/test_expert_mesh.py --configs "[(1, 1, 2, 4)]"
"""
import os

import fire
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard

import xllm.distributed.initialize as parallel_state
from xllm.distributed import initialize_model_parallel, destroy_model_parallel
from xllm.distributed.utils import get_device_mesh, get_expert_device_mesh, get_expert_shard_size
from xllm.modules.moe.layout import ExpertLayout

# (TP, CP, EP, fully sharded size) per world size; EP None means EP = TP
CONFIGS = {
    8: [
        (1, 1, None, 8), (2, 1, None, 4), (2, 1, None, 2), (4, 1, None, 2), (1, 2, None, 4), (2, 2, None, 2),
        (1, 1, 2, 8), (1, 1, 2, 2), (1, 1, 4, 2), (1, 1, 4, 8), (2, 1, 4, 4), (2, 1, 4, 2), (1, 2, 2, 4),
        (1, 2, 4, 4), (1, 1, 8, 8),
    ],
    # 3 expert groups: a fully sharded size of 2 does not divide them, the experts are not sharded
    6: [(1, 1, None, 2), (1, 1, 2, 2), (1, 1, 2, 6), (1, 1, 3, 3), (1, 1, 3, 2), (1, 1, 6, 6), (2, 1, 6, 3)],
}
# (fully sharded size, expert groups) -> expert shard size
SHARD_SIZES = {(8, 4): 4, (4, 8): 4, (2, 3): 1, (8, 12): 6, (8, 24): 8, (16, 12): 12, (6, 9): 3, (1, 5): 1}
N_EXPERTS = 48


def check(tp: int, cp: int, ep: int, fs: int) -> list:
    """Every rule that does not hold on this rank."""
    world_size, rank = dist.get_world_size(), dist.get_rank()
    dp = world_size // (tp * cp)
    e_out = ep // tp
    errors = []
    # rank grid (DP, CP, TP), TP fastest; the replica i is the replica o of expert group i // E_out
    k, i = rank % tp, rank // (tp * cp)
    o = i % e_out
    # the expert data parallel GPUs: TP rank k of replica o of every expert group, at every CP rank
    expert_dp = sorted((d * cp + j) * tp + k for d in range(o, dp, e_out) for j in range(cp))

    mesh = get_expert_device_mesh(fs, cp, tp, e_out)
    shard = max(d for d in range(1, fs + 1) if dp // e_out % d == 0)
    replicate = dp // e_out * cp // shard
    ranks = mesh.mesh.flatten().tolist()
    expected_shape = (shard,) if replicate == 1 else (replicate, shard)
    if tuple(mesh.mesh.shape) != expected_shape:
        errors.append(f"mesh shape {tuple(mesh.mesh.shape)} != {expected_shape}")
    if sorted(ranks) != expert_dp:
        errors.append(f"mesh ranks {sorted(ranks)} are not the expert data parallel GPUs {expert_dp}")
    if e_out == 1:
        dense = get_device_mesh(fs, dp // fs * cp, tp)
        if not torch.equal(dense.mesh, mesh.mesh):
            errors.append(f"mesh {mesh.mesh.tolist()} != model mesh {dense.mesh.tolist()}")

    # through the collectives: every GPU of a shard group and of a replicate group owns the same experts
    layout = ExpertLayout.from_process_groups(N_EXPERTS)
    owned = torch.zeros(N_EXPERTS, device='cuda')
    owned[layout.start:layout.end] = 1
    for dim, size in enumerate(mesh.mesh.shape):
        counts = owned.clone()
        dist.all_reduce(counts, group=mesh.get_group(dim))
        if not torch.equal(counts, owned * size):
            errors.append(f"mesh dim {dim} mixes GPUs owning different experts")

    # FSDP shards on it and gathers back the same weights
    module = nn.Linear(16, layout.n_local * 8, bias=False, device='cuda')
    with torch.no_grad():
        module.weight.copy_(torch.arange(module.weight.numel(), device='cuda').view_as(module.weight) + layout.start)
    expected = module.weight.detach().clone()
    fully_shard(module, mesh=mesh)
    if not torch.equal(module.weight.full_tensor(), expected):
        errors.append("FSDP on the mesh does not gather the weights back")
    return errors


def main(configs=None):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    failed = False
    shard_sizes = {k: get_expert_shard_size(*k) for k in SHARD_SIZES}
    if shard_sizes != SHARD_SIZES:
        failed = True
        if dist.get_rank() == 0:
            print(f"FAIL: expert shard sizes {shard_sizes} != {SHARD_SIZES}", flush=True)
    for tp, cp, ep, fs in CONFIGS[dist.get_world_size()] if configs is None else configs:
        initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=ep)
        errors = check(tp, cp, tp if ep is None else ep, fs)
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

    dist.destroy_process_group()
    assert not failed, "expert mesh checks failed"


if __name__ == '__main__':
    fire.Fire(main)
