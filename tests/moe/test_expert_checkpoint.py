"""Check the checkpoints of the routed experts, saved at one EP and loaded at the same or another EP.

A toy model has a weight on the model's FSDP mesh and real routed experts (Expert) on the expert mesh, sharded
with the same wrap calls as the model. The experts are initialized expert by expert in global order, so a
model built from the same seed has the same experts at every EP. For each (TP, CP, fully sharded size, EP of
the save, EP of the load):

- a model built from seed 1 takes an AdamW step on gradients computed elementwise from its weights, and is
  saved with DCP as train.py does (model and DCP optimizer state, one directory per TP rank, over the hybrid
  shard group), and with the non-DCP optimizer files (one per master rank) when both EPs are the same;
- the expert weights and AdamW moments must be saved as one (N / TP x rows per expert) tensor per TP rank,
  the shape EP = TP saves;
- a model built from seed 2 at the EP of the load takes the same step and loads the checkpoint: its weights and
  AdamW states must be bitwise those of a model built from seed 1 at that EP that took the same step;
- loading non-DCP optimizer files saved at another EP must raise.

Launch one process per rank; it exits with 1 if any check fails, e.g.:

    torchrun --nproc_per_node 8 tests/moe/test_expert_checkpoint.py
    torchrun --nproc_per_node 8 tests/moe/test_expert_checkpoint.py --configs "[(2, 1, 4, 4, 8)]"
    torchrun --nproc_per_node 8 tests/moe/test_expert_checkpoint.py --bug noconvert   # must fail with EP > TP

--bug injects a known error to check that the test catches it: 'noconvert' saves and loads the experts as they
are (as before), which must fail with EP > TP; 'master' writes one non-DCP optimizer file per model replicate
group (as before), which must fail with EP > TP when the data parallel replicas of a model shard hold different
experts (HSDP).
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import fire
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import CheckpointException, FileSystemReader

import xllm.distributed.initialize as parallel_state
import xllm.modules.moe.checkpoint as expert_checkpoint
import xllm.utils as xllm_utils
from xllm.checkpointing import (
    check_optimizer_files_expert_parallel_size,
    sharded_model_dir,
    sharded_optim_dir,
    sharded_optim_name,
)
from xllm.distributed import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_hybrid_shard_data_parallel_group,
    get_model_parallel_rank,
)
from xllm.distributed.utils import get_device_mesh, get_expert_device_mesh
from xllm.distributed.wrap import enable_wrap, wrap, wrap_experts
from xllm.modules.moe.expert import Expert
from xllm.utils import (
    get_master_rank_in_replicated_group,
    get_model_state,
    get_sharded_optimizer_state,
    set_model_state,
    set_sharded_optimizer_state,
)

# (TP, CP, fully sharded size, EP of the save, EP of the load) on 8 GPUs
CONFIGS = [
    # same EP: EP = TP, then EP > TP with FSDP, HSDP and CP
    (2, 1, 4, 2, 2), (1, 2, 4, 1, 1), (2, 1, 4, 4, 4), (2, 1, 4, 8, 8), (2, 1, 2, 4, 4), (1, 2, 2, 4, 4),
    (1, 1, 8, 4, 4), (1, 1, 2, 2, 2), (1, 1, 2, 4, 4), (1, 2, 4, 2, 2), (2, 2, 2, 4, 4), (1, 1, 8, 8, 8),
    # another EP
    (2, 1, 4, 2, 4), (2, 1, 4, 4, 2), (2, 1, 4, 2, 8), (2, 1, 4, 8, 2), (2, 1, 4, 4, 8), (2, 1, 4, 8, 4),
    (2, 1, 2, 2, 8), (2, 1, 2, 8, 4), (1, 1, 8, 1, 8), (1, 1, 2, 8, 2), (1, 2, 4, 1, 4), (2, 2, 2, 2, 4),
]
N_EXPERTS, MODEL_DIM, INTER_DIM, DENSE_SHAPE = 16, 6, 5, (12, 6)
WEIGHT_NAMES = ('weight1', 'weight2', 'weight3')


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(DENSE_SHAPE[1], DENSE_SHAPE[0], bias=False)
        self.experts = Expert(MODEL_DIM, INTER_DIM, N_EXPERTS)


def build(tp: int, cp: int, fs: int, ep: int, seed: int):
    """A toy model built from `seed` on the meshes of build_model_fsdp2, after one AdamW step."""
    dp = dist.get_world_size() // (tp * cp)
    mesh = get_device_mesh(fs, dp // fs * cp, tp)
    expert_mesh = mesh if ep == tp else get_expert_device_mesh(fs, cp, tp, ep // tp)
    torch.manual_seed(seed)
    with torch.device('cuda'), enable_wrap(expert_mesh=expert_mesh, mesh=mesh):
        model = Toy()
        wrap_experts(model.experts)
        model = wrap(model)
    for p in model.parameters():
        p.grad = torch.sin(p.detach() * 3.0) + 0.25
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    optimizer.step()
    return model, optimizer


def local_states(model, optimizer) -> dict:
    """This GPU's shards of the weights and of their AdamW states."""
    out = {}
    for name, p in model.named_parameters():
        out[name] = p.to_local().detach().clone()
        for k, v in optimizer.state[p].items():
            out[f'{name}.{k}'] = (v.to_local() if hasattr(v, 'to_local') else v).detach().clone()
    return out


def compare(got: dict, ref: dict, what: str) -> list:
    errors = []
    for k in ref:
        if k not in got:
            errors.append(f"{what}: {k} missing")
        elif got[k].shape != ref[k].shape or not torch.equal(got[k].cpu(), ref[k].cpu()):
            errors.append(f"{what}: {k} differs (max |diff| "
                          f"{(got[k].cpu().double() - ref[k].cpu().double()).abs().max().item():.3g})"
                          if got[k].shape == ref[k].shape else f"{what}: {k} shape {got[k].shape} != {ref[k].shape}")
    return errors


def save(model, optimizer, root: Path, non_dcp: bool, fs: int):
    """Save as train.py: the DCP model and optimizer states per TP rank, and the non-DCP optimizer files."""
    tp_rank, group = get_model_parallel_rank(), get_hybrid_shard_data_parallel_group()
    dcp.save(get_model_state(model), checkpoint_id=root / sharded_model_dir(tp_rank), process_group=group)
    dcp.save(get_sharded_optimizer_state(model, optimizer, True), checkpoint_id=root / 'dcp_optim' /
             sharded_optim_dir(tp_rank), process_group=group)
    if non_dcp:
        rank = dist.get_rank()
        master = get_master_rank_in_replicated_group(rank, fs)
        if rank == master:
            path = root / sharded_optim_dir(tp_rank)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(optimizer.state_dict(), path / sharded_optim_name(master))
    dist.barrier()


def load(model, optimizer, root: Path, non_dcp: bool, fs: int):
    tp_rank, group = get_model_parallel_rank(), get_hybrid_shard_data_parallel_group()
    state = get_model_state(model)
    dcp.load(state, checkpoint_id=root / sharded_model_dir(tp_rank), process_group=group)
    set_model_state(model, state)
    if non_dcp:
        master = get_master_rank_in_replicated_group(dist.get_rank(), fs)
        path = root / sharded_optim_dir(tp_rank) / sharded_optim_name(master)
        optimizer.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    else:
        state = get_sharded_optimizer_state(model, optimizer, True)
        dcp.load(state, checkpoint_id=root / 'dcp_optim' / sharded_optim_dir(tp_rank), process_group=group)
        set_sharded_optimizer_state(model, optimizer, state, True)


def check_metadata(root: Path, tp: int) -> list:
    """The experts are saved as the N / TP experts of each TP rank, whatever the EP."""
    errors = []
    rows = {'weight1': INTER_DIM, 'weight2': MODEL_DIM, 'weight3': INTER_DIM}
    cols = {'weight1': MODEL_DIM, 'weight2': INTER_DIM, 'weight3': MODEL_DIM}
    for k in range(tp):
        for sub, prefix in ((sharded_model_dir(k), ''), (f'dcp_optim/{sharded_optim_dir(k)}', 'state.')):
            md = FileSystemReader(root / sub).read_metadata().state_dict_metadata
            for name in WEIGHT_NAMES:
                keys = [f'{prefix}experts.{name}'] if not prefix else [f'state.experts.{name}.exp_avg',
                                                                       f'state.experts.{name}.exp_avg_sq']
                for key in keys:
                    expected = (N_EXPERTS // tp * rows[name], cols[name])
                    if key not in md:
                        errors.append(f"{sub}: no {key}")
                    elif tuple(md[key].size) != expected:
                        errors.append(f"{sub}: {key} saved as {tuple(md[key].size)}, not {expected}")
    return errors


def check(tp: int, cp: int, fs: int, ep_save: int, ep_load: int, root: Path) -> list:
    """Every rule that does not hold on this rank."""
    errors = []
    same_ep = ep_save == ep_load
    initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=ep_save)
    model, optimizer = build(tp, cp, fs, ep_save, seed=1)
    save(model, optimizer, root, same_ep, fs)
    if dist.get_rank() == 0:
        errors += check_metadata(root, tp)
        with (root / 'config.json').open('w') as fp:
            json.dump({'expert_parallel_size': ep_save, 'model_parallel_size': tp}, fp)
    del model, optimizer
    if not same_ep:  # the non-DCP optimizer files hold DTensors on the meshes of the save: keep them
        reset_groups()
        initialize_model_parallel(tp, cp, timeout=600, expert_parallel_size=ep_load)
    ref = local_states(*build(tp, cp, fs, ep_load, seed=1))
    model, optimizer = build(tp, cp, fs, ep_load, seed=2)
    if compare(local_states(model, optimizer), ref, 'seed 2') == []:
        errors.append("the models built from seeds 1 and 2 are the same: the test checks nothing")
    for non_dcp in (False, True) if same_ep else (False,):
        load(model, optimizer, root, non_dcp, fs)
        errors += compare(local_states(model, optimizer), ref, 'non-DCP optimizer' if non_dcp else 'DCP')
    if dist.get_rank() == 0:
        cfg = SimpleNamespace(expert_parallel_size=ep_load, model_parallel_size=tp)
        try:
            check_optimizer_files_expert_parallel_size(root, cfg)
            if not same_ep:
                errors.append(f"non-DCP optimizer files saved at EP {ep_save} accepted at EP {ep_load}")
        except RuntimeError as e:
            if same_ep:
                errors.append(f"non-DCP optimizer files refused at the same EP: {e}")
    reset_groups()
    return errors


def reset_groups():
    groups = {id(group): group for name in dir(parallel_state) if name.endswith('_GROUP')
              for group in [getattr(parallel_state, name)] if group is not None}
    for group in groups.values():
        dist.destroy_process_group(group)
    destroy_model_parallel()


def inject(bug: str):
    if bug == 'noconvert':
        expert_checkpoint.get_expert_outer_world_size = lambda: 1
    elif bug == 'master':
        xllm_utils.get_expert_outer_world_size = lambda: 1
    else:
        assert bug is None, f"unknown bug {bug}"


def main(configs=None, bug=None):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    inject(bug)

    failed = False
    for tp, cp, fs, ep_save, ep_load in CONFIGS if configs is None else configs:
        root = [Path(tempfile.mkdtemp(prefix='expert_ckpt_')) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(root)
        root = root[0]
        try:
            errors = check(tp, cp, fs, ep_save, ep_load, root)
        except (Exception, CheckpointException) as e:  # report and fail on every rank rather than hang the others
            errors = [f"{type(e).__name__}: {e}"]
            if parallel_state.model_parallel_is_initialized():
                reset_groups()
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(root, ignore_errors=True)

        bad = torch.tensor(len(errors), device='cuda')
        dist.all_reduce(bad)
        failed |= bad.item() > 0
        for line in errors[:6]:
            print(f"rank {dist.get_rank()}: tp={tp} cp={cp} fs={fs} ep {ep_save} -> {ep_load}: {line}", flush=True)
        if dist.get_rank() == 0:
            print(f"tp={tp} cp={cp} fs={fs} ep {ep_save} -> {ep_load}: {'ok' if bad.item() == 0 else 'FAIL'}",
                  flush=True)

    if dist.get_rank() == 0:
        print('FAILED' if failed else 'passed', flush=True)
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    fire.Fire(main)
