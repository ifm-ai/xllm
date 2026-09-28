import sys
import math
import datetime
from typing import Dict, Tuple, Any
import logging
from pathlib import Path
import contextlib
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel, fully_shard
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)

logger = logging.getLogger()


@contextlib.contextmanager
def create_on_gpu():
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device("cpu")


def initialize_logger() -> logging.Logger:
    # log everything
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    # set stream handlers
    logger.handlers.clear()
    assert len(logger.handlers) == 0, logger.handlers
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    return logger


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensor_parallel_rank: int,
        tensor_parallel_size: int,
    ) -> None:
        super(Linear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        # Divide the weight matrix along the last dimension.
        assert out_features % tensor_parallel_size == 0
        self.output_size_per_partition = out_features // tensor_parallel_size
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))

        # Initialize master weight
        master_weight = torch.empty(out_features, in_features, dtype=self.weight.dtype, requires_grad=False)
        nn.init.kaiming_normal_(master_weight, a=math.sqrt(5.0))

        # Split and copy
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        my_weight = weight_list[tensor_parallel_rank]

        with torch.no_grad():
            self.weight.copy_(my_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return F.linear(x, self.weight)


def init_torch_distributed(timeout: int = 1800) -> Tuple[int, int]:
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
    """
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    logger.info(f"Run launched with torchrun, local rank: {local_rank}")

    # set GPU device
    assert 0 <= local_rank < 8
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
        timeout=datetime.timedelta(seconds=timeout),
    )

    assert global_rank == torch.distributed.get_rank()
    assert world_size == torch.distributed.get_world_size()

    # sanity check
    assert 0 <= local_rank <= global_rank < world_size

    return global_rank, world_size


def get_parallel_ranks(
    global_rank: int,
    tensor_parallel_size: int,
    context_parallel_size: int
) -> Tuple[int, int, int]:
    tensor_parallel_rank = global_rank % tensor_parallel_size
    global_rank = global_rank // tensor_parallel_size
    context_parallel_rank = global_rank % context_parallel_size
    data_parallel_rank = global_rank // context_parallel_size

    return data_parallel_rank, context_parallel_rank, tensor_parallel_rank


def get_device_mesh(
    data_parallel_size: int,
    context_parallel_size: int,
    tensor_parallel_size: int,
) -> DeviceMesh:
    world_size = torch.distributed.get_world_size()
    assert world_size == data_parallel_size * context_parallel_size * tensor_parallel_size
    if context_parallel_size == 1:
        if tensor_parallel_size == 1:
            mesh = torch.arange(world_size)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("dp",),
            )
        else:
            mesh = torch.arange(world_size).view(data_parallel_size, tensor_parallel_size)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("dp", "tp"),
            )["dp"]
    else:
        if tensor_parallel_size == 1:
            mesh = torch.arange(world_size).view(data_parallel_size, context_parallel_size)
            mesh = mesh.swapdims(0, 1)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("cp", "dp"),
            )
        else:
            mesh = torch.arange(world_size).view(data_parallel_size, context_parallel_size, tensor_parallel_size)
            mesh = mesh.swapdims(0, 1)
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=mesh,
                mesh_dim_names=("cp", "dp", "tp"),
            )["cp", "dp"]

    return device_mesh


def build_model_fsdp1(
    in_features: int,
    out_features: int,
    data_parallel_size: int,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    context_parallel_size: int
) -> nn.Module:
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    if context_parallel_size == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD

    device_mesh = get_device_mesh(data_parallel_size, context_parallel_size, tensor_parallel_size)
    mixed_precision = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    fsdp1_cfg = {
        "sharding_strategy": sharding_strategy,
        "mixed_precision": mixed_precision,
        "sync_module_states": False,
        "use_orig_params": False,  # flatten parameters
        "device_mesh": device_mesh
    }
    with create_on_gpu():
        with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp1_cfg):
            model = Linear(in_features, out_features, tensor_parallel_rank, tensor_parallel_size)
            model = wrap(model.cuda())
            model.train()

        return model


def build_model_fsdp2(
    in_features: int,
    out_features: int,
    data_parallel_size: int,
    tensor_parallel_rank: int,
    tensor_parallel_size: int,
    context_parallel_size: int
) -> nn.Module:
    from xllm.distributed.wrap import enable_wrap, wrap

    device_mesh = get_device_mesh(data_parallel_size, context_parallel_size, tensor_parallel_size)
    mixed_precision = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)

    fsdp2_cfg = {
        "mesh": device_mesh,
        "reshard_after_forward": True,
        "mp_policy": mixed_precision,
    }
    with create_on_gpu():
        with enable_wrap(**fsdp2_cfg):
            model = Linear(in_features, out_features, tensor_parallel_rank, tensor_parallel_size)
            model = wrap(model.cuda())
            model.train()

    return model


def get_model_state(model, full_state: bool = False) -> Dict[str, Any]:
    state_dict_options = StateDictOptions(
        full_state_dict=full_state,
        cpu_offload=True,
    )
    return get_model_state_dict(model, options=state_dict_options)


def set_model_state(model, model_state_dict, full_state: bool = False):
    state_dict_options = StateDictOptions(
        full_state_dict=full_state,
        cpu_offload=True,
    )
    set_model_state_dict(model, model_state_dict, options=state_dict_options)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    initialize_logger()

    global_rank, world_size = init_torch_distributed()

    tensor_parallel_size = 2
    context_parallel_size = 2
    data_parallel_size = world_size // (tensor_parallel_size * context_parallel_size)

    dp_rank, cp_rank, tp_rank = get_parallel_ranks(global_rank, tensor_parallel_size, context_parallel_size)

    logger.info(
        f"Global rank: {global_rank} -- "
        f"data parallel rank: {dp_rank}/{data_parallel_size} -- "
        f"tensor parallel rank: {tp_rank}/{tensor_parallel_size} -- "
        f"context parallel rank: {cp_rank}/{context_parallel_size}"
    )

    in_features = 4
    out_features = 10
    ddp_backend = 'fsdp2'

    if ddp_backend == 'fsdp1':
        model = build_model_fsdp1(
            in_features, out_features, data_parallel_size, tp_rank, tensor_parallel_size, context_parallel_size
        )
    elif ddp_backend == 'fsdp2':
        model = build_model_fsdp2(
            in_features, out_features, data_parallel_size, tp_rank, tensor_parallel_size, context_parallel_size
        )
    else:
        raise ValueError(f"Unknown DDP: {ddp_backend}")

    for name, pa in model.named_parameters():
        print(name)
        print(pa)

    x = torch.ones(2, in_features).cuda()
    y = model(x).sum()
    print(y)

    checkpoint_dir = Path("saved_models/ckpt")
    model_path = checkpoint_dir / f"sharded_model.tp{tp_rank:02d}"
    sharded_model_state = get_model_state(model, full_state=False)
    groups = torch.LongTensor(range(world_size)).reshape(-1, tensor_parallel_size)
    hsdp_group = None
    for k in range(tensor_parallel_size):
        ranks = groups[:, k].tolist()
        group = torch.distributed.new_group(ranks)
        if k == tp_rank:
            hsdp_group = group
    assert hsdp_group is not None

    logger.info(
        f"Global rank: {global_rank} -- "
        f"tensor parallel rank: {tp_rank}/{tensor_parallel_size} -- "
        f"hsdp group: {groups}"
    )
    dcp.save(sharded_model_state, checkpoint_id=model_path, process_group=hsdp_group)

    logger.info(f"Global rank: {global_rank} -- Checkpoint done.")

    torch.distributed.destroy_process_group()
