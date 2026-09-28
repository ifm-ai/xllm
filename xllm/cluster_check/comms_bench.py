import os
from dataclasses import dataclass
from logging import getLogger
import socket
import time
from datetime import timedelta
from typing import List, Optional

import numpy as np
import torch
from torch import distributed as dist
from torch.distributed import ProcessGroup

from xllm.distributed.slurm import get_world_size, get_global_rank
from xllm.distributed.utils import reduce_scalar


TENSOR_SIZE = int(2**28)  # ~256M elements = 1GB message size in fp32
N_GPUS_PER_NODE = int(os.getenv("N_GPUS_PER_NODE", "8"))
NNODES_PER_GROUP = 4
N_REPEAT = 10

logger = getLogger()


@dataclass
class CollectiveCheck:
    global_bandwidth: float = -1
    avg_bandwidth_in_subgroup: float = -1
    max_bandwidth_in_subgroup: float = -1
    min_bandwidth_in_subgroup: float = -1
    succeed: bool = False


@torch.no_grad()
def collective_bench(
    tensor_size: int = TENSOR_SIZE,
    n_nodes_per_group: int = 4,
    ngpu_per_node: int = 8,
    n_repeats: int = N_REPEAT,
) -> CollectiveCheck:
    word_size = get_world_size()
    if word_size == 1:
        logger.warning(f"collective_bench: skipping since world_size is 1")
        return CollectiveCheck()

    assert word_size < ngpu_per_node or word_size % ngpu_per_node == 0, (word_size, ngpu_per_node)
    assert n_nodes_per_group > 0, n_nodes_per_group

    total_nodes = word_size // ngpu_per_node
    if total_nodes < n_nodes_per_group:
        subgroup_size = word_size
        logger.warning(
            f"collective_bench: setting subgroup size to world_size: {word_size}"
        )
    else:
        assert total_nodes % n_nodes_per_group == 0, (total_nodes, n_nodes_per_group)
        subgroup_size = ngpu_per_node * n_nodes_per_group

    gpus_by_group = np.arange(get_world_size()).reshape((-1, subgroup_size)).tolist()
    subgroup_bandwidth = measure_bus_bandwidth_on_groups(gpus_by_group, n_repeats, tensor_size)
    logger.info(
        f"collective_bench: subgroup bandwidth (over {n_repeats} tests): "
        f"min: {subgroup_bandwidth.min_bandwidth:.02f}GB/s, "
        f"max: {subgroup_bandwidth.max_bandwidth:.02f}GB/s, "
        f"avg: {subgroup_bandwidth.avg_bandwidth:.02f}GB/s "
        f"for host: {socket.gethostname()}"
    )

    global_bandwidth = measure_bus_bandwidth_on_groups([list(range(word_size))], n_repeats, tensor_size)
    logger.info(
        f"collective_bench: global   bandwidth (over {n_repeats} tests): "
        f"min: {global_bandwidth.min_bandwidth:.02f}GB/s, "
        f"max: {global_bandwidth.max_bandwidth:.02f}GB/s, "
        f"avg: {global_bandwidth.avg_bandwidth:.02f}GB/s "
        f"for host: {socket.gethostname()}"
    )

    bench = CollectiveCheck(
        succeed=True,
        global_bandwidth=global_bandwidth.avg_bandwidth,
        avg_bandwidth_in_subgroup=subgroup_bandwidth.avg_bandwidth,
        min_bandwidth_in_subgroup=subgroup_bandwidth.min_bandwidth,
        max_bandwidth_in_subgroup=subgroup_bandwidth.max_bandwidth,
    )

    under_150 = 1.0 if bench.avg_bandwidth_in_subgroup < 150 else 0.0
    under_250 = 1.0 if bench.avg_bandwidth_in_subgroup < 250 else 0.0

    subgroup_bandwidth = subgroup_bandwidth.avg_bandwidth
    global_bandwidth = global_bandwidth.avg_bandwidth
    avg_global_bandwidth=reduce_scalar(global_bandwidth, 'mean')
    min_global_bandwidth=reduce_scalar(global_bandwidth, 'min')
    max_global_bandwidth=reduce_scalar(global_bandwidth, 'max')
    avg_subgroup_bandwidth =reduce_scalar(subgroup_bandwidth, 'mean')
    min_subgroup_bandwidth =reduce_scalar(subgroup_bandwidth, 'min')
    max_subgroup_bandwidth = reduce_scalar(subgroup_bandwidth, 'max')
    n_groups_with_busbw_under_250=int(reduce_scalar(under_250, 'sum')) // subgroup_size
    n_groups_with_busbw_under_150=int(reduce_scalar(under_150, 'sum')) // subgroup_size

    logger.info(
        f"collective_bench: aggregated bandwidth (over {word_size // subgroup_size} subgroups): "
        f"min global bandwidth: {min_global_bandwidth:.02f}GB/s, "
        f"max global bandwidth: {max_global_bandwidth:.02f}GB/s, "
        f"avg global bandwidth: {avg_global_bandwidth:.02f}GB/s, "
        f"min subgroup bandwidth: {min_subgroup_bandwidth:.02f}GB/s, "
        f"max subgroup bandwidth: {max_subgroup_bandwidth:.02f}GB/s, "
        f"avg subgroup bandwidth: {avg_subgroup_bandwidth:.02f}GB/s, "
        f"{n_groups_with_busbw_under_250} groups with bandwidth < 250GB, "
        f"{n_groups_with_busbw_under_150} groups with bandwidth < 150GB."
    )

    return bench


@dataclass
class BusBandwidth:
    avg_bandwidth: float
    min_bandwidth: float
    max_bandwidth: float


def measure_bus_bandwidth_on_groups(
    gpus_by_group: List[List[int]], n_repeats: int, tensor_size: int, timeout: int = 1800,
) -> BusBandwidth:
    to_gather = torch.randn(tensor_size, device="cuda", dtype=torch.float32)
    my_group: Optional[ProcessGroup] = None
    for gpus in gpus_by_group:
        assert len(gpus) > 0
        group = torch.distributed.new_group(ranks=gpus, timeout=timedelta(timeout))
        if get_global_rank() in gpus:
            assert my_group is None
            my_group = group

    assert my_group is not None
    subgroup_size = torch.distributed.get_world_size(my_group)
    torch.distributed.all_reduce(to_gather, group=my_group)
    torch.cuda.synchronize()
    all_speeds = []
    for n in range(n_repeats):
        torch.distributed.barrier(group=my_group)
        s = time.time()
        torch.distributed.all_reduce(to_gather, group=my_group)
        torch.cuda.synchronize()
        delta = time.time() - s
        size_b = tensor_size * 4
        speed = size_b / delta / (1024**3)  # GB/s
        all_speeds.append(speed)

    # avg over n_repeats
    bandwidth = np.mean([s * (2 * (subgroup_size - 1) / subgroup_size) for s in all_speeds]).item()

    bandwidths = []
    for op in [dist.ReduceOp.AVG, dist.ReduceOp.MIN, dist.ReduceOp.MAX]:
        busbw = torch.tensor(bandwidth).float().cuda()
        torch.distributed.all_reduce(busbw, op=op, group=my_group)
        bandwidths.append(busbw.item())

    avg_bandwidth, min_bandwidth, max_bandwidth = bandwidths
    dist.destroy_process_group(my_group)

    return BusBandwidth(avg_bandwidth=avg_bandwidth, min_bandwidth=min_bandwidth, max_bandwidth=max_bandwidth)
