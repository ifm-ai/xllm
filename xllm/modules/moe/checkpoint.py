"""The routed experts in the sharded (DCP) checkpoints.

With EP = TP x E_out, the GPU at TP rank k in replica o of its expert group holds the experts
k * N / TP + o * N / EP + i, i < N / EP, sharded by FSDP over its expert mesh. Their parameters are DTensors on
that mesh: their global shape covers the N / EP experts of the replica only, under the same names on every
replica. Saved as they are, the E_out replicas would write different experts at the same offsets.

In the state dicts saved and loaded with DCP, they are DTensors on a checkpoint mesh instead: the expert mesh
with the E_out replicas of the expert group as an outer dimension, on which they are sharded too. Their global
shape is then that of EP = TP, the N / TP experts of the TP rank in order, and each replica reads and writes its
own rows. Checkpoints thus have the same keys and global shapes at every EP with the same TP, so DCP loads them
at another EP, and they are unchanged at EP = TP, where nothing is converted.
"""
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from xllm.distributed import (
    get_context_parallel_world_size,
    get_model_parallel_world_size,
    get_expert_outer_group,
    get_expert_outer_rank,
    get_expert_outer_world_size,
)
from .expert import Expert

OUTER_DIM = "expert_outer"


def routed_expert_fqns(model: nn.Module) -> List[str]:
    """The state dict keys of the routed expert weights of a model (FSDP2 keeps the parameter names)."""
    return [
        f"{name}.{param_name}" if name else param_name
        for name, module in model.named_modules() if isinstance(module, Expert)
        for param_name, _ in module.named_parameters(recurse=False)
    ]


def _checkpoint_mesh(expert_mesh: DeviceMesh) -> DeviceMesh:
    """The expert mesh of this GPU with the E_out replicas of its expert group as an outer dimension. Only used
    to place the experts' rows in the checkpoints: it has no process groups."""
    e_out, outer_rank = get_expert_outer_world_size(), get_expert_outer_rank()
    # ranks are in order expert group x replica x CP x TP: the next replica is CP x TP ranks further
    stride = get_context_parallel_world_size() * get_model_parallel_world_size()
    ranks = expert_mesh.mesh
    mesh = torch.stack([ranks + (o - outer_rank) * stride for o in range(e_out)])
    # the GPUs at this GPU's place in the expert mesh of every replica must be its expert outer group
    outer = mesh[(slice(None), *expert_mesh.get_coordinate())].tolist()
    assert outer == dist.get_process_group_ranks(get_expert_outer_group()), (outer, expert_mesh)
    names = expert_mesh.mesh_dim_names or tuple(f"dim{i}" for i in range(expert_mesh.ndim))
    return DeviceMesh(expert_mesh.device_type, mesh, mesh_dim_names=(OUTER_DIM, *names), _init_backend=False)


def _contiguous_stride(shape) -> tuple:
    return torch.empty(shape, device="meta").stride()


def _to_checkpoint(value: DTensor, meshes: Dict[int, DeviceMesh]) -> DTensor:
    """The expert weight (or state) `value` of this replica as its rows of the TP rank's experts."""
    expert_mesh = value.device_mesh
    if id(expert_mesh) not in meshes:
        meshes[id(expert_mesh)] = _checkpoint_mesh(expert_mesh)
    mesh = meshes[id(expert_mesh)]
    assert all((type(p) is Shard and p.dim == 0) or isinstance(p, Replicate) for p in value.placements), \
        value.placements
    shape = torch.Size((mesh.size(0) * value.shape[0], *value.shape[1:]))
    placements = (Shard(0), *value.placements)
    spec = DTensorSpec(mesh, placements, tensor_meta=TensorMeta(shape, _contiguous_stride(shape), value.dtype))
    out = DTensor(value._local_tensor, spec, requires_grad=False)
    # the rows of this GPU: its replica's block of the TP rank's experts, then its FSDP shard in the block
    local_shape, offset = compute_local_shape_and_global_offset(value.shape, expert_mesh, value.placements)
    ckpt_shape, ckpt_offset = compute_local_shape_and_global_offset(shape, mesh, placements)
    expected = (offset[0] + get_expert_outer_rank() * value.shape[0], *offset[1:])
    assert tuple(ckpt_shape) == tuple(local_shape) == tuple(value._local_tensor.shape), (ckpt_shape, local_shape)
    assert tuple(ckpt_offset) == expected, (ckpt_offset, expected)
    return out


def _from_checkpoint(value: Any, param: DTensor, key: str) -> DTensor:
    """The expert weight (or state) `value` of the checkpoint back on the mesh of the parameter `param`."""
    if not (isinstance(value, DTensor) and value.device_mesh.mesh_dim_names
            and value.device_mesh.mesh_dim_names[0] == OUTER_DIM):
        raise RuntimeError(f"{key}: expected an expert state on the checkpoint mesh, got {type(value).__name__}")
    assert value.placements[1:] == param.placements, (key, value.placements, param.placements)
    assert value.shape[0] == value.device_mesh.size(0) * param.shape[0], (key, value.shape, param.shape)
    spec = DTensorSpec(
        param.device_mesh, param.placements, tensor_meta=TensorMeta(param.shape, param.stride(), value.dtype)
    )
    return DTensor(value._local_tensor, spec, requires_grad=False)


def experts_to_checkpoint(model: nn.Module, model_state: Dict[str, Any]) -> Dict[str, Any]:
    """Place the routed experts of a sharded model state dict at their rows in the TP rank's experts (EP > TP)."""
    if get_expert_outer_world_size() == 1:
        return model_state
    meshes: Dict[int, DeviceMesh] = {}
    for key in routed_expert_fqns(model):
        assert isinstance(model_state[key], DTensor), (key, type(model_state[key]))
        model_state[key] = _to_checkpoint(model_state[key], meshes)
    return model_state


def experts_from_checkpoint(model: nn.Module, model_state: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of experts_to_checkpoint, after loading."""
    if get_expert_outer_world_size() == 1:
        return model_state
    for key in routed_expert_fqns(model):
        model_state[key] = _from_checkpoint(model_state[key], model.get_parameter(key), key)
    return model_state


def expert_states_to_checkpoint(model: nn.Module, optim_state: Dict[str, Any]) -> Dict[str, Any]:
    """Place the optimizer states of the routed experts (e.g. AdamW's moments; the scalar step is the same on
    every GPU) of a sharded optimizer state dict at their rows in the TP rank's experts (EP > TP)."""
    if get_expert_outer_world_size() == 1:
        return optim_state
    meshes: Dict[int, DeviceMesh] = {}
    states = optim_state["state"]
    for key in routed_expert_fqns(model):
        for name, value in states.get(key, {}).items():
            if isinstance(value, DTensor):
                states[key][name] = _to_checkpoint(value, meshes)
            else:
                assert not isinstance(value, torch.Tensor) or value.dim() == 0, (key, name, type(value))
    return optim_state


def expert_states_from_checkpoint(model: nn.Module, optim_state: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of expert_states_to_checkpoint, after loading."""
    if get_expert_outer_world_size() == 1:
        return optim_state
    states = optim_state["state"]
    for key in routed_expert_fqns(model):
        param = model.get_parameter(key)
        for name, value in states.get(key, {}).items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                states[key][name] = _from_checkpoint(value, param, f"{key}.{name}")
    return optim_state
