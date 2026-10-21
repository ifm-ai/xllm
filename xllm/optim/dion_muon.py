"""Muon for hidden projections; Dion's AdamW for other trainable weights."""

import hashlib
import json
from logging import getLogger

import torch
import torch.distributed as dist
from torch import nn

from xllm.distributed import get_context_parallel_world_size, get_data_parallel_world_size
from xllm.modules.model_parallel import ColumnParallelLinear, RowParallelLinear


logger = getLogger()

# Contract fields that make a saved optimizer state incompatible. Hyperparameters
# and the scheduler horizon are deliberately excluded so they can change on resume.
CONTRACT_KEYS = (
    'optimizer',
    'groups',
    'mesh_shape',
    'mesh_dims',
    'context_parallel_size',
    'data_parallel_size',
)
# Per-group hyperparameters taken from the config, not the checkpoint, on resume.
# The LR follows the scheduler, whose base LRs also come from the config.
CONFIGURED_HPARAMS = ('weight_decay', 'mu', 'nesterov', 'adjust_lr', 'beta1', 'beta2', 'epsilon')


@torch.no_grad()
def synchronize_replicas(optimizer, args, kwargs):
    """Keep weights and moments identical across HSDP/CP replicas after NS."""
    tensors = []
    for group in optimizer.param_groups:
        for parameter in group['params']:
            tensors.append(parameter.detach().to_local())
            for value in optimizer.state[parameter].values():
                if isinstance(value, torch.Tensor):
                    tensors.append(value.to_local() if hasattr(value, 'to_local') else value)
    dist._broadcast_coalesced(optimizer._xllm_replica_group, tensors, 64 * 1024**2, 0)


def parameter_groups(model, cfg):
    """Use module semantics: Gekko's 2-D depthwise kernels stay in AdamW.

    Groups: Muon for hidden projections; AdamW with weight decay for other
    matrices; AdamW without weight decay for norms, biases and other vectors
    (plus the embedding and readout with no_decay_embed_readout). Empty groups
    are dropped.
    """
    if model.arch not in ('transformer', 'gekko') or model.num_experts or model.num_values:
        raise ValueError('dion_muon requires a Transformer or Gekko with dense FFNs')
    muon_weight_decay = cfg.weight_decay if cfg.muon_weight_decay is None else cfg.muon_weight_decay
    modules = dict(model.named_modules())
    groups = [
        {'params': [], 'group_id': 'muon', 'algorithm': 'muon', 'lr': cfg.muon_lr,
         'weight_decay': muon_weight_decay},
        {'params': [], 'group_id': 'adamw', 'algorithm': 'adamw', 'lr': cfg.lr,
         'weight_decay': cfg.weight_decay},
        {'params': [], 'group_id': 'adamw_no_decay', 'algorithm': 'adamw', 'lr': cfg.lr,
         'weight_decay': 0.0},
    ]
    names = [[], [], []]
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f'dion_muon does not support tied parameter {name}')
        seen.add(id(parameter))
        if parameter.ndim > 2:
            raise ValueError(f'Unsupported parameter {name}: shape {tuple(parameter.shape)}')
        owner, _, leaf = name.rpartition('.')
        projection = isinstance(
            modules[owner], (nn.Linear, ColumnParallelLinear, RowParallelLinear)
        )
        muon = name.startswith('layers.') and projection and leaf == 'weight'
        embed_or_readout = name.startswith('embed.') or name == 'output.output.weight'
        if muon:
            index = 0
        elif parameter.ndim < 2 or (cfg.no_decay_embed_readout and embed_or_readout):
            index = 2
        else:
            index = 1
        groups[index]['params'].append(parameter)
        names[index].append((name, list(parameter.shape)))
    if not groups[0]['params']:
        raise ValueError('No trainable hidden projection matrices for Muon')
    groups, names = zip(*[(g, n) for g, n in zip(groups, names) if g['params']])
    contract = [
        {
            'group_id': group['group_id'],
            'algorithm': group['algorithm'],
            'membership_sha256': hashlib.sha256(
                json.dumps(group_names, separators=(',', ':')).encode()
            ).hexdigest(),
        }
        for group, group_names in zip(groups, names)
    ]
    return list(groups), contract


def build_dion_muon(model, cfg):
    try:
        from dion import Muon
    except ImportError as error:
        raise ImportError('Install requirements-dion.txt to use optimizer=dion_muon') from error
    mesh = getattr(model, '_xllm_dion_mesh', None)
    if mesh is None:
        raise ValueError('dion_muon requires a model built with FSDP2')
    groups, contract = parameter_groups(model, cfg)
    # Tensor operations may reuse an equivalent DeviceMesh when a second
    # FSDP2 model is built in-process. Use the parameters' actual shard group,
    # not a newly created group with the same rank layout.
    parameter_mesh = groups[0]['params'][0].device_mesh['dp']
    if parameter_mesh.mesh.tolist() != mesh.mesh.tolist():
        raise ValueError('Muon parameters do not match the native FSDP2 shard layout')
    mesh = parameter_mesh
    if any(p.device_mesh['dp'].get_group() != mesh.get_group() for p in groups[0]['params']):
        raise ValueError('Muon projection matrices must share one FSDP2 shard group')
    model._xllm_dion_mesh = mesh
    optimizer = Muon(
        groups,
        distributed_mesh=mesh,
        mu=cfg.muon_momentum,
        betas=(cfg.beta1, cfg.beta2),
        epsilon=cfg.epsilon,
        nesterov=cfg.muon_nesterov,
        adjust_lr=cfg.muon_adjust_lr,
        use_polar_express=cfg.muon_orthogonalizer == 'polar_express',
        use_gram_newton_schulz=cfg.muon_use_gram_newton_schulz,
        use_triton=cfg.muon_use_triton,
    )
    # FSDP2 already all-reduces gradients across CP/HSDP replicas, so replicas
    # compute the same update; the broadcast is an opt-in guard against drift.
    full_mesh = groups[0]['params'][0].device_mesh
    replicated = 'cp' in full_mesh.mesh_dim_names and full_mesh['cp'].size() > 1
    if replicated and cfg.muon_replica_sync:
        optimizer._xllm_replica_group = full_mesh['cp'].get_group()
        optimizer.register_step_post_hook(synchronize_replicas)
    optimizer._xllm_optimizer_contract = {
        'optimizer': 'dion_muon',
        'groups': contract,
        'mesh_shape': list(model._xllm_fsdp_mesh.mesh.shape),
        'mesh_dims': list(model._xllm_fsdp_mesh.mesh_dim_names),
        'context_parallel_size': get_context_parallel_world_size(),
        'data_parallel_size': get_data_parallel_world_size(),
    }
    for group in optimizer.param_groups:
        logger.info(
            f"Param group {group['group_id']}: {len(group['params'])} tensors, "
            f"lr={group['lr']:.6g}, weight_decay={group['weight_decay']:.6g}"
        )
    return optimizer


def validate_optimizer_contract(optimizer, training_state):
    current = getattr(optimizer, '_xllm_optimizer_contract', None)
    saved = training_state.get('optimizer_contract')
    if current is None or saved is None:
        if current is not saved:
            raise ValueError('Optimizer checkpoint contract changed (optimizer type)')
        return
    changed = [key for key in CONTRACT_KEYS if current.get(key) != saved.get(key)]
    if changed:
        raise ValueError(
            f'Optimizer checkpoint contract changed ({", ".join(changed)}); '
            'load the weights only with base_model_dir and reload_base_optim_state=false'
        )


def get_group_hparams(optimizer):
    """Configured per-group hyperparameters that loading optimizer state would overwrite."""
    return [{k: group[k] for k in CONFIGURED_HPARAMS if k in group} for group in optimizer.param_groups]


def set_group_hparams(optimizer, group_hparams):
    """Restore hyperparameters saved by get_group_hparams after loading optimizer state."""
    for group, hparams in zip(optimizer.param_groups, group_hparams, strict=True):
        changed = {k: (group.get(k), v) for k, v in hparams.items() if group.get(k) != v}
        if changed:
            logger.info(f"Param group {group['group_id']}: using configured values over reloaded ones: {changed}")
        group.update(hparams)
