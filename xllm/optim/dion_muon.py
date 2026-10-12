"""Muon for hidden projections; Dion's AdamW for other trainable weights."""

import hashlib
import json

from torch import nn

from xllm.distributed import get_context_parallel_world_size, get_data_parallel_world_size
from xllm.modules.model_parallel import ColumnParallelLinear, RowParallelLinear


def parameter_groups(model, cfg):
    """Use module semantics: Gekko's 2-D depthwise kernels stay in AdamW."""
    if model.arch not in ('transformer', 'gekko') or model.num_experts or model.num_values:
        raise ValueError('dion_muon requires a Transformer or Gekko with dense FFNs')
    modules = dict(model.named_modules())
    groups = [
        {'params': [], 'group_id': 'muon', 'algorithm': 'muon', 'lr': cfg.muon_lr},
        {'params': [], 'group_id': 'adamw', 'algorithm': 'adamw', 'lr': cfg.lr},
    ]
    membership = [[], []]
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
        if muon and parameter.ndim != 2:
            raise ValueError(f'Muon projection {name} must be a matrix')
        index = 0 if muon else 1
        groups[index]['params'].append(parameter)
        membership[index].append((name, list(parameter.shape)))
    if not groups[0]['params']:
        raise ValueError('No trainable hidden projection matrices for Muon')
    contract = [
        {
            'algorithm': group['algorithm'],
            'membership_sha256': hashlib.sha256(
                json.dumps(names, separators=(',', ':')).encode()
            ).hexdigest(),
        }
        for group, names in zip(groups, membership)
    ]
    return groups, contract


def build_dion_muon(model, cfg):
    try:
        from dion import Muon
    except ImportError as error:
        raise ImportError('Install requirements-dion.txt to use optimizer=dion_muon') from error
    mesh = getattr(model, '_xllm_dion_mesh', None)
    if mesh is None:
        raise ValueError('dion_muon requires a model built with FSDP2')
    groups, membership = parameter_groups(model, cfg)
    optimizer = Muon(
        groups,
        distributed_mesh=mesh,
        mu=cfg.muon_momentum,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        epsilon=cfg.epsilon,
        nesterov=cfg.muon_nesterov,
        adjust_lr=cfg.muon_adjust_lr,
        use_polar_express=cfg.muon_orthogonalizer == 'polar_express',
        use_gram_newton_schulz=False,
        use_triton=False,
    )
    optimizer._xllm_optimizer_contract = {
        'optimizer': 'dion_muon',
        'groups': membership,
        'mesh_shape': list(model._xllm_fsdp_mesh.mesh.shape),
        'mesh_dims': list(model._xllm_fsdp_mesh.mesh_dim_names),
        'context_parallel_size': get_context_parallel_world_size(),
        'data_parallel_size': get_data_parallel_world_size(),
        'orthogonalizer': cfg.muon_orthogonalizer,
        'recipe': {
            name: getattr(cfg, name)
            for name in (
                'lr',
                'muon_lr',
                'muon_momentum',
                'muon_nesterov',
                'muon_adjust_lr',
                'beta1',
                'beta2',
                'weight_decay',
                'epsilon',
            )
        },
    }
    return optimizer


def validate_optimizer_contract(optimizer, training_state):
    current = getattr(optimizer, '_xllm_optimizer_contract', None)
    saved = training_state.get('optimizer_contract')
    if current != saved:
        raise ValueError(
            'Optimizer checkpoint contract changed (optimizer, parameters or topology)'
        )
