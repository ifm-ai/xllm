"""Muon for hidden projections; Dion's AdamW for other trainable weights."""

import hashlib
import json
from copy import deepcopy

import torch
import torch.distributed as dist
from torch import nn

from xllm.distributed import get_context_parallel_world_size, get_data_parallel_world_size
from xllm.modules.model_parallel import ColumnParallelLinear, RowParallelLinear
from .weight_decay import no_weight_decay_param_ids


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


def _is_muon(name, modules):
    owner, _, leaf = name.rpartition('.')
    projection = isinstance(modules[owner], (nn.Linear, ColumnParallelLinear, RowParallelLinear))
    return name.startswith('layers.') and projection and leaf == 'weight'


def param_role(name, muon):
    """Role of a parameter by name (RolesConf in xllm/config.py)."""
    if muon:
        return 'hidden_matrix'
    if name.startswith('embed.'):
        return 'embed'
    if name.startswith('output.output.'):
        return 'readout'
    if name.startswith('output.'):
        return 'final_norm'
    if name.startswith('layers.'):
        return 'hidden_vector'
    raise ValueError(f'No role for parameter {name}')


MATRIX_SUBROLES = {'wq': 'attn_in', 'wk': 'attn_in', 'wv': 'attn_in', 'wr': 'attn_in', 'wg': 'attn_in',
                   'wo': 'attn_out', 'fc1': 'ffn_in', 'fc3': 'ffn_in', 'fc2': 'ffn_out'}


def param_subrole(name, role):
    """Sub-role of a hidden_matrix / hidden_vector parameter by name (RolesConf in xllm/config.py)."""
    owner, _, leaf = name.rpartition('.')
    module = owner.rpartition('.')[2]
    if role == 'hidden_matrix':
        if module not in MATRIX_SUBROLES:
            raise ValueError(f'No hidden_matrix sub-role for {name}')
        return MATRIX_SUBROLES[module]
    if role == 'hidden_vector':
        if module.endswith('_conv'):
            return 'vec_conv'
        if 'norm' in module:
            return 'vec_norm'
        if module in MATRIX_SUBROLES and leaf == 'bias':
            return 'vec_bias'
        raise ValueError(f'No hidden_vector sub-role for {name}')
    return None


def role_parameter_groups(model, cfg):
    """Muon groups for hidden_matrix and AdamW groups per (role, decayed): lr, epsilon and weight decay are the
    configured values times cfg.role_factor (optim.roles.* x mup.*). disable_wd_{emb,out} still zero the decay.
    A parent with a sub-role set (optim.roles.attn_in etc.) gets one group per sub-role. Groups are created in
    parameter order, the Muon group first when hidden_matrix is not split."""
    modules = dict(model.named_modules())
    no_decay_ids = no_weight_decay_param_ids(model, cfg)
    split = {parent: cfg.roles.split(parent) for parent in ('hidden_matrix', 'hidden_vector')}
    groups, membership = [], []
    index = {}
    if not split['hidden_matrix']:
        index[('hidden_matrix', None, True)] = 0
        groups.append({
            'params': [], 'group_id': 'muon', 'algorithm': 'muon',
            'lr': cfg.muon_lr * cfg.role_factor('hidden_matrix', 'lr'),
            'weight_decay': cfg.weight_decay * cfg.role_factor('hidden_matrix', 'weight_decay'),
        })
        membership.append([])
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f'dion_muon does not support tied parameter {name}')
        seen.add(id(parameter))
        if parameter.ndim > 2:
            raise ValueError(f'Unsupported parameter {name}: shape {tuple(parameter.shape)}')
        muon = _is_muon(name, modules)
        if muon and id(parameter) in no_decay_ids:
            raise ValueError(f'No-weight-decay parameter {name} was routed to Muon')
        role = param_role(name, muon)
        sub = param_subrole(name, role) if split.get(role) else None
        decayed = id(parameter) not in no_decay_ids
        key = index.get((role, sub, decayed))
        if key is None:
            key = index[(role, sub, decayed)] = len(groups)
            if muon:
                groups.append({
                    'params': [], 'group_id': f'muon_{sub}', 'algorithm': 'muon',
                    'lr': cfg.muon_lr * cfg.role_factor(role, 'lr', sub),
                    'weight_decay': cfg.weight_decay * cfg.role_factor(role, 'weight_decay', sub),
                })
            else:
                groups.append({
                    'params': [], 'group_id': f'adamw_{sub or role}' + ('' if decayed else '_no_wd'),
                    'algorithm': 'adamw',
                    'lr': cfg.lr * cfg.role_factor(role, 'lr', sub),
                    'epsilon': cfg.epsilon * cfg.role_factor(role, 'eps', sub),
                    'weight_decay': cfg.weight_decay * cfg.role_factor(role, 'weight_decay', sub) if decayed else 0.0,
                })
            membership.append([])
        groups[key]['params'].append(parameter)
        membership[key].append((name, list(parameter.shape)))
    if not any(g['params'] for g in groups if g['algorithm'] == 'muon'):
        raise ValueError('No trainable hidden projection matrices for Muon')
    return groups, membership


def parameter_groups(model, cfg):
    """Use module semantics: Gekko's 2-D depthwise kernels stay in AdamW."""
    if model.arch not in ('transformer', 'gekko') or model.num_experts or model.num_values:
        raise ValueError('dion_muon requires a Transformer or Gekko with dense FFNs')
    if cfg.role_groups():
        groups, membership = role_parameter_groups(model, cfg)
        return groups, _contract(groups, membership)
    modules = dict(model.named_modules())
    groups = [
        {'params': [], 'group_id': 'muon', 'algorithm': 'muon', 'lr': cfg.muon_lr},
        {'params': [], 'group_id': 'adamw', 'algorithm': 'adamw', 'lr': cfg.lr},
    ]
    membership = [[], []]
    # optim.disable_wd_{emb,out}: a third, AdamW-only group without weight decay. Only added
    # when a flag is set, so the default two-group layout (and checkpoints) are unchanged.
    no_decay_ids = no_weight_decay_param_ids(model, cfg)
    if no_decay_ids:
        groups.append({
            'params': [], 'group_id': 'adamw_no_wd', 'algorithm': 'adamw', 'lr': cfg.lr,
            'weight_decay': 0.0,
        })
        membership.append([])
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
        if id(parameter) in no_decay_ids:
            if muon:
                raise ValueError(f'No-weight-decay parameter {name} was routed to Muon')
            index = 2
        else:
            index = 0 if muon else 1
        groups[index]['params'].append(parameter)
        membership[index].append((name, list(parameter.shape)))
    if not groups[0]['params']:
        raise ValueError('No trainable hidden projection matrices for Muon')
    return groups, _contract(groups, membership)


def _contract(groups, membership):
    contract = [
        {
            'algorithm': group['algorithm'],
            'membership_sha256': hashlib.sha256(
                json.dumps(names, separators=(',', ':')).encode()
            ).hexdigest(),
        }
        for group, names in zip(groups, membership)
    ]
    return contract


def build_dion_muon(model, cfg):
    try:
        from dion import Muon
    except ImportError as error:
        raise ImportError('Install requirements-dion.txt to use optimizer=dion_muon') from error
    mesh = getattr(model, '_xllm_dion_mesh', None)
    if mesh is None:
        raise ValueError('dion_muon requires a model built with FSDP2')
    groups, membership = parameter_groups(model, cfg)
    if not dist.is_initialized() or dist.get_rank() == 0:
        for group in groups:
            print(f"dion_muon group {group['group_id']}: {len(group['params'])} params, lr {group['lr']:.4g}, "
                  f"wd {group.get('weight_decay', cfg.weight_decay):.4g}, eps {group.get('epsilon', cfg.epsilon):.3g}", flush=True)
    muon_params = [p for g in groups if g['algorithm'] == 'muon' for p in g['params']]
    # Tensor operations may reuse an equivalent DeviceMesh when a second
    # FSDP2 model is built in-process. Use the parameters' actual shard group,
    # not a newly created group with the same rank layout.
    parameter_mesh = muon_params[0].device_mesh['dp']
    if parameter_mesh.mesh.tolist() != mesh.mesh.tolist():
        raise ValueError('Muon parameters do not match the native FSDP2 shard layout')
    mesh = parameter_mesh
    if any(p.device_mesh['dp'].get_group() != mesh.get_group() for p in muon_params):
        raise ValueError('Muon projection matrices must share one FSDP2 shard group')
    model._xllm_dion_mesh = mesh
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
        use_gram_newton_schulz=cfg.muon_use_gram_newton_schulz,
        use_triton=cfg.muon_use_triton,
    )
    full_mesh = muon_params[0].device_mesh
    replicated = 'cp' in full_mesh.mesh_dim_names and full_mesh['cp'].size() > 1
    if replicated:
        optimizer._xllm_replica_group = full_mesh['cp'].get_group()
        optimizer.register_step_post_hook(synchronize_replicas)
    optimizer._xllm_optimizer_contract = {
        'optimizer': 'dion_muon',
        'groups': membership,
        'mesh_shape': list(model._xllm_fsdp_mesh.mesh.shape),
        'mesh_dims': list(model._xllm_fsdp_mesh.mesh_dim_names),
        'context_parallel_size': get_context_parallel_world_size(),
        'data_parallel_size': get_data_parallel_world_size(),
        'replica_sync': replicated,
        'orthogonalizer': cfg.muon_orthogonalizer,
        'recipe': {
            name: getattr(cfg, name)
            for name in (
                'lr',
                'muon_lr',
                'muon_momentum',
                'muon_nesterov',
                'muon_adjust_lr',
                'muon_use_gram_newton_schulz',
                'muon_use_triton',
                'beta1',
                'beta2',
                'weight_decay',
                'disable_wd_emb',
                'disable_wd_out',
                'epsilon',
            )
        },
    }
    return optimizer


def validate_optimizer_contract(optimizer, training_state):
    current = getattr(optimizer, '_xllm_optimizer_contract', None)
    saved = training_state.get('optimizer_contract')
    if saved is not None and 'recipe' in saved:
        # Older checkpoints always used the unaccelerated path.
        saved = deepcopy(saved)
        for name in ('muon_use_gram_newton_schulz', 'muon_use_triton', 'disable_wd_emb', 'disable_wd_out'):
            saved['recipe'].setdefault(name, False)
    if current != saved:
        raise ValueError(
            'Optimizer checkpoint contract changed (optimizer, parameters or topology)'
        )
