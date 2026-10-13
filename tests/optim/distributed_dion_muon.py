"""Real-kernel FSDP2/CP parity and checkpoint test; launch with torchrun.

Run CP=1 first with --save-reference; compare CP=2 with --reference using the
same DP size, global tokens, model seed and batches. No mocked CUDA operators.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from xllm.config import ModelConf, OptimConf
from xllm.distributed import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_hybrid_shard_data_parallel_group,
    init_torch_distributed,
    initialize_model_parallel,
)
from xllm.models import build_model, init_cache
from xllm.monitor import Timings
from xllm.optim import build_optimizer
from xllm.optim.dion_muon import validate_optimizer_contract
from xllm.utils import (
    get_model_state,
    get_sharded_optimizer_state,
    set_model_state,
    set_sharded_optimizer_state,
)
from train import State, get_optim_state, set_optim_state


def full_state(model, gradients=False):
    return {
        name: (p.grad if gradients else p.detach()).full_tensor().cpu()
        for name, p in model.named_parameters()
    }


def local_optimizer_state(model, optimizer):
    result = {}
    for name, parameter in model.named_parameters():
        for key, value in optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor):
                value = value.to_local() if hasattr(value, 'to_local') else value
                value = value.detach().cpu().clone()
            result[(name, key)] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['transformer', 'gekko', 'hybrid'], required=True)
    parser.add_argument('--cp', type=int, default=1)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--no-recompute', action='store_true')
    parser.add_argument('--save-awk', action='store_true')
    parser.add_argument('--full-attn-backend', choices=['xattn', 'flash'], default='xattn')
    parser.add_argument('--sca-backend', choices=['xattn', 'swift'], default='xattn')
    parser.add_argument('--optimizer', choices=['dion_muon', 'adamw'], default='dion_muon')
    parser.add_argument(
        '--orthogonalizer',
        choices=['newton_schulz5', 'polar_express'],
        default='newton_schulz5',
    )
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--save-reference', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    init_torch_distributed(timeout=300)
    initialize_model_parallel(1, args.cp, timeout=300)
    rank = dist.get_rank()
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    recompute = args.fused and not args.no_recompute
    cfg = ModelConf(
        arch='transformer' if args.arch == 'transformer' else 'gekko',
        full_attention_interval=4 if args.arch == 'hybrid' else 0,
        ddp_backend='fsdp2',
        num_layers=4,
        model_dim=512,
        ffn_hidden_dim=1024,
        vocab_size=256,
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        v_head_dim=128,
        rope_head_dim=64,
        chunk_size=128,
        swiglu=True,
        causal_attn_backend=args.full_attn_backend,
        sca_backend=args.sca_backend,
        causal_conv_backend='triton',
        timenorm_num_groups=8,
        layernorm_num_groups=1,
        memory_efficient_norm=False,
        apply_rmsnorm=True,
        fused_block=args.fused,
        recompute_q=recompute,
        recompute_v=recompute,
        recompute_attention=recompute,
        recompute_awk=recompute and args.arch != 'transformer' and not args.save_awk,
        recompute_fc1_out=recompute,
        recompute_fc3_out=recompute,
    )
    tokenizer = SimpleNamespace(vocab_size=256, bos_id=1, eos_id=2, pad_id=0)
    optim_cfg = OptimConf(
        optimizer=args.optimizer,
        muon_lr=0.01,
        lr=0.001,
        warmup=2,
        weight_decay=0.01,
        muon_orthogonalizer=args.orthogonalizer,
    )

    def build():
        model = build_model(cfg, 'bf16', None, True, True, False, tokenizer)
        optimizer, scheduler = build_optimizer(model, optim_cfg, 20, 'bf16')
        return model, optimizer, scheduler

    model, optimizer, scheduler = build()
    assert model._xllm_dion_mesh.ndim == 1
    assert model._xllm_dion_mesh.size() == get_data_parallel_world_size()
    if args.arch == 'hybrid':
        assert model.full_attention_layers == (0,)
        assert model.layers[0].__class__.__name__.startswith('FSDPTransformerBlock')
    dp_rank = get_data_parallel_rank()
    generator = torch.Generator().manual_seed(123 + dp_rank)
    batches = []
    for _ in range(args.steps + 1):
        tokens = torch.randint(3, 256, (1, args.seq_len), generator=generator)
        # Boundaries inside chunks and directly next to CP splits.
        for position in (73, 129, args.seq_len // 2 - 2, args.seq_len // 2 + 5):
            tokens[0, position] = 2
        targets = torch.roll(tokens, -1, 1)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[:, -17:] = False
        batches.append((tokens.cuda(), targets.cuda(), mask.cuda()))

    def update(model, optimizer, scheduler, batch, capture_gradients=False):
        tokens, targets, mask = batch
        losses, _, _ = model(
            tokens,
            multi_segments=True,
            targets=targets,
            token_mask=mask,
            deterministic=True,
        )
        count = mask.sum()
        dist.all_reduce(count, group=get_data_parallel_group())
        divisor = count / (get_data_parallel_world_size() * args.cp)
        train_loss = losses.sum() / (divisor + 1e-6)
        train_loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f'Missing gradient: {name}'
            assert torch.isfinite(p.grad.to_local()).all(), f'Nonfinite gradient: {name}'
        gradients = full_state(model, gradients=True) if capture_gradients else None
        grad_norm = model.clip_grad_norm_(1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss.detach() / args.cp
        dist.all_reduce(loss)
        return float(loss / dist.get_world_size()), grad_norm, gradients

    initial = full_state(model)
    metrics = []
    torch.cuda.synchronize()
    start = time.monotonic()
    for step in range(args.steps):
        loss, grad_norm, gradients = update(
            model, optimizer, scheduler, batches[step], capture_gradients=step == 0
        )
        if step == 0:
            first_gradients = gradients
        metrics.append({'step': step + 1, 'loss': loss, 'grad_norm': grad_norm})
        if rank == 0:
            print(json.dumps(metrics[-1]), flush=True)
    trained = full_state(model)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    result = {
        'arch': args.arch,
        'cp': args.cp,
        'dp': get_data_parallel_world_size(),
        'fused': args.fused,
        'recompute': recompute,
        'recompute_awk': cfg.recompute_awk,
        'full_attn_backend': args.full_attn_backend,
        'sca_backend': args.sca_backend if args.arch != 'transformer' else None,
        'optimizer': args.optimizer,
        'orthogonalizer': args.orthogonalizer if args.optimizer == 'dion_muon' else None,
        'steps': args.steps,
        'seq_len': args.seq_len,
        'metrics': metrics,
        'torch': torch.__version__,
        'wps': args.steps * args.seq_len * get_data_parallel_world_size() / elapsed,
        'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
    }
    reference = {
        'initial': initial,
        'trained': trained,
        'gradients': first_gradients,
        'metrics': metrics,
    }
    if args.reference:
        saved = torch.load(args.reference, weights_only=True)
        gradient_errors = {}
        for name, actual in first_gradients.items():
            expected = saved['gradients'][name]
            delta = actual.float() - expected.float()
            gradient_errors[name] = {
                'relative_l2': float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
                'l2': float(delta.norm()),
                'reference_l2': float(expected.float().norm()),
                'numel': actual.numel(),
                'max_abs': float(delta.abs().max()),
                'reference_max_abs': float(expected.abs().max()),
            }
        max_delta = 0.0
        for name in trained:
            torch.testing.assert_close(initial[name], saved['initial'][name], rtol=0, atol=0)
            expected_weights = saved['trained'][name].float()
            weight_delta = trained[name].float() - expected_weights
            max_delta = max(max_delta, weight_delta.abs().max().item())
            # Small gradient differences can flip AdamW updates at a few
            # near-zero entries. Bound tensor error and scale-aware outliers.
            weight_rms_error = float(weight_delta.norm() / weight_delta.numel() ** 0.5)
            reference_weight_rms = float(expected_weights.norm() / weight_delta.numel() ** 0.5)
            assert weight_rms_error <= 0.03 * reference_weight_rms + 0.1 * optim_cfg.lr, (
                name,
                weight_rms_error,
                reference_weight_rms,
            )
            assert weight_delta.abs().max() <= 0.03 * expected_weights.abs().max() + optim_cfg.lr, (
                name
            )
            # CP changes BF16 GEMM/reduction partitioning. Near-zero individual
            # entries can cancel differently, so bound both whole-tensor error
            # and the largest error relative to the tensor's gradient scale.
            error = gradient_errors[name]
            assert error['l2'] <= 0.01 * error['reference_l2'] + 1e-6 * error['numel'] ** 0.5, (
                name,
                error,
            )
            assert error['max_abs'] <= 0.03 * error['reference_max_abs'] + 1e-5, (name, error)
        for actual, expected in zip(metrics, saved['metrics']):
            torch.testing.assert_close(
                torch.tensor(actual['loss']),
                torch.tensor(expected['loss']),
                rtol=0.003,
                atol=0.001,
            )
            torch.testing.assert_close(
                torch.tensor(actual['grad_norm']),
                torch.tensor(expected['grad_norm']),
                rtol=0.03,
                atol=0.001,
            )
        result['cp_parity_max_parameter_delta'] = max_delta
        result['gradient_parity_max_relative_l2'] = max(
            error['relative_l2'] for error in gradient_errors.values()
        )
        result['gradient_parity_max_absolute_rms'] = max(
            error['l2'] / error['numel'] ** 0.5 for error in gradient_errors.values()
        )
    if args.save_reference and rank == 0:
        args.save_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference, args.save_reference)

    # Exercise the release's DCP helpers, then compare an uninterrupted update
    # against the identical update after rebuilding FSDP2, optimizer and scheduler.
    checkpoint = args.output.with_suffix('')
    group = get_hybrid_shard_data_parallel_group()
    model_state = get_model_state(model, full_state=False)
    data_iterator = SimpleNamespace(
        get_state=lambda: {'batch': args.steps}, set_state=lambda state: None
    )
    training_status = State(args.steps, 1.0, 0, 0)
    optimizer_state, training_state = get_optim_state(
        model, optimizer, True, scheduler, training_status, data_iterator, Timings()
    )
    saved_optimizer_tensors = local_optimizer_state(model, optimizer)
    dcp.save(
        {'model': model_state, 'optimizer': optimizer_state},
        checkpoint_id=checkpoint,
        process_group=group,
    )
    training_state_path = checkpoint / f'training_state_{rank}.pt'
    torch.save(training_state, training_state_path)
    expected_loss, expected_norm, expected_gradients = update(
        model, optimizer, scheduler, batches[-1], capture_gradients=True
    )
    expected_weights = full_state(model)
    resumed, resumed_optimizer, resumed_scheduler = build()
    training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)
    validate_optimizer_contract(resumed_optimizer, training_state)
    resumed_model_state = get_model_state(resumed, full_state=False)
    resumed_optim_state = get_sharded_optimizer_state(resumed, resumed_optimizer, True)
    dcp.load(
        {'model': resumed_model_state, 'optimizer': resumed_optim_state},
        checkpoint_id=checkpoint,
        process_group=group,
    )
    set_model_state(resumed, resumed_model_state, full_state=False)
    set_sharded_optimizer_state(resumed, resumed_optimizer, resumed_optim_state, True)
    set_optim_state(
        resumed_scheduler,
        training_state,
        State(0, 1.0, 0, 0),
        data_iterator,
        Timings(),
        optim_cfg,
    )
    loaded_weights = full_state(resumed)
    for name in trained:
        torch.testing.assert_close(
            loaded_weights[name],
            trained[name],
            rtol=0,
            atol=0,
            msg=lambda message: f'Loaded model {name}: {message}',
        )
    loaded_optimizer_tensors = local_optimizer_state(resumed, resumed_optimizer)
    assert loaded_optimizer_tensors.keys() == saved_optimizer_tensors.keys()
    for key, expected in saved_optimizer_tensors.items():
        actual = loaded_optimizer_tensors[key]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0,
                atol=0,
                msg=lambda message: f'Loaded optimizer {key}: {message}',
            )
        else:
            assert actual == expected, key
    actual_loss, actual_norm, actual_gradients = update(
        resumed, resumed_optimizer, resumed_scheduler, batches[-1], capture_gradients=True
    )
    assert actual_norm == expected_norm
    for name in actual_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=0, atol=0)
    actual_weights = full_state(resumed)
    for name in expected_weights:
        torch.testing.assert_close(
            actual_weights[name],
            expected_weights[name],
            rtol=0,
            atol=0,
            msg=lambda message: f'Resumed update {name}: {message}',
        )
    assert actual_loss == expected_loss
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    result['checkpoint_resume'] = 'exact'

    # Hybrid cache dispatch must also work for ordinary unpacked generation.
    if args.arch == 'hybrid' and args.cp == 1:
        resumed.eval()
        with torch.no_grad():
            cache = init_cache(resumed)
            _, _, cache = resumed(batches[0][0][:, :128], multi_segments=False, cache=cache)
            logits, _, cache = resumed(batches[0][0][:, 128:129], multi_segments=False, cache=cache)
            assert logits.shape == (1, 1, 256)
            assert cache[0][0][-1] == 129
        result['hybrid_cache'] = 'passed'
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print('QUALIFICATION_PASSED', json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
