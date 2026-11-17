import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from xllm.config import ModelConf
from xllm.distributed import destroy_model_parallel, initialize_model_parallel
from xllm.models.fused_blocks.moe import moe_bwd, moe_fwd
from xllm.models.transformer import Transformer, TransformerMoEBlock
from xllm.modules.moe.moe import NormalizedMoE
from xllm.modules.rotary_positional_embedding import RotaryEmbedding


def _reference_latent_moe(
    x,
    router_w,
    w_down,
    w_up,
    expert_w1,
    expert_w2,
    expert_w3,
    topk,
    routing_type,
    qb_bias,
    latent_norm_w=None,
    post_rmsnorm=False,
    norm_eps=1e-6,
):
    logits = F.linear(x, router_w)
    if routing_type == 'topk':
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        selected_experts = torch.topk(scores, topk, dim=-1)[1]
        routing_scores = torch.gather(scores, dim=-1, index=selected_experts)
        if topk > 1:
            routing_scores = routing_scores / routing_scores.sum(dim=-1, keepdim=True)
        token_indices = torch.arange(x.shape[0], device=x.device).repeat_interleave(topk)
        expert_indices = selected_experts.flatten()
        routing_scores = routing_scores.flatten()
    else:
        assert routing_type == 'dynamic_qb'
        scores = torch.sigmoid(logits.to(torch.float32))
        active_mask = scores > qb_bias.view(1, -1)
        token_indices, expert_indices = torch.nonzero(active_mask, as_tuple=True)
        routing_scores = scores[token_indices, expert_indices]
        if topk > 1:
            routes_per_token = torch.bincount(token_indices, minlength=x.shape[0])
            normalize_mask = routes_per_token[token_indices] > 1
            score_mass = torch.zeros(x.shape[0], device=x.device, dtype=routing_scores.dtype)
            score_mass.index_add_(0, token_indices, routing_scores)
            normalized_scores = routing_scores / score_mass[token_indices]
            routing_scores = torch.where(normalize_mask, normalized_scores, routing_scores)
    routing_scores = routing_scores.to(x)

    latent_x = F.linear(x, w_down)
    latent_out = torch.zeros_like(latent_x)
    for expert_id in range(expert_w1.shape[0]):
        selected = torch.nonzero(expert_indices == expert_id, as_tuple=True)[0]
        if selected.numel() == 0:
            continue
        expert_tokens = token_indices[selected]
        expert_x = latent_x[expert_tokens]
        hidden = F.silu(F.linear(expert_x, expert_w1[expert_id]))
        hidden = hidden * F.linear(expert_x, expert_w3[expert_id])
        expert_out = F.linear(hidden, expert_w2[expert_id])
        expert_out = expert_out * routing_scores[selected].unsqueeze(1)
        latent_out.index_add_(0, expert_tokens, expert_out)

    if post_rmsnorm:
        latent_out = latent_out * torch.rsqrt(latent_out.square().mean(dim=-1, keepdim=True) + norm_eps)
        if latent_norm_w is not None:
            latent_out = latent_out * (latent_norm_w + 1.0)
    return F.linear(latent_out, w_up)


def _torch_rmsnorm_fwd(x, weight, num_groups, eps):
    """CPU reference for the CUDA RMSNorm kernel used by the fused latent norm."""
    assert num_groups == 1
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    y = x.float() * rstd
    if weight is not None:
        y = y * (weight.float() + 1.0)
    return y.to(x), rstd


def _torch_rmsnorm_bwd(y_grad, x, rstd, weight, num_groups):
    """Analytic backward matching ``_torch_rmsnorm_fwd``."""
    assert num_groups == 1
    x_hat = x.float() * rstd
    grad = y_grad.float()
    weight_grad = None
    if weight is not None:
        weight_grad = (grad * x_hat).flatten(end_dim=-2).sum(dim=0).to(weight)
        grad = grad * (weight.float() + 1.0)
    x_grad = rstd * (grad - x_hat * (grad * x_hat).mean(dim=-1, keepdim=True))
    return x_grad.to(x), weight_grad


def test_latent_moe_config_validation():
    cfg = ModelConf(
        arch='transformer',
        model_dim=16,
        num_experts=8,
        num_activated_experts=2,
        expert_inter_dim=12,
        moe_latent_alpha=4,
    )
    assert cfg.moe_latent_alpha == 4
    assert cfg.effective_num_experts == 32
    assert cfg.effective_num_activated_experts == 8
    assert not cfg.apply_rmsnorm

    with pytest.raises(AssertionError):
        ModelConf(
            arch='transformer',
            model_dim=16,
            num_experts=8,
            num_activated_experts=2,
            expert_inter_dim=12,
            moe_latent_alpha=1,
        )

    with pytest.raises(AssertionError):
        ModelConf(
            arch='transformer',
            model_dim=16,
            num_experts=8,
            num_activated_experts=2,
            expert_inter_dim=12,
            moe_latent_alpha=3,
        )

    with pytest.raises(AssertionError, match='not supported by MoVA'):
        ModelConf(
            arch='transformer',
            model_dim=16,
            num_values=4,
            num_activated_values=1,
            num_experts=8,
            num_activated_experts=2,
            expert_inter_dim=12,
            moe_latent_alpha=4,
        )

    with pytest.raises(AssertionError, match='requires moe_latent_alpha'):
        ModelConf(
            arch='transformer',
            model_dim=16,
            num_experts=8,
            num_activated_experts=2,
            expert_inter_dim=12,
            moe_latent_post_rmsnorm=True,
        )


def test_latent_moe_weight_shapes_keep_router_and_shared_experts_full_dimensional():
    moe = NormalizedMoE(
        layer_id=0,
        model_dim=16,
        latent_dim=4,
        expert_inter_dim=12,
        num_experts=8,
        num_activated_experts=2,
        num_shared_experts=1,
        expert_backend='sequential',
        router_score_func='softmax',
    )

    assert moe.router.weight.shape == (8, 16)
    assert moe.w_down.weight.shape == (4, 16)
    assert moe.w_up.weight.shape == (16, 4)
    assert moe.experts.weight1.shape == (8 * 12, 4)
    assert moe.experts.weight2.shape == (8 * 4, 12)
    assert moe.fc1.weight.shape == (12, 16)
    assert moe.fc2.weight.shape == (16, 12)
    assert moe.latent_norm is None


def test_latent_post_rmsnorm_is_explicit():
    moe = NormalizedMoE(
        layer_id=0,
        model_dim=16,
        latent_dim=4,
        latent_post_rmsnorm=True,
        expert_inter_dim=12,
        num_experts=8,
        num_activated_experts=2,
        num_shared_experts=1,
        expert_backend='sequential',
        router_score_func='softmax',
    )
    assert moe.latent_norm.weight.shape == (4,)
    assert torch.count_nonzero(moe.latent_norm.weight) == 0

    with pytest.raises(AssertionError, match='requires a latent dimension'):
        NormalizedMoE(
            layer_id=0,
            model_dim=16,
            latent_post_rmsnorm=True,
            expert_inter_dim=12,
            num_experts=8,
            num_activated_experts=2,
            num_shared_experts=1,
        )


def test_transformer_block_uses_latent_effective_counts():
    cfg = ModelConf(
        arch='transformer',
        model_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_experts=8,
        num_activated_experts=2,
        num_shared_experts=1,
        expert_inter_dim=12,
        moe_latent_alpha=4,
        moe_router_score_func='softmax',
    )

    block = TransformerMoEBlock(cfg, layer_id=0)

    assert block.moe.n_experts == 32
    assert block.moe.topk == 8
    assert block.moe.router.weight.shape == (32, 16)
    assert block.moe.experts.weight1.shape == (32 * 12, 4)


@pytest.mark.parametrize('post_rmsnorm', [False, True])
def test_latent_parameter_accounting_uses_effective_counts(post_rmsnorm):
    cfg = ModelConf(
        arch='transformer',
        num_layers=1,
        model_dim=16,
        num_heads=2,
        num_kv_heads=2,
        head_dim=8,
        vocab_size=32,
        ffn_hidden_dim=24,
        num_experts=8,
        num_activated_experts=2,
        num_shared_experts=1,
        expert_inter_dim=12,
        moe_latent_alpha=4,
        moe_latent_post_rmsnorm=post_rmsnorm,
        moe_router_score_func='softmax',
        apply_rmsnorm=True,
        swiglu=True,
    )
    tokenizer = type('Tokenizer', (), {'bos_id': 0, 'eos_id': 1, 'pad_id': 2})()

    model = Transformer(cfg, tokenizer)
    total_params, _, _ = model.num_parameters()

    assert model.num_experts == 8
    assert model.num_activated_experts == 2
    assert model.effective_num_experts == 32
    assert model.effective_num_activated_experts == 8
    assert total_params == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize('post_rmsnorm', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='MoE kernels require CUDA')
def test_unfused_latent_moe_matches_reference(post_rmsnorm):
    torch.manual_seed(2)
    # The CUDA RMSNorm kernel needs at least 128 features.
    moe = NormalizedMoE(
        layer_id=0,
        model_dim=256,
        latent_dim=128,
        latent_post_rmsnorm=post_rmsnorm,
        expert_inter_dim=32,
        num_experts=4,
        num_activated_experts=2,
        num_shared_experts=0,
        expert_backend='sequential',
        router_score_func='softmax',
        apply_rmsnorm=True,
        norm_affine=post_rmsnorm,
    ).cuda()
    if post_rmsnorm:
        with torch.no_grad():
            moe.latent_norm.weight.normal_()
    x = torch.randn(2, 3, 256, device='cuda')

    out, aux_loss = moe(x)
    with torch.no_grad():
        normalized_x = moe.norm(x).view(-1, 256)
        reference_out = _reference_latent_moe(
            normalized_x,
            moe.router.weight,
            moe.w_down.weight,
            moe.w_up.weight,
            moe.experts.weight1.view(4, 32, 128),
            moe.experts.weight2.view(4, 128, 32),
            moe.experts.weight3.view(4, 32, 128),
            moe.topk,
            'topk',
            None,
            latent_norm_w=moe.latent_norm.weight if post_rmsnorm else None,
            post_rmsnorm=post_rmsnorm,
        )

    assert aux_loss is None
    # the block adds the residual
    torch.testing.assert_close((out - x).view_as(reference_out), reference_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('post_rmsnorm', [False, True])
@pytest.mark.parametrize('routing_type', ['topk', 'dynamic_qb'])
def test_fused_latent_moe_matches_reference(routing_type, post_rmsnorm, monkeypatch):
    from xllm.models.fused_blocks import moe as fused_moe_module

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        monkeypatch.setattr(fused_moe_module, 'grouped_experts_fwd', _torch_grouped_experts_fwd)
        monkeypatch.setattr(fused_moe_module, 'grouped_experts_bwd', _torch_grouped_experts_bwd)
    # The CUDA RMSNorm kernel needs at least 128 features; the block tests cover it at that width.
    monkeypatch.setattr(fused_moe_module, 'rmsnorm_fwd', _torch_rmsnorm_fwd)
    monkeypatch.setattr(fused_moe_module, 'rmsnorm_bwd', _torch_rmsnorm_bwd)
    num_tokens = 5
    model_dim = 8
    latent_dim = 4
    inter_dim = 6
    num_experts = 4
    topk = 2

    # Sampled on the CPU so the routing preconditions below hold on every device.
    tensors = {
        'x': torch.randn(num_tokens, model_dim),
        'router_w': torch.randn(num_experts, model_dim),
        'w_down': torch.randn(latent_dim, model_dim),
        'w_up': torch.randn(model_dim, latent_dim),
        'expert_w1': torch.randn(num_experts, inter_dim, latent_dim),
        'expert_w2': torch.randn(num_experts, latent_dim, inter_dim),
        'expert_w3': torch.randn(num_experts, inter_dim, latent_dim),
    }
    if post_rmsnorm:
        tensors['latent_norm_w'] = torch.randn(latent_dim)
    tensors = {name: value.to(device) for name, value in tensors.items()}
    qb_bias = None
    score_func = 'softmax'
    if routing_type == 'dynamic_qb':
        score_func = 'sigmoid'
        qb_bias = torch.tensor([0.25, 0.45, 0.55, 0.75], device=device)
        with torch.no_grad():
            scores = torch.sigmoid(F.linear(tensors['x'], tensors['router_w']))
            active_mask = scores > qb_bias
            routes_per_token = active_mask.sum(dim=1)
            assert torch.any(routes_per_token == 0)
            assert torch.any(routes_per_token == 1)
            assert torch.any(routes_per_token > 1)
            assert routes_per_token.unique().numel() > 1
            token_indices = torch.nonzero(active_mask, as_tuple=True)[0]
            selected_scores = scores[active_mask]
            score_mass = torch.zeros(num_tokens, device=device)
            score_mass.index_add_(0, token_indices, selected_scores)
            assert torch.any(score_mass > 1.0)

    reference_tensors = {
        name: value.detach().clone().requires_grad_(True) for name, value in tensors.items()
    }
    reference_out = _reference_latent_moe(
        reference_tensors['x'],
        reference_tensors['router_w'],
        reference_tensors['w_down'],
        reference_tensors['w_up'],
        reference_tensors['expert_w1'],
        reference_tensors['expert_w2'],
        reference_tensors['expert_w3'],
        topk,
        routing_type,
        qb_bias,
        latent_norm_w=reference_tensors.get('latent_norm_w'),
        post_rmsnorm=post_rmsnorm,
    )

    # n x s x d experts, stored as (n s) x d
    expert_ws = [tensors[name].flatten(0, 1) for name in ['expert_w1', 'expert_w2', 'expert_w3']]
    with torch.no_grad():
        fused_out, _, routing_out, _, routing_state, rng_states = moe_fwd(
            tensors['x'].view(1, num_tokens, model_dim),
            tensors['router_w'], qb_bias, score_func, None, None,
            None, None, None, tensors['w_down'], tensors['w_up'], tensors.get('latent_norm_w'), post_rmsnorm, 1e-6,
            num_experts, num_experts, 0, num_experts, topk,
            routing_type, False,
            *expert_ws, 'sequential', 'torch', 0.0, False, False,
        )

    torch.testing.assert_close(fused_out.view_as(reference_out), reference_out, rtol=1e-4, atol=1e-4)

    out_grad = torch.randn_like(reference_out)
    reference_out.backward(out_grad)
    original_scores, routing_indices, org_topk_scores, token_indices, tokens_per_expert = routing_out
    group_sizes, input_splits, output_splits, curr_bsz = routing_state
    expert_hidden_rng_state, shared_hidden_rng_state = rng_states
    grads = moe_bwd(
        out_grad.view(1, num_tokens, model_dim), None,
        tensors['x'].view(1, num_tokens, model_dim), tensors['x'],
        original_scores, routing_indices, org_topk_scores, token_indices, tokens_per_expert,
        tensors['router_w'], qb_bias, None, score_func, None, None,
        None, None, None, None, None, tensors['w_down'], tensors['w_up'],
        tensors.get('latent_norm_w'), post_rmsnorm, 1e-6, num_experts, num_experts, topk,
        routing_type, 'average', 0.9, False,
        *expert_ws, 'sequential', 'torch', 0.0,
        group_sizes, input_splits, output_splits, curr_bsz,
        expert_hidden_rng_state, shared_hidden_rng_state, False,
    )
    (
        x_grad, expert_w1_grad, expert_w2_grad, expert_w3_grad, _, _, _,
        router_w_grad, w_down_grad, w_up_grad, latent_norm_w_grad,
    ) = grads

    actual_grads = {
        'x': x_grad.view_as(tensors['x']),
        'router_w': router_w_grad,
        'w_down': w_down_grad,
        'w_up': w_up_grad,
        'expert_w1': expert_w1_grad.view_as(tensors['expert_w1']),
        'expert_w2': expert_w2_grad.view_as(tensors['expert_w2']),
        'expert_w3': expert_w3_grad.view_as(tensors['expert_w3']),
    }
    if post_rmsnorm:
        actual_grads['latent_norm_w'] = latent_norm_w_grad
    else:
        assert latent_norm_w_grad is None
    for name, actual_grad in actual_grads.items():
        torch.testing.assert_close(actual_grad, reference_tensors[name].grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize('post_rmsnorm', [False, True])
@pytest.mark.parametrize('apply_rmsnorm', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused Transformer block requires CUDA')
def test_fused_transformer_latent_moe_autograd_wiring(apply_rmsnorm, post_rmsnorm):
    # Latent width 128, the smallest the CUDA RMSNorm kernel supports.
    cfg = ModelConf(
        arch='transformer',
        num_layers=1,
        model_dim=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        num_experts=4,
        num_activated_experts=2,
        num_shared_experts=1,
        expert_inter_dim=64,
        moe_latent_alpha=2,
        moe_latent_post_rmsnorm=post_rmsnorm,
        moe_router_score_func='softmax',
        causal_attn_backend='flash',
        fused_block=True,
        apply_rmsnorm=apply_rmsnorm,
    )
    block = TransformerMoEBlock(cfg, layer_id=0).cuda().to(torch.bfloat16).train()
    x = torch.randn(1, 4, cfg.model_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    freqs_cis = RotaryEmbedding(cfg.head_dim, 4, cfg.rope_base).get_freqs_cis(0, 4, x.device)

    out, aux_loss, cache = block(x, freqs_cis)
    assert out.shape == x.shape
    assert aux_loss is None
    assert cache is None

    out.float().square().mean().backward()
    parameters = [
        x,
        block.moe.w_down.weight,
        block.moe.w_up.weight,
        block.moe.router.weight,
        block.moe.experts.weight1,
        block.moe.experts.weight2,
        block.moe.experts.weight3,
    ]
    if post_rmsnorm:
        parameters.append(block.moe.latent_norm.weight)
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize('post_rmsnorm', [False, True])
@pytest.mark.parametrize('router_type', ['topk', 'dynamic_qb'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused Transformer block requires CUDA')
def test_fused_and_unfused_latent_moe_blocks_match(router_type, post_rmsnorm):
    # Training runs the fused block and evaluation the unfused modules, so they must compute the same function.
    torch.manual_seed(3)
    cfg = ModelConf(
        arch='transformer',
        num_layers=1,
        model_dim=256,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        num_experts=4,
        num_activated_experts=2,
        num_shared_experts=1,
        expert_inter_dim=64,
        moe_latent_alpha=2,
        moe_latent_post_rmsnorm=post_rmsnorm,
        moe_router_type=router_type,
        moe_router_score_func='sigmoid' if router_type == 'dynamic_qb' else 'softmax',
        causal_attn_backend='flash',
        fused_block=True,
        apply_rmsnorm=True,
    )
    block = TransformerMoEBlock(cfg, layer_id=0).cuda().to(torch.bfloat16)
    if post_rmsnorm:
        with torch.no_grad():
            block.moe.latent_norm.weight.normal_()
    seq_len = 16
    x = torch.randn(1, seq_len, cfg.model_dim, device='cuda', dtype=torch.bfloat16)
    freqs_cis = RotaryEmbedding(cfg.head_dim, seq_len, cfg.rope_base).get_freqs_cis(0, seq_len, x.device)
    # Backward updates the dynamic-QB thresholds, so both passes start from the same ones.
    initial_bias = block.moe.router.bias.detach().clone() if block.moe.router.bias is not None else None

    def forward_backward(training):
        block.train(training)
        block.zero_grad(set_to_none=True)
        if initial_bias is not None:
            block.moe.router.bias.copy_(initial_bias)
        inp = x.clone().requires_grad_(True)
        out = block(inp, freqs_cis)[0]
        out.float().square().mean().backward()
        grads = {name: p.grad.float() for name, p in block.named_parameters() if p.grad is not None}
        grads['x'] = inp.grad.float()
        return out.detach(), grads

    fused_out, fused_grads = forward_backward(True)
    unfused_out, unfused_grads = forward_backward(False)

    torch.testing.assert_close(fused_out, unfused_out, rtol=2e-2, atol=2e-2)
    assert fused_grads.keys() == unfused_grads.keys()
    if post_rmsnorm:
        assert 'moe.latent_norm.weight' in fused_grads
    for name, unfused_grad in unfused_grads.items():
        error = (fused_grads[name] - unfused_grad).norm() / unfused_grad.norm().clamp_min(1e-6)
        assert error < 3e-2, f'{name}: relative gradient error {error:.3e}'


def _torch_grouped_experts_fwd(
    x,
    w1,
    w2,
    w3,
    group_sizes,
    backend,
    memory_efficient,
    dropout,
    hidden_rng_state=None,
):
    """CPU reference for the CUDA grouped-expert kernels used by the Gloo test."""
    del backend, memory_efficient, hidden_rng_state
    assert dropout == 0.0
    h1 = torch.empty(x.shape[0], w1.shape[1], dtype=x.dtype, device=x.device)
    h3 = torch.empty_like(h1)
    hidden = torch.empty_like(h1)
    out = torch.empty(x.shape[0], w2.shape[1], dtype=x.dtype, device=x.device)
    offset = 0
    for expert_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        expert_x = x[offset:next_offset]
        h1[offset:next_offset] = F.linear(expert_x, w1[expert_idx])
        h3[offset:next_offset] = F.linear(expert_x, w3[expert_idx])
        hidden[offset:next_offset] = F.silu(h1[offset:next_offset]) * h3[offset:next_offset]
        out[offset:next_offset] = F.linear(hidden[offset:next_offset], w2[expert_idx])
        offset = next_offset
    assert offset == x.shape[0]
    return out, h1, F.silu(h1), h3, hidden, None


def _torch_grouped_experts_bwd(
    y_grad,
    x,
    w1,
    w2,
    w3,
    h1,
    h1s,
    h3,
    hidden,
    group_sizes,
    backend,
    memory_efficient,
    dropout,
    hidden_rng_state,
    hidden_noise,
):
    """Analytic backward matching ``_torch_grouped_experts_fwd``."""
    del backend, memory_efficient, hidden_rng_state, hidden_noise
    assert dropout == 0.0
    x_grad = torch.empty_like(x)
    w1_grad = torch.zeros_like(w1)
    w2_grad = torch.zeros_like(w2)
    w3_grad = torch.zeros_like(w3)
    offset = 0
    for expert_idx, group_size in enumerate(group_sizes):
        next_offset = offset + group_size
        expert_x = x[offset:next_offset]
        expert_y_grad = y_grad[offset:next_offset]
        expert_hidden = hidden[offset:next_offset]
        hidden_grad = expert_y_grad.matmul(w2[expert_idx])
        h1_grad = torch.ops.aten.silu_backward(
            hidden_grad * h3[offset:next_offset],
            h1[offset:next_offset],
        )
        h3_grad = hidden_grad * h1s[offset:next_offset]
        x_grad[offset:next_offset] = (
            h1_grad.matmul(w1[expert_idx]) + h3_grad.matmul(w3[expert_idx])
        )
        w1_grad[expert_idx] = h1_grad.t().matmul(expert_x)
        w2_grad[expert_idx] = expert_y_grad.t().matmul(expert_hidden)
        w3_grad[expert_idx] = h3_grad.t().matmul(expert_x)
        offset = next_offset
    assert offset == x.shape[0]
    return x_grad, w1_grad, w2_grad, w3_grad


def _gloo_reduce_scatter(x, parallel_region, dim=None, async_op=False):
    """Gloo-compatible equivalent of the production reduce-scatter wrapper."""
    del async_op
    assert parallel_region == "model"
    scatter_dim = x.dim() - 1 if dim is None else dim
    dist.all_reduce(x)
    rank = dist.get_rank()
    return torch.chunk(x, dist.get_world_size(), dim=scatter_dim)[rank].contiguous(), None


def _gloo_all_gather(x, parallel_region, dim=None, async_op=False):
    """Gloo-compatible equivalent of the production all-gather wrapper."""
    del async_op
    assert parallel_region == "model"
    gather_dim = x.dim() - 1 if dim is None else dim
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=gather_dim), None


def _assert_tp2_fused_latent_moe_parity(rank, device, routing_type, post_rmsnorm):
    world_size = 2
    num_tokens = 5
    model_dim = 8
    latent_dim = 4
    inter_dim = 6
    num_experts = 4
    topk = 2
    local_model_dim = model_dim // world_size
    local_latent_dim = latent_dim // world_size
    local_experts = num_experts // world_size

    generator = torch.Generator(device="cpu").manual_seed(29)
    full_tensors = {
        "x": torch.randn(num_tokens, model_dim, generator=generator),
        "router_w": torch.randn(num_experts, model_dim, generator=generator),
        "w_down": torch.randn(latent_dim, model_dim, generator=generator),
        "w_up": torch.randn(model_dim, latent_dim, generator=generator),
        "expert_w1": torch.randn(num_experts, inter_dim, latent_dim, generator=generator),
        "expert_w2": torch.randn(num_experts, latent_dim, inter_dim, generator=generator),
        "expert_w3": torch.randn(num_experts, inter_dim, latent_dim, generator=generator),
    }
    if post_rmsnorm:
        full_tensors["latent_norm_w"] = torch.randn(latent_dim, generator=generator)
    if routing_type == "dynamic_qb":
        score_func = "sigmoid"
        # Select each expert's top two tokens. Thresholds sit midway between scores so the
        # TP-sharded logits cannot flip a tie against the reference.
        sorted_scores = torch.sigmoid(
            F.linear(full_tensors["x"], full_tensors["router_w"])
        ).sort(dim=0).values
        qb_bias = ((sorted_scores[2] + sorted_scores[3]) / 2).to(device)
    else:
        score_func = "softmax"
        qb_bias = None
    reference_tensors = {
        name: tensor.to(device).detach().clone().requires_grad_(True)
        for name, tensor in full_tensors.items()
    }
    reference_out = _reference_latent_moe(
        reference_tensors["x"],
        reference_tensors["router_w"],
        reference_tensors["w_down"],
        reference_tensors["w_up"],
        reference_tensors["expert_w1"],
        reference_tensors["expert_w2"],
        reference_tensors["expert_w3"],
        topk,
        routing_type,
        qb_bias,
        latent_norm_w=reference_tensors.get("latent_norm_w"),
        post_rmsnorm=post_rmsnorm,
    )
    output_grad = torch.randn(num_tokens, model_dim, generator=generator).to(device)
    reference_out.backward(output_grad)

    model_slice = slice(rank * local_model_dim, (rank + 1) * local_model_dim)
    latent_slice = slice(rank * local_latent_dim, (rank + 1) * local_latent_dim)
    expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
    local_tensors = {
        "x": full_tensors["x"][:, model_slice].contiguous().to(device),
        "router_w": full_tensors["router_w"][:, model_slice].contiguous().to(device),
        "w_down": full_tensors["w_down"][:, model_slice].contiguous().to(device),
        "w_up": full_tensors["w_up"][:, latent_slice].contiguous().to(device),
        # local experts, stored as (n s) x d
        "expert_w1": full_tensors["expert_w1"][expert_slice].flatten(0, 1).to(device),
        "expert_w2": full_tensors["expert_w2"][expert_slice].flatten(0, 1).to(device),
        "expert_w3": full_tensors["expert_w3"][expert_slice].flatten(0, 1).to(device),
    }
    if post_rmsnorm:
        local_tensors["latent_norm_w"] = full_tensors["latent_norm_w"][latent_slice].contiguous().to(device)

    with torch.no_grad():
        fused_out, _, routing_out, _, routing_state, rng_states = moe_fwd(
            local_tensors["x"].view(1, num_tokens, local_model_dim),
            local_tensors["router_w"], qb_bias, score_func, None, None,
            None, None, None, local_tensors["w_down"], local_tensors["w_up"],
            local_tensors.get("latent_norm_w"), post_rmsnorm, 1e-6,
            num_experts, local_experts, expert_slice.start, expert_slice.stop, topk,
            routing_type, False,
            local_tensors["expert_w1"], local_tensors["expert_w2"], local_tensors["expert_w3"],
            "sequential", "torch", 0.0, False, False,
        )

    torch.testing.assert_close(
        fused_out.view(num_tokens, local_model_dim),
        reference_out.detach()[:, model_slice],
        rtol=2e-4,
        atol=2e-4,
    )

    original_scores, routing_indices, org_topk_scores, token_indices, tokens_per_expert = routing_out
    group_sizes, input_splits, output_splits, curr_bsz = routing_state
    expert_hidden_rng_state, shared_hidden_rng_state = rng_states
    grads = moe_bwd(
        # NCCL all-gathers need contiguous inputs, as autograd provides in training.
        output_grad[:, model_slice].contiguous().view(1, num_tokens, local_model_dim), None,
        local_tensors["x"].view(1, num_tokens, local_model_dim), local_tensors["x"],
        original_scores, routing_indices, org_topk_scores, token_indices, tokens_per_expert,
        local_tensors["router_w"], qb_bias, None, score_func, None, None,
        None, None, None, None, None, local_tensors["w_down"], local_tensors["w_up"],
        local_tensors.get("latent_norm_w"), post_rmsnorm, 1e-6,
        num_experts, local_experts, topk, routing_type, "average", 0.9, False,
        local_tensors["expert_w1"], local_tensors["expert_w2"], local_tensors["expert_w3"],
        "sequential", "torch", 0.0,
        group_sizes, input_splits, output_splits, curr_bsz,
        expert_hidden_rng_state, shared_hidden_rng_state, False,
    )
    (
        x_grad, expert_w1_grad, expert_w2_grad, expert_w3_grad, _, _, _,
        router_w_grad, w_down_grad, w_up_grad, latent_norm_w_grad,
    ) = grads

    expected_grads = {
        "x": reference_tensors["x"].grad[:, model_slice],
        "router_w": reference_tensors["router_w"].grad[:, model_slice],
        "w_down": reference_tensors["w_down"].grad[:, model_slice],
        "w_up": reference_tensors["w_up"].grad[:, latent_slice],
        "expert_w1": reference_tensors["expert_w1"].grad[expert_slice],
        "expert_w2": reference_tensors["expert_w2"].grad[expert_slice],
        "expert_w3": reference_tensors["expert_w3"].grad[expert_slice],
    }
    actual_grads = {
        "x": x_grad.view(num_tokens, local_model_dim),
        "router_w": router_w_grad,
        "w_down": w_down_grad,
        "w_up": w_up_grad,
        "expert_w1": expert_w1_grad.view_as(expected_grads["expert_w1"]),
        "expert_w2": expert_w2_grad.view_as(expected_grads["expert_w2"]),
        "expert_w3": expert_w3_grad.view_as(expected_grads["expert_w3"]),
    }
    if post_rmsnorm:
        expected_grads["latent_norm_w"] = reference_tensors["latent_norm_w"].grad[latent_slice]
        actual_grads["latent_norm_w"] = latent_norm_w_grad
    else:
        assert latent_norm_w_grad is None
    for name, actual_grad in actual_grads.items():
        torch.testing.assert_close(
            actual_grad,
            expected_grads[name],
            rtol=3e-4,
            atol=3e-4,
            msg=f"{name} gradient mismatch",
        )


def _tp2_fused_latent_moe_worker(rank, init_method, routing_type, post_rmsnorm):
    world_size = 2
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    initialize_model_parallel(
        model_parallel_size=world_size,
        context_parallel_size=1,
        timeout=120,
        model_parallel_backend="nccl",
        context_parallel_backend="nccl",
        ddp_backend="nccl",
    )
    # The CUDA RMSNorm kernel needs at least 128 features, more than this test's latent width.
    from xllm.models.fused_blocks import moe as fused_moe_module

    fused_moe_module.rmsnorm_fwd = _torch_rmsnorm_fwd
    fused_moe_module.rmsnorm_bwd = _torch_rmsnorm_bwd
    try:
        _assert_tp2_fused_latent_moe_parity(rank, torch.device("cuda", rank), routing_type, post_rmsnorm)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


def _tp2_fused_latent_moe_gloo_worker(rank, init_method, routing_type, post_rmsnorm):
    import importlib

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    initialize_module = importlib.import_module("xllm.distributed.initialize")
    fused_moe_module = importlib.import_module("xllm.models.fused_blocks.moe")
    initialize_module._MODEL_PARALLEL_GROUP = dist.group.WORLD
    initialize_module._MODEL_PARALLEL_RANK = None
    initialize_module._MODEL_PARALLEL_WORLD_SIZE = None
    fused_moe_module.grouped_experts_fwd = _torch_grouped_experts_fwd
    fused_moe_module.grouped_experts_bwd = _torch_grouped_experts_bwd
    fused_moe_module.rmsnorm_fwd = _torch_rmsnorm_fwd
    fused_moe_module.rmsnorm_bwd = _torch_rmsnorm_bwd
    # Gloo does not accept the stacked reduce-scatter layout used by NCCL in
    # the production wrapper. These equivalents preserve the exact TP math.
    fused_moe_module.reduce_scatter = _gloo_reduce_scatter
    fused_moe_module.all_gather = _gloo_all_gather
    try:
        _assert_tp2_fused_latent_moe_parity(rank, torch.device("cpu"), routing_type, post_rmsnorm)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("post_rmsnorm", [False, True])
@pytest.mark.parametrize("routing_type", ["topk", "dynamic_qb"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="TP=2 LatentMoE test requires two GPUs")
def test_fused_latent_moe_tensor_parallel_forward_backward(tmp_path, routing_type, post_rmsnorm):
    mp.spawn(
        _tp2_fused_latent_moe_worker,
        args=(f"file://{tmp_path / 'nccl_init'}", routing_type, post_rmsnorm),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("post_rmsnorm", [False, True])
@pytest.mark.parametrize("routing_type", ["topk", "dynamic_qb"])
def test_fused_latent_moe_tensor_parallel_gloo_forward_backward(tmp_path, routing_type, post_rmsnorm):
    """Exercise TP=2 collectives and gradients without requiring a GPU allocation."""
    mp.spawn(
        _tp2_fused_latent_moe_gloo_worker,
        args=(f"file://{tmp_path / 'gloo_init'}", routing_type, post_rmsnorm),
        nprocs=2,
        join=True,
    )
