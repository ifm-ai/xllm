import pytest
import torch
import torch.nn.functional as F
import torch.nn.init as init

from xllm.config import ModelConf
from xllm.models.transformer import TransformerMoEBlock
from xllm.modules.moe.router import build_router, dynamic_qb as router, DynamicQBRouter, TopKRouter
from xllm.modules.rotary_positional_embedding import RotaryEmbedding


@pytest.fixture(autouse=True)
def single_process_hybrid_shard(monkeypatch):
    monkeypatch.setattr(router, 'get_hybrid_shard_data_parallel_world_size', lambda: 1)
    monkeypatch.setattr(router, 'get_hybrid_shard_data_parallel_group', lambda: None)


def _expected_routes(flat_scores, bias):
    # Routes above the threshold, grouped by expert (the grouped-experts dispatch order).
    expert_indices, token_indices = torch.nonzero(flat_scores.t() > bias.view(-1, 1), as_tuple=True)
    return token_indices, expert_indices


def test_dynamic_qb_quantile_uses_target_plus_one_largest_score():
    scores = torch.tensor(
        [
            [[0.9, 0.2, 0.6], [0.8, 0.7, 0.1], [0.4, 0.6, 0.3]],
            [[0.7, 0.1, 0.8], [0.3, 0.9, 0.5], [0.2, 0.8, 0.4]],
        ]
    )

    # Six tokens and target_k=1 give two selected tokens per expert.
    expected = torch.tensor([0.7, 0.7, 0.5])
    torch.testing.assert_close(router.dynamic_qb_quantile(scores, topk=1), expected)


def test_dynamic_qb_binary_search_matches_exact_quantile():
    scores = torch.tensor(
        [
            [[0.10, 0.25, 0.31, 0.93], [0.77, 0.12, 0.44, 0.02], [0.49, 0.87, 0.19, 0.60]],
            [[0.33, 0.55, 0.81, 0.71], [0.91, 0.43, 0.08, 0.36], [0.66, 0.74, 0.52, 0.15]],
        ]
    )

    expected = router.dynamic_qb_quantile(scores, topk=2)
    actual = router.dynamic_qb_binary_search(
        scores, topk=2, world_size=1, group=None, num_iters=32
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.0)


def test_dynamic_qb_histogram_tracks_exact_quantile():
    scores = torch.tensor(
        [
            [[0.11, 0.21, 0.41, 0.91], [0.73, 0.18, 0.49, 0.04], [0.47, 0.84, 0.16, 0.62]],
            [[0.36, 0.57, 0.79, 0.69], [0.94, 0.46, 0.09, 0.34], [0.64, 0.76, 0.54, 0.13]],
        ]
    )
    num_bins = 4096

    expected = router.dynamic_qb_quantile(scores, topk=2)
    actual = router.dynamic_qb_histogram(
        scores, topk=2, world_size=1, group=None, num_bins=num_bins
    )
    flat_scores = scores.flatten(end_dim=-2)
    bin_width = (flat_scores.max(dim=0).values - flat_scores.min(dim=0).values) / num_bins
    assert torch.all((actual - expected).abs() <= bin_width + 1e-6)


@pytest.mark.parametrize('update_method', ['average', 'histogram', 'binary_search'])
def test_dynamic_qb_threshold_update_methods(update_method):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    bias = torch.full((5,), 0.5)

    with torch.no_grad():
        original_scores = torch.sigmoid(F.linear(x, weight).float())
        if update_method == 'average':
            expected = router.dynamic_qb_quantile(original_scores, topk=2)
        elif update_method == 'histogram':
            expected = router.dynamic_qb_histogram(original_scores, topk=2, world_size=1, group=None)
        else:
            expected = router.dynamic_qb_binary_search(original_scores, topk=2, world_size=1, group=None)

    scores, _, _ = router.dynamic_qb_routing(
        x, weight, 5, 2, 'sigmoid', bias, update_method, 0.0, False, None
    )
    scores.sum().backward()
    torch.testing.assert_close(bias, expected)


def test_dynamic_qb_uses_topk_normalized_scores_and_updates_after_forward():
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    bias = torch.full((5,), 0.5)

    with torch.no_grad():
        original_scores = torch.sigmoid(F.linear(x, weight).float())
        old_bias = bias.clone()
        flat_scores = original_scores.view(-1, 5)
        expected_token_indices, expected_expert_indices = _expected_routes(flat_scores, old_bias)
        expected_scores = flat_scores[expected_token_indices, expected_expert_indices]
        routes_per_token = torch.bincount(
            expected_token_indices, minlength=flat_scores.shape[0]
        )
        normalize_mask = routes_per_token[expected_token_indices] > 1
        denom = torch.zeros(flat_scores.shape[0])
        denom.index_add_(0, expected_token_indices, expected_scores)
        normalized_scores = expected_scores / denom[expected_token_indices]
        expected_scores = torch.where(normalize_mask, normalized_scores, expected_scores)
        expected_next_bias = router.dynamic_qb_quantile(original_scores, topk=2)

    scores, token_indices, tokens_per_expert = router.dynamic_qb_routing(
        x, weight, 5, 2, 'sigmoid', bias, 'average', 0.0, False, None
    )

    torch.testing.assert_close(bias, old_bias)
    torch.testing.assert_close(scores, expected_scores)
    # routes are grouped by expert
    torch.testing.assert_close(token_indices, expected_token_indices)
    torch.testing.assert_close(tokens_per_expert, torch.bincount(expected_expert_indices, minlength=5))

    scores.sum().backward()
    torch.testing.assert_close(bias, expected_next_bias)


def test_dynamic_qb_allows_two_forwards_before_backward():
    # Backward updates the threshold in place; that must not invalidate a pending backward.
    weight = torch.randn(5, 4, requires_grad=True)
    bias = torch.full((5,), 0.5)
    outputs = [
        router.dynamic_qb_routing(
            torch.randn(2, 3, 4), weight, 5, 2, 'sigmoid', bias, 'average', 0.5, False, None
        )[0]
        for _ in range(2)
    ]
    for scores in outputs:
        scores.sum().backward()
    assert weight.grad is not None


def test_dynamic_qb_globalnorm_matches_autograd():
    x = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        requires_grad=True,
    )
    bias = torch.full((4,), 0.2)

    scores, token_indices, tokens_per_expert = router.dynamic_qb_routing(
        x, weight, 4, 2, 'sigmoid', bias, 'average', 0.0, True, None
    )

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    original_scores = torch.sigmoid(F.linear(x_ref, weight_ref).float())
    normalized_scores = original_scores / original_scores.sum(dim=-1, keepdim=True)
    flat_scores = normalized_scores.view(-1, 4)
    expected_token_indices, expected_expert_indices = _expected_routes(flat_scores, bias)
    expected_scores = flat_scores[expected_token_indices, expected_expert_indices]

    torch.testing.assert_close(token_indices, expected_token_indices)
    torch.testing.assert_close(tokens_per_expert, torch.bincount(expected_expert_indices, minlength=4))
    torch.testing.assert_close(scores, expected_scores)

    score_weights = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    (scores * score_weights).sum().backward()
    (expected_scores * score_weights).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)
    torch.testing.assert_close(
        bias, router.dynamic_qb_quantile(normalized_scores.detach(), topk=2)
    )


def test_dynamic_qb_topk_normalization_matches_autograd():
    x = torch.tensor(
        [[[2.0, 2.0], [2.0, 0.0], [-2.0, -2.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    weight = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        requires_grad=True,
    )
    bias = torch.full((4,), 0.5)

    scores, token_indices, tokens_per_expert = router.dynamic_qb_routing(
        x, weight, 4, 2, 'sigmoid', bias, 'average', 0.9, False, None
    )

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    flat_scores = torch.sigmoid(F.linear(x_ref, weight_ref).float()).view(-1, 4)
    expected_token_indices, expected_expert_indices = _expected_routes(flat_scores, bias)
    expected_scores = flat_scores[expected_token_indices, expected_expert_indices]
    routes_per_token = torch.bincount(expected_token_indices, minlength=4)
    normalize_mask = routes_per_token[expected_token_indices] > 1
    denom = torch.zeros(4)
    denom.index_add_(0, expected_token_indices, expected_scores)
    normalized_scores = expected_scores / denom[expected_token_indices]
    expected_scores = torch.where(normalize_mask, normalized_scores, expected_scores)

    torch.testing.assert_close(token_indices, expected_token_indices)
    torch.testing.assert_close(tokens_per_expert, torch.bincount(expected_expert_indices, minlength=4))
    torch.testing.assert_close(scores, expected_scores)

    score_weights = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
    (scores * score_weights).sum().backward()
    (expected_scores * score_weights).sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)


def test_dynamic_qb_globalnorm_scales_initial_sigmoid_bias():
    kwargs = dict(
        weight_std=0.02,
        in_features=64,
        num_experts=128,
        topk=4,
        score_func='sigmoid',
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    raw_bias = router.dynamic_qb_initial_bias(globalnorm=False, **kwargs)
    normalized_bias = router.dynamic_qb_initial_bias(globalnorm=True, **kwargs)
    torch.testing.assert_close(normalized_bias, raw_bias * (2.0 / 128))


def test_dynamic_qb_single_route_keeps_router_gradient():
    x = torch.ones(2, 3, 4, requires_grad=True)
    weight = torch.zeros(3, 4, requires_grad=True)
    with torch.no_grad():
        weight[0].fill_(0.1)
    bias = torch.tensor([0.5, 0.99, 0.99])

    scores, token_indices, tokens_per_expert = router.dynamic_qb_routing(
        x, weight, 3, 2, 'sigmoid', bias, 'average', 0.9, False, None
    )

    num_tokens = x.shape[0] * x.shape[1]
    torch.testing.assert_close(token_indices, torch.arange(num_tokens))
    torch.testing.assert_close(tokens_per_expert, torch.tensor([num_tokens, 0, 0]))
    assert not torch.allclose(scores, torch.ones_like(scores))

    scores.sum().backward()
    assert weight.grad is not None
    assert weight.grad[0].abs().sum() > 0


def test_dynamic_qb_allows_zero_routes():
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.zeros(5, 4, requires_grad=True)
    bias = torch.full((5,), 0.9)

    scores, token_indices, tokens_per_expert = router.dynamic_qb_routing(
        x, weight, 5, 2, 'sigmoid', bias, 'average', 0.9, False, None
    )

    assert scores.numel() == 0
    assert token_indices.numel() == 0
    torch.testing.assert_close(tokens_per_expert, torch.zeros(5, dtype=torch.long))

    scores.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def test_dynamic_qb_scaling_preserves_normalized_score_sum():
    x = torch.ones(1, 2, 3, requires_grad=True)
    weight = torch.full((4, 3), 0.1, requires_grad=True)
    bias = torch.zeros(4)
    scaling_factor = 2.5

    scores, token_indices, _ = router.dynamic_qb_routing(
        x, weight, 4, 2, 'sigmoid', bias, 'average', 0.9, False, scaling_factor
    )

    expected = torch.tensor(scaling_factor / 4)
    torch.testing.assert_close(scores, torch.full_like(scores, expected))
    per_token = torch.zeros(2)
    per_token.index_add_(0, token_indices, scores.detach())
    torch.testing.assert_close(per_token, torch.full((2,), scaling_factor))


def _combine_inputs(num_experts, num_tokens, dim, device):
    # routes grouped by expert, as dynamic_qb_select returns them; token 7 gets no route
    generator = torch.Generator(device=device).manual_seed(0)
    mask = torch.rand(num_experts, num_tokens, generator=generator, device=device) < 0.5
    mask[:, 7] = False
    _, token_indices = torch.nonzero(mask, as_tuple=True)
    num_routes = token_indices.numel()
    values = torch.randn(num_routes, dim, generator=generator, device=device).to(torch.bfloat16)
    weights = torch.rand(num_routes, generator=generator, device=device)
    return token_indices, values, weights


@pytest.mark.parametrize('weighted', [False, True])
def test_dynamic_qb_combine_sums_routes_in_fp32(monkeypatch, weighted):
    num_tokens, dim = 200, 8
    token_indices, values, weights = _combine_inputs(8, num_tokens, dim, 'cpu')
    # 300 routes per chunk, so the ~800 routes are split at token boundaries
    monkeypatch.setattr(router, '_COMBINE_CHUNK_NUMEL', 300 * dim)
    lengths = torch.bincount(token_indices, minlength=num_tokens)
    chunks = router._token_chunks(lengths, token_indices.numel(), 300)
    assert len(chunks) > 1
    assert [c[0] for c in chunks[1:]] == [c[1] for c in chunks[:-1]]
    assert [c[2] for c in chunks[1:]] == [c[3] for c in chunks[:-1]]
    assert (chunks[0][0], chunks[-1][1], chunks[0][2], chunks[-1][3]) == (0, num_tokens, 0, token_indices.numel())

    out = router.dynamic_qb_combine(values, token_indices, num_tokens, weights if weighted else None)
    reference = torch.zeros(num_tokens, dim, dtype=torch.float64).index_add_(
        0, token_indices, values.double() * (weights.double().unsqueeze(1) if weighted else 1.0)
    )
    assert out.dtype == torch.bfloat16
    # fp32 accumulation: only the final bf16 rounding remains
    assert torch.all((out.double() - reference).abs() <= reference.abs() * 2 ** -8 + 1e-6)
    assert torch.all(out[7] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_dynamic_qb_combine_is_deterministic():
    # index_add_ accumulates with atomics, in a different order every run
    token_indices, values, weights = _combine_inputs(64, 4096, 256, 'cuda')
    outs = [router.dynamic_qb_combine(values, token_indices, 4096, weights) for _ in range(5)]
    assert all(torch.equal(outs[0], out) for out in outs[1:])


def test_dynamic_qb_router_rejects_auxiliary_loss():
    dynamic_router = DynamicQBRouter(4, 8, 2)
    x = torch.randn(1, 2, 4)

    with pytest.raises(AssertionError, match='does not use auxiliary'):
        dynamic_router(x, 'entropy')


def test_dynamic_qb_config_allows_fused_router_recompute():
    ModelConf(
        num_experts=4,
        num_activated_experts=2,
        expert_inter_dim=8,
        moe_router_type='dynamic_qb',
        fused_block=True,
        recompute_router=True,
        causal_attn_backend='flash',
    )


def test_dynamic_qb_config_uses_new_update_method_names():
    config = ModelConf(
        num_experts=4,
        num_activated_experts=2,
        expert_inter_dim=8,
        moe_router_type='dynamic_qb',
    )
    assert config.moe_router_dynamic_qb_update_method == 'average'
    assert config.moe_router_dynamic_qb_globalnorm

    for update_method in ['average', 'histogram', 'binary_search']:
        ModelConf(
            num_experts=4,
            num_activated_experts=2,
            expert_inter_dim=8,
            moe_router_type='dynamic_qb',
            moe_router_dynamic_qb_update_method=update_method,
        )
    with pytest.raises(AssertionError):
        ModelConf(
            num_experts=4,
            num_activated_experts=2,
            expert_inter_dim=8,
            moe_router_type='dynamic_qb',
            moe_router_dynamic_qb_update_method='quantile',
        )


@pytest.mark.parametrize('globalnorm', [False, True])
def test_fused_dynamic_qb_router_recompute_matches_saved_scores(globalnorm):
    from xllm.models.fused_blocks import moe as fused_moe

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor(
        [
            [[1.0, 0.0, 0.5, -0.5], [-1.0, 0.0, -0.5, 0.5], [0.0, 1.0, 0.5, 0.5]],
            [[0.0, -1.0, -0.5, -0.5], [1.0, 1.0, 0.5, 0.0], [-1.0, -1.0, 0.0, 0.5]],
        ],
        device=device,
    )
    router_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
        ],
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(17)
    expert_w1 = torch.randn(4, 6, 4, generator=generator, device=device)
    expert_w2 = torch.randn(4, 4, 6, generator=generator, device=device)
    expert_w3 = torch.randn(4, 6, 4, generator=generator, device=device)
    initial_bias = torch.full((4,), 0.2 if globalnorm else 0.5, device=device)

    # n x s x d experts, stored as (n s) x d
    flat_w1, flat_w2, flat_w3 = expert_w1.flatten(0, 1), expert_w2.flatten(0, 1), expert_w3.flatten(0, 1)

    out, _, routing_out, shared_hiddens, routing_state, rng_states = fused_moe.moe_fwd(
        x, router_weight, initial_bias, 'sigmoid', 1.0, None,
        None, None, None, 4, 4, 0, 4, 2,
        'dynamic_qb', globalnorm,
        flat_w1, flat_w2, flat_w3, 'sequential', 'torch', 0.0, False, False,
    )
    original_scores, routing_indices, selected_scores, token_indices, tokens_per_expert = routing_out
    h1, h3 = shared_hiddens
    group_sizes, input_splits, output_splits, curr_bsz = routing_state
    expert_rng_state, shared_rng_state = rng_states
    out_grad = torch.randn(out.shape, generator=generator, device=device)

    routing_scores = original_scores
    if globalnorm:
        routing_scores = routing_scores / routing_scores.sum(dim=-1, keepdim=True)
    flat_scores = routing_scores.view(-1, 4)
    expected_token_indices, expected_expert_indices = torch.nonzero(
        flat_scores > initial_bias.view(1, 4), as_tuple=True
    )
    expected_scores = flat_scores[expected_token_indices, expected_expert_indices]
    if not globalnorm:
        routes_per_token = torch.bincount(
            expected_token_indices, minlength=x.shape[0] * x.shape[1]
        )
        normalize_mask = routes_per_token[expected_token_indices] > 1
        denom = torch.zeros(x.shape[0] * x.shape[1], device=device)
        denom.index_add_(0, expected_token_indices, expected_scores)
        normalized_scores = expected_scores / denom[expected_token_indices]
        expected_scores = torch.where(normalize_mask, normalized_scores, expected_scores)
    flat_x = x.view(-1, x.shape[-1])
    routed_x = flat_x[expected_token_indices]
    expert_h1 = F.silu(torch.einsum('rd,rhd->rh', routed_x, expert_w1[expected_expert_indices]))
    expert_h3 = torch.einsum('rd,rhd->rh', routed_x, expert_w3[expected_expert_indices])
    routed_out = torch.einsum(
        'rh,rdh->rd', expert_h1 * expert_h3, expert_w2[expected_expert_indices]
    )
    expected_out = torch.zeros_like(flat_x)
    expected_out.index_add_(0, expected_token_indices, routed_out * expected_scores.unsqueeze(1))
    torch.testing.assert_close(out, expected_out.view_as(out))

    def backward(scores, bias):
        return fused_moe.moe_bwd(
            out_grad, None, x, x.flatten(end_dim=-2),
            scores, routing_indices, selected_scores, token_indices, tokens_per_expert,
            router_weight, bias, 0.0, 'sigmoid', 1.0, None,
            h1, h3, None, None, None, 4, 4, 2,
            'dynamic_qb', 'average', 0.3, globalnorm,
            flat_w1, flat_w2, flat_w3, 'sequential', 'torch', 0.0,
            group_sizes, input_splits, output_splits, curr_bsz,
            expert_rng_state, shared_rng_state, False,
        )

    saved_bias = initial_bias.clone()
    recomputed_bias = initial_bias.clone()
    saved_grads = backward(original_scores, saved_bias)
    recomputed_grads = backward(None, recomputed_bias)

    # The routes are combined with atomic adds on GPU, so the grads match up to summation order.
    for saved, recomputed in zip(saved_grads, recomputed_grads):
        if saved is None:
            assert recomputed is None
        else:
            torch.testing.assert_close(recomputed, saved)
    torch.testing.assert_close(recomputed_bias, saved_bias, rtol=0, atol=0)


def test_build_router_types():
    assert type(build_router('topk', 8, 4, 2)) is TopKRouter
    assert type(build_router('dynamic_qb', 8, 4, 2)) is DynamicQBRouter
    with pytest.raises(AssertionError, match='thresholds'):
        build_router('dynamic_qb', 8, 4, 2, bias=True, bias_update_rate=1e-3)
    with pytest.raises(ValueError, match='Unknown router type'):
        build_router('expert_choice', 8, 4, 2)


def test_routers_share_weight_initialization():
    torch.manual_seed(0)
    topk_router = TopKRouter(8, 4, 2, bias=True, bias_update_rate=1e-3)
    torch.manual_seed(0)
    dynamic_router = DynamicQBRouter(8, 4, 2)
    torch.manual_seed(0)
    expected = torch.empty(4, 8)
    init.xavier_normal_(expected)

    torch.testing.assert_close(topk_router.weight.data, expected, rtol=0, atol=0)
    torch.testing.assert_close(dynamic_router.weight.data, expected, rtol=0, atol=0)
    torch.testing.assert_close(topk_router.bias, torch.zeros(4))
    assert list(topk_router.state_dict()) == ['weight', 'bias']
    assert list(dynamic_router.state_dict()) == ['weight', 'bias']


@pytest.mark.parametrize('router_type', ['topk', 'dynamic_qb'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='fused Transformer block requires CUDA')
def test_fused_and_unfused_moe_blocks_match(router_type):
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
        moe_router_type=router_type,
        moe_router_score_func='sigmoid' if router_type == 'dynamic_qb' else 'softmax',
        causal_attn_backend='flash',
        fused_block=True,
        apply_rmsnorm=True,
    )
    block = TransformerMoEBlock(cfg, layer_id=0).cuda().to(torch.bfloat16)
    seq_len = 16
    x = torch.randn(1, seq_len, cfg.model_dim, device='cuda', dtype=torch.bfloat16)
    freqs_cis = RotaryEmbedding(cfg.head_dim, seq_len, cfg.rope_base).get_freqs_cis(0, seq_len, x.device)
    # Backward updates the dynamic QB thresholds, so both passes start from the same ones.
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
    for name, unfused_grad in unfused_grads.items():
        error = (fused_grads[name] - unfused_grad).norm() / unfused_grad.norm().clamp_min(1e-6)
        assert error < 3e-2, f'{name}: relative gradient error {error:.3e}'
