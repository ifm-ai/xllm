"""Aux-loss gradients through a stack of fused MoVA blocks.

Transformer.forward sums the layers' aux losses with `+`, so autograd hands one aux-loss grad tensor to every
layer's backward. A layer's backward must not change it in place, or the layers it reaches later (the earlier
ones) get a scaled gradient. Each layer's router gradients from the summed loss are compared with the sum of
the gradients from each layer's aux loss backpropagated alone, and with the unfused path on the same weights.

Needs a GPU (te expert and value backends, flash attention): pytest tests/moe/test_fused_mova_aux_grad.py
"""
import pytest
import torch
import torch.distributed as dist

from xllm.config import ModelConf
from xllm.distributed import destroy_model_parallel, initialize_model_parallel
from xllm.models.transformer import TransformerMoVABlock
from xllm.modules import RotaryEmbedding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoVA blocks need a GPU")

NUM_LAYERS = 4
DIM, SEQ, BSZ = 512, 512, 2


@pytest.fixture
def one_rank():
    dist.init_process_group("nccl", rank=0, world_size=1, store=dist.HashStore())
    initialize_model_parallel(1, 1, timeout=600)
    yield
    destroy_model_parallel()
    dist.destroy_process_group()


def mova_stack():
    cfg = ModelConf(
        arch='transformer', num_layers=NUM_LAYERS + 1, num_dense_layers=1, model_dim=DIM, num_heads=8,
        num_kv_heads=4, head_dim=128, rope_head_dim=128, rope_base=500000, apply_attn_gate=True,
        attn_gate_func='softplus', ffn_hidden_dim=1024, swiglu=True, num_values=16, num_activated_values=4,
        num_experts=16, num_activated_experts=4, num_shared_experts=1, expert_inter_dim=128,
        moe_router_score_func='sigmoid', moe_router_bias=True, moe_router_bias_update_rate=1e-3,
        moe_router_scaling_factor=2.5, apply_rmsnorm=True, layernorm_num_groups=2, rmsnorm_eps=1e-6,
        causal_attn_backend='flash', fused_block=True, moe_expert_backend='te', mova_backend='te',
        moe_permutation_backend='triton', init_mode='gaussian', init_std=0.02, recompute_q=True, recompute_v=True,
        recompute_fc1_out=True, recompute_fc3_out=True, recompute_router=False,
    )
    torch.manual_seed(1234)
    blocks = torch.nn.ModuleList([TransformerMoVABlock(cfg, layer_id=i + 1) for i in range(NUM_LAYERS)])
    return blocks.cuda().to(torch.bfloat16).train()


def router_grads(blocks, x0, freqs, fused, only_layer=None):
    """Router weight gradients of each layer when only the aux loss (all layers', or one layer's) is backpropagated."""
    blocks.zero_grad(set_to_none=True)
    for b in blocks:
        b.fused_block = fused
    h = x0.clone().requires_grad_()
    auxes = []
    for b in blocks:
        h, aux, _ = b(h, freqs, None, None, 'dot', 0, False)
        auxes.append(aux)
    aux_loss = auxes[only_layer] if only_layer is not None else sum(auxes[1:], auxes[0])  # summed as in Transformer
    ((h.float() * 0).sum() + aux_loss).backward()
    return [(b.mova.router.weight.grad.double(), b.moe.router.weight.grad.double()) for b in blocks]


def test_each_layer_gets_its_full_aux_loss_gradient(one_rank):
    blocks = mova_stack()
    buffers = {n: b.clone() for n, b in blocks.named_buffers()}  # the router bias updates in backward

    def run(fused, only_layer=None):
        with torch.no_grad():
            for n, b in blocks.named_buffers():
                b.copy_(buffers[n])
        return router_grads(blocks, x0, freqs, fused, only_layer)

    x0 = torch.randn(BSZ, SEQ, DIM, device='cuda', generator=torch.Generator('cuda').manual_seed(9)).bfloat16()
    freqs = RotaryEmbedding(128, SEQ * 2, base=500000).get_freqs_cis(0, SEQ, 'cuda')

    summed = run(fused=True)
    unfused = run(fused=False)
    alone = [run(fused=True, only_layer=i) for i in range(NUM_LAYERS)]

    for i in range(NUM_LAYERS):
        for j, name in enumerate(('mova.router', 'moe.router')):
            ref = sum(g[i][j] for g in alone)
            assert ref.norm() > 0, f"layer {i} {name}: no aux-loss gradient"
            ratio = ((summed[i][j] * ref).sum() / (ref * ref).sum()).item()
            unfused_ratio = ((unfused[i][j] * ref).sum() / (ref * ref).sum()).item()
            assert ratio == pytest.approx(1.0, abs=0.02), f"layer {i} {name}: fused summed/alone {ratio:.4f}"
            assert unfused_ratio == pytest.approx(1.0, abs=0.05), f"layer {i} {name}: unfused/alone {unfused_ratio:.4f}"


def test_fused_blocks_train_without_a_load_balancing_loss(one_rank):
    # moe_router_load_balancing_type None: the blocks return no aux loss, so the backward gets no aux-loss grad
    blocks = mova_stack()
    x0 = torch.randn(BSZ, SEQ, DIM, device='cuda', generator=torch.Generator('cuda').manual_seed(9)).bfloat16()
    freqs = RotaryEmbedding(128, SEQ * 2, base=500000).get_freqs_cis(0, SEQ, 'cuda')
    h = x0.clone().requires_grad_()
    x = h
    for b in blocks:
        x, aux, _ = b(x, freqs, None, None, None, 0, False)
    assert aux is None
    x.float().sum().backward()
    assert torch.isfinite(h.grad).all()
