import pytest
import torch

import xllm.optim.cpu_master_adamw as cma
from xllm.optim.cpu_master_adamw import CPUMasterAdamW

# CPUMasterAdamW must equal mixed-precision AdamW with fp32 masters: fp32 master weights, forward and backward on
# their bf16 copy, bf16 grads summed in fp32 over the micro-batches, clipped by the 2-norm, then fused AdamW.
# Both paths see the same bf16 weights every step, so the bf16 grads match exactly and the masters must too.
# One GPU, plain tensors (the FSDP2 DTensor path differs only in to_local() and the norm's all-reduce).

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
KW = dict(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
SHAPES = [(512, 256), (512,), (128, 512)]


@pytest.fixture(autouse=True)
def single_model_parallel_rank(monkeypatch):
    monkeypatch.setattr(cma, "get_model_parallel_world_size", lambda: 1)


def _loss(ws, x):
    return (torch.relu(x @ ws[0].t() + ws[1]) @ ws[2].t()).pow(2).mean()


def _init():
    torch.manual_seed(0)
    return [(torch.randn(s) * 0.05).bfloat16().float() for s in SHAPES]  # a bf16 checkpoint


def test_matches_fp32_master_adamw_with_accumulation_and_clipping():
    acc_steps, steps, clip = 3, 6, 0.5
    data = [[torch.randn(64, 256) for _ in range(acc_steps)] for _ in range(steps)]
    ref = _init()
    ref_opt = torch.optim.AdamW(ref, fused=True, **KW)
    gpu = [w.to("cuda", torch.bfloat16).requires_grad_() for w in ref]
    opt = CPUMasterAdamW(gpu, **KW)
    clipped = 0
    for step in range(steps):
        acc = [torch.zeros_like(w) for w in ref]
        for x in data[step]:
            wb = [w.to("cuda", torch.bfloat16).requires_grad_() for w in ref]
            _loss(wb, x.to("cuda", torch.bfloat16)).backward()
            for a, w in zip(acc, wb):
                a.add_(w.grad.cpu())
            _loss(gpu, x.to("cuda", torch.bfloat16)).backward()
            opt.accumulate()
        norm = torch.sqrt(sum(torch.linalg.vector_norm(a).double() ** 2 for a in acc)).item()
        if clip / (norm + 1e-6) < 1.0:
            torch._foreach_mul_(acc, clip / (norm + 1e-6))
            clipped += 1
        for w, a in zip(ref, acc):
            w.grad = a
        ref_opt.step()
        assert opt.clip_grad_norm_(clip) == pytest.approx(norm, rel=1e-9)
        opt.step()
        for m, w, g in zip(opt.masters, ref, gpu):
            assert torch.equal(m, w)
            assert torch.equal(g.detach().cpu(), w.to(torch.bfloat16))
    assert clipped > 0  # the clipping branch was exercised


def test_state_dict_round_trip_restores_masters_and_gpu_copy():
    gpu = [w.to("cuda", torch.bfloat16).requires_grad_() for w in _init()]
    opt = CPUMasterAdamW(gpu, **KW)
    for _ in range(2):
        _loss(gpu, torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).backward()
        opt.accumulate()
        opt.step()
    gpu2 = [torch.zeros_like(g).requires_grad_() for g in gpu]
    opt2 = CPUMasterAdamW(gpu2, **KW)
    opt2.load_state_dict(opt.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(opt2.masters, opt.masters))
    assert all(torch.equal(a, b) for a, b in zip(gpu2, gpu))
    assert all(torch.equal(opt2.state[a]["exp_avg"], opt.state[b]["exp_avg"]) for a, b in zip(opt2.masters, opt.masters))


def test_sync_masters_from_params_after_weights_are_loaded():
    gpu = [torch.zeros(s, device="cuda", dtype=torch.bfloat16, requires_grad=True) for s in SHAPES]
    opt = CPUMasterAdamW(gpu, **KW)
    loaded = _init()
    with torch.no_grad():
        for g, w in zip(gpu, loaded):
            g.copy_(w)  # what loading a checkpoint into the model does after the optimizer was built
    opt.sync_masters_from_params()
    assert all(torch.equal(m, w) for m, w in zip(opt.masters, loaded))
