"""Tests for train.py gradient clipping and resume.

Run on one GPU: pytest -q tests/test_train_fixes.py
"""
import copy
import math
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.optim import AdamW

import train
from xllm.config import ModelConf, OptimConf
from xllm.distributed import initialize_model_parallel
from xllm.models import build_model
from xllm.monitor import Timings
from xllm.optim import build_lr_scheduler


@pytest.fixture(scope="module", autouse=True)
def distributed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not dist.is_initialized():
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(port))
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", rank=0, world_size=1)
        initialize_model_parallel(1, 1, 600)
    yield


class _Tokenizer:
    vocab_size = 512
    bos_id = 1
    eos_id = 2
    pad_id = 0


def build():
    cfg = ModelConf(
        arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", num_layers=2, model_dim=64,
        num_heads=4, head_dim=16, ffn_hidden_dim=128, chunk_size=16, vocab_size=_Tokenizer.vocab_size,
        apply_rmsnorm=True, rope_head_dim=16,
    )
    torch.manual_seed(0)
    return build_model(
        cfg, dtype="fp32", fully_sharded_size=None, fp32_reduce_scatter=True,
        reshard_after_forward=True, forward_prefetch=False, tokenizer=_Tokenizer(),
    )


def local(p):
    return p.to_local() if hasattr(p, "to_local") else p


@pytest.mark.parametrize("clip,scale,expected", [
    (1.0, 1.0, 1.0),
    (1.0, 4.0, 4.0),
    (0.0, 4.0, math.inf),
    (-1.0, 4.0, math.inf),
])
def test_clip_max_norm(clip, scale, expected):
    assert train.get_clip_max_norm(clip, scale) == expected


def test_infinite_clip_threshold_leaves_gradients_unchanged():
    model = build()
    tokens = torch.randint(3, _Tokenizer.vocab_size, (2, 33), device="cuda")
    loss, _, _ = model(tokens[:, :-1], multi_segments=False, targets=tokens[:, 1:])
    loss.mean().backward()
    grads = [local(p.grad).clone() for p in model.parameters()]
    expected_norm = torch.cat([g.flatten() for g in grads]).norm().item()
    norm = model.clip_grad_norm_(math.inf)
    assert norm == pytest.approx(expected_norm, rel=1e-5)
    assert all(torch.equal(local(p.grad), g) for p, g in zip(model.parameters(), grads))


def test_resume_keeps_one_base_lr_per_param_group():
    cfg = OptimConf(lr=1e-2, warmup=10, scheduler="cosine")
    weights = [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))]

    def build_optim():
        optimizer = AdamW([{"params": [weights[0]], "lr": 1e-2}, {"params": [weights[1]], "lr": 1e-3}])
        return optimizer, build_lr_scheduler(optimizer, cfg, 100)

    # uninterrupted: 20 scheduler steps
    optimizer, scheduler = build_optim()
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    training_state = {
        "scheduler": copy.deepcopy(scheduler.state_dict()), "step": 10, "scale": 1.0, "scale_updates": 0,
        "clip_cumulative": 0, "data_state": None, "timings": {},
    }
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    expected = [group["lr"] for group in optimizer.param_groups]

    # resumed at step 10
    optimizer, scheduler = build_optim()
    data_iterator = SimpleNamespace(set_state=lambda state: None)
    train.set_optim_state(scheduler, training_state, SimpleNamespace(), data_iterator, Timings())
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert scheduler.base_lrs == [1e-2, 1e-3]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(expected, rel=1e-12)
