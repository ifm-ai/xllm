import datetime
import os

import pytest
import torch

from xllm.distributed.initialize import initialize_model_parallel, model_parallel_is_initialized


@pytest.fixture
def exact_matmul():
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32


@pytest.fixture(scope="module")
def single_process():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29523")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0,
                                             timeout=datetime.timedelta(seconds=600))
    if not model_parallel_is_initialized():
        initialize_model_parallel(1, 1, 600)
