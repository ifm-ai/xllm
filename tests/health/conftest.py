import pytest
import torch.distributed as dist

import tests.health.helpers as helpers  # installs the CPU stubs before any xllm import
import xllm.distributed.initialize as parallel_state
from xllm.modules.rms_norm import GroupRMSNorm


@pytest.fixture(autouse=True)
def cpu_norms(monkeypatch):
    monkeypatch.setattr(GroupRMSNorm, "forward", helpers.cpu_rms_norm_forward)


@pytest.fixture
def single_process_group(monkeypatch):
    # FSDP needs a process group; a one-rank gloo group lives entirely in this process.
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    monkeypatch.setattr(parallel_state, "_DATA_PARALLEL_GROUP", dist.group.WORLD)
    yield
    dist.destroy_process_group()
