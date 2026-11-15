import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from xllm.distributed.wrap import enable_wrap, wrap

# With optimizer offload the model is built and initialized in fp32 and wrap() casts only the params it shards to
# bf16; buffers stay fp32 (a bf16 router bias would drop its +-1e-3 updates). One-rank gloo group on CPU.


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 16)
        self.register_buffer("bias", torch.zeros(4))

    def forward(self, x):
        return self.lin(x)


def test_wrap_casts_sharded_params_and_keeps_buffers_fp32():
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        mesh = init_device_mesh("cpu", (1,))
        with enable_wrap(mesh=mesh, storage_dtype=torch.bfloat16):
            inner = wrap(Block())
            root = wrap(nn.Sequential(inner, nn.Linear(16, 8)))
        params = list(root.parameters())
        assert params and all(isinstance(p, DTensor) and p.dtype == torch.bfloat16 for p in params)
        assert inner.bias.dtype == torch.float32
    finally:
        dist.destroy_process_group()
