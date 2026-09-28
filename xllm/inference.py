from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.distributed.utils import reduce_scalar
from xllm.models import init_cache

logger = getLogger()

MAX_CHUNKS_PER_BATCH = 4


@torch.inference_mode()
def inference(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    multi_segments: bool,
) -> torch.Tensor:
    model.eval()

    if multi_segments:
        if model.support_multi_segments_with_cache():
            return _inference_with_cache(model, tokens, targets, multi_segments)
        else:
            return _inference_without_cache(model, tokens, targets, multi_segments)
    else:
        return _inference_with_cache(model, tokens, targets, multi_segments)


def _inference_with_cache(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    multi_segments: bool,
) -> torch.Tensor:
    model.eval()

    bsz, length = tokens.shape
    max_length = get_max_length(length)
    tokens = F.pad(tokens, (0, max_length - length), value=0)
    targets = F.pad(targets, (0, max_length - length), value=-100)
    n_chunks = max_length // model.chunk_size
    prev_pos = n_chunks * model.chunk_size

    cache = init_cache(model)
    losses = []
    for c in range(0, n_chunks, MAX_CHUNKS_PER_BATCH):
        s = c * model.chunk_size
        e = min(c + MAX_CHUNKS_PER_BATCH, n_chunks) * model.chunk_size
        # B x L x V
        x = tokens[:, s:e]
        y = targets[:, s:e]
        loss, _, cache = model(x, multi_segments, targets=y, cache=cache)
        losses.append(loss)

    if prev_pos < max_length:
        x = tokens[:, prev_pos:]
        y = targets[:, prev_pos:]
        loss, _, _ = model(x, multi_segments, targets=y, cache=cache)
        losses.append(loss)

    # B x L
    loss = torch.cat(losses, dim=1)[:, :length]
    return loss


def _inference_without_cache(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    multi_segments: bool,
) -> torch.Tensor:
    model.eval()

    loss, _, _ = model(tokens, multi_segments, targets=targets)
    return loss


def get_max_length(length: int) -> int:
    # reduce max length prompt over all processes to have an equal number of call on each process with fsdp
    if torch.distributed.is_initialized():
        max_length = int(reduce_scalar(length, op='max'))
    else:
        max_length = length
    return max_length
