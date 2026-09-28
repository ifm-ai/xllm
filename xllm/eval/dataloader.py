from typing import Any, List, Dict, Iterator
from collections import defaultdict
import torch
import numpy as np
from numpy.random import RandomState
from functools import partial

from xllm.data.dataset_streamer.data_iterator.data_iterator import JSONLFileIterator
from xllm.eval.task.base import BaseTask
from xllm.config import ValidConf


class EvalTaskDataloader:
    def __init__(
        self,
        path: str,
        batch_size: int,
        task: BaseTask,
        world_rank: int,
        world_size: int,
        seed: int,
        cfg: ValidConf,
    ):
        self.world_rank = world_rank
        self.world_size = world_size
        self.path = path
        self.batch_size = batch_size
        self.task = task
        self.task.add_template = cfg.add_template
        self.rng = RandomState((seed, world_rank))

    def batch_iterator(self):
        reader = JSONLFileIterator(
            self.path,
            self.world_rank,
            self.world_size,
            infinite=False,
        )
        try:
            examples = [instance.raw_data for instance in reader]
        finally:
            reader.close()

        process_fn = partial(self.task.process, rng=self.rng)
        batch: Dict[str, List] = defaultdict(list)
        curr_bsz = 0
        batch_counter = 0
        for example in examples:
            example = process_fn(example)
            batch["examples"].append(example)
            for k, v in example.items():
                batch[k].append(v)
            curr_bsz += 1
            if curr_bsz == self.batch_size:
                batch_counter += 1
                yield batch_to_tensor(batch, max_length=self.task.max_text_len)
                batch = defaultdict(list)
                curr_bsz = 0
        if curr_bsz > 0:
            yield batch_to_tensor(batch, max_length=self.task.max_text_len)


def batch_to_tensor(batch: Dict, max_length: int) -> Dict:
    if not ("text_x" in batch and "text_y" in batch):
        return batch

    key_padding = {"text_x": 0, "text_y": -100, "completion_x": 0, "completion_y": -100}
    for key, padding in key_padding.items():
        if key in batch:
            batch[key] = to_tensor(batch[key], max_length, padding)

    if "n_completion" in batch:
        batch["completion_index"] = np.cumsum([0] + batch["n_completion"])

    return batch


def to_tensor(
    batch_tokens: List[List[List[int]]], max_length: int, pad_value: int
) -> torch.Tensor:
    tokens = [t for ex_tokens in batch_tokens for t in ex_tokens]
    batch_max_length = max([len(t) for t in tokens])

    max_length = min(batch_max_length, max_length)
    padded_tokens = [
        pad(x, max_length=max_length, value=pad_value, truncating="pre") for x in tokens
    ]
    return torch.tensor(padded_tokens, dtype=torch.int64)


def pad(
    tokens: List[int],
    max_length: int,
    value: int,
    padding: str = "post",
    truncating: str = "post",
):
    if len(tokens) < max_length:
        if padding == "post":
            tokens = tokens + [value] * (max_length - len(tokens))
        elif padding == "pre":
            tokens = [value] * (max_length - len(tokens)) + tokens
    if truncating == "post":
        tokens = tokens[:max_length]
    elif truncating == "pre":
        if len(tokens) > max_length:
            tokens = tokens[len(tokens) - max_length:]
    return tokens
