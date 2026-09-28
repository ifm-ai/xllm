from typing import Optional, Callable, Sequence, List, Dict
import json
import glob
import shutil
import logging
import multiprocessing
import re
import string
from pathlib import Path
from collections import Counter
import numpy as np
import torch

from xllm.distributed import (
    get_model_parallel_rank,
    get_context_parallel_rank,
)
from xllm.distributed.utils import reduce_scalar
from xllm.eval.task import BaseTask
from xllm.eval.coding import execute
from xllm.eval.math import math_accuracy
from xllm.utils import mkdir

logger = logging.getLogger()

"""
Normalization and score functions from SQuAD evaluation script
https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
"""


def weighted_average(x, count):
    loss = reduce_scalar(x * count, op='sum')
    total = reduce_scalar(count, op='sum')
    return (loss / total), total


def avg_dist_dict(
    keys: List[str], dictionary: Dict[str, List[float]]
) -> Dict[str, float]:
    avg = {}
    for k in keys:
        v = dictionary[k]
        if len(v) > 0:
            avg_v = float(np.mean(v))
        else:
            avg_v = 0.0
        try:
            dist_avg_v, _ = weighted_average(avg_v, len(v))
        except ZeroDivisionError:
            logger.warning(f"Error when computing metric {k}: metric never appears")
            dist_avg_v = -1
        avg[k] = dist_avg_v
    return avg


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def fix_white_space(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    return fix_white_space(remove_articles(remove_punc(s.lower())))


def em(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def math_accuracy_score(prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]):
    return max([math_accuracy(prediction, gt, normalize_fn) for gt in ground_truths])


def code_pass_test(
    code: str,
    test: str,
    timeout: Optional[float] = 1.0,
) -> bool:
    actual_code = "\n".join([l for l in code.split("\n")])
    actual_test = "\n".join([l for l in test.split("\n")])
    script = f"{actual_code}\n\n{actual_test}"

    def unsafe_execute():
        execute(script, result, timeout)

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1 if timeout is not None else None)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"


def save_cur_data(
    data: List,
    dataset_name: str,
    dump_dir: str,
    world_rank: int,  # get_data_parallel_rank()
) -> None:
    assert isinstance(data, list), data
    if get_model_parallel_rank() != 0 or get_context_parallel_rank() != 0:
        return

    cur_save_dir = Path(dump_dir) / "eval_results" / dataset_name
    mkdir([cur_save_dir], world_rank == 0, exist_ok=True)

    cur_save_path = cur_save_dir / f"{world_rank}.jsonl"
    with open(cur_save_path, "a") as f:
        for x in data:
            x.pop("text_x", None)
            x.pop("text_y", None)
            x.pop("prompt", None)
            x.pop("prompted_input", None)
            x['raw'].pop('input', None)
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def gather_and_save_data(
    dataset_name: str,
    dump_dir: str,
    world_rank: int,
    world_size: int,
) -> None:
    assert 0 <= world_rank < world_size
    torch.distributed.barrier()
    if world_rank != 0 or get_model_parallel_rank() != 0 or get_context_parallel_rank() != 0:
        return
    save_dir = Path(dump_dir) / "eval_results"
    save_path = save_dir / f"{dataset_name}.jsonl"
    cur_save_dir = save_dir / dataset_name
    results_path = sorted(glob.glob(str(cur_save_dir / "*.jsonl")))
    if len(results_path) != world_size:
        raise RuntimeError(
            f"Expected {world_size} files, found {len(results_path)} "
            f"files in {cur_save_dir}"
        )
    with open(save_path, "w") as f:
        for path in results_path:
            with open(path, "r") as g:
                for line in g:
                    # check that each line is a json.dumps
                    f.write(json.dumps(json.loads(line), ensure_ascii=False))
                    f.write("\n")
    logger.info(f"Saved eval results in {save_path}.")
    shutil.rmtree(str(cur_save_dir))


def check_available_tasks(tasks: List[str]):
    available_tasks = set(BaseTask.tasks.keys())
    unavailable_tasks = set(tasks) - available_tasks
    if unavailable_tasks:
        raise ValueError(
            f"Could not import tasks {unavailable_tasks}.\n"
            f"The available tasks are {available_tasks}. "
            "Check the rest of the logs for more information and check that these "
            "tasks were imported in src/eval/task/__init__.py"
        )


class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return ClassPropertyDescriptor(func)
