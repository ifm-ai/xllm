from collections import defaultdict
from logging import getLogger
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import math
import pickle

import torch
import torch.nn as nn

from .task import get_task
from .task.human_eval import HumanEvalTask
from .task.mbpp import MBPPTask
from .dataloader import EvalTaskDataloader
from .task.base import ChoiceTask, GenerationTask
from .task.mmlu import get_mmlu_scores
from .task.arabic_mmlu import get_arabic_mmlu_scores
from .utils import (
    check_available_tasks,
    avg_dist_dict,
    save_cur_data,
    gather_and_save_data,
)
from xllm.config import ValidConf
from xllm.data.dataloader import PPLDataLoader
from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.generation import generate
from xllm.inference import inference
from xllm.utils import check_batch_for_sync, set_random_seed
from xllm.distributed.utils import reduce_scalar
from xllm.distributed import (
    get_context_parallel_world_size,
    get_data_parallel_group,
)

logger = getLogger()


@torch.no_grad()
def eval_ppl(
    model: nn.Module,
    tokenizer: Tokenizer,
    jsonl_path: str,
    cfg: ValidConf,
    multi_segments: bool,
    world_rank,
    world_size,
) -> Tuple[float, int]:

    model.eval()
    data_iterator = PPLDataLoader(
        tokenizer=tokenizer,
        data=jsonl_path,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        world_rank=world_rank,
        world_size=world_size,
        keep_tail=True,
    )

    if cfg.n_batches > 0:
        batches = []
        for data in iter(data_iterator):
            batches.append(data)
            if len(batches) == cfg.n_batches:
                break
    else:
        batches = list(iter(data_iterator))

    data_iterator.close()

    dummy_start = len(batches)
    len_data = reduce_scalar(len(batches), op="max")
    extra = int(len_data) - len(batches)
    if extra > 0:
        dummy_batch = batches[0] if len(batches) > 0 else data_iterator.dummy_batch(8)
        batches.extend([dummy_batch for _ in range(extra)])

    metric = 0.0
    n_toks = 0
    check_sync_freq = max(len_data // 10, 1)

    for i, batch in enumerate(batches):
        if i >= cfg.n_batches > 0:
            break

        x = torch.from_numpy(batch.x).cuda()
        y = torch.from_numpy(batch.y).cuda()
        mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda()
        if i % check_sync_freq == 0:
            check_batch_for_sync({"x": x, "y": y, "mask": mask})

        tok_loss = inference(model, x, y, multi_segments)
        if mask is not None:
            tok_loss = tok_loss * mask
            loss = tok_loss.sum()
            num_tokens = mask.sum()
        else:
            loss = tok_loss.sum()
            num_tokens = y.nelement()

        if i < dummy_start:
            metric += loss.item()
            n_toks += num_tokens

    tot_loss = reduce_scalar(metric, op="sum", group=get_data_parallel_group())
    tot_toks = reduce_scalar(n_toks, op="sum", group=get_data_parallel_group())
    return math.exp(tot_loss / tot_toks), tot_toks


def eval_choice(model: nn.Module, batch: Dict, task: ChoiceTask):
    assert isinstance(task, ChoiceTask)
    batch_metrics: List[Dict[str, float]] = []
    pad_tensor = torch.zeros((1, 1), dtype=torch.long)

    text_x = batch.get("text_x", pad_tensor).cuda()
    text_y = batch.get("text_y", pad_tensor).cuda()
    text_nll = inference(model, text_x, text_y, False)
    text_nll = text_nll.view(text_x.size()).sum(dim=-1)
    text_ntokens = (text_y != -100).sum(dim=-1)

    completion_x = batch.get("completion_x", pad_tensor).cuda()
    completion_y = batch.get("completion_y", pad_tensor).cuda()
    completion_nll = inference(model, completion_x, completion_y, False)
    completion_nll = completion_nll.view(completion_x.size()).sum(dim=-1)

    if "text_x" not in batch:  # padding example
        return batch_metrics

    for idx, ex in enumerate(batch["examples"]):
        start_idx = batch["completion_index"][idx]
        end_idx = batch["completion_index"][idx + 1]
        sample_metrics = task.evaluate(
            text_nll[start_idx:end_idx],
            ex["raw"],
            text_ntokens[start_idx:end_idx],
            batch["completion_text"][idx],
            completion_nll[start_idx:end_idx],
        )
        batch_metrics.append(sample_metrics)
        batch["examples"][idx]["metrics"] = sample_metrics
    return batch_metrics


def eval_generation(
    cfg: ValidConf,
    model: nn.Module,
    batch: Dict,
    task: GenerationTask,
    tokenizer: Tokenizer,
):
    assert isinstance(task, GenerationTask)
    batch_metrics: List[Dict[str, float]] = []
    if "nll" in task.metrics:
        pad_tensor = torch.zeros((1, 1), dtype=torch.long)
        text_x = batch.get("text_x", pad_tensor).cuda()
        text_y = batch.get("text_y", pad_tensor).cuda()
        nll = inference(model, text_x, text_y, False)
        nll = nll.view(text_x.size()).sum(dim=-1)
    else:
        nll = None

    generation = generate(
        model,
        tokenizer,
        batch.get("prompt", None),
        max_prompt_len=task.max_text_len,
        max_gen_len=task.max_gen_len,
        use_sampling=cfg.use_sampling,
        temp=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
    )

    if "prompt" not in batch:
        return batch_metrics

    for idx, g in enumerate(generation):
        pred = task.postprocess(g)
        sample_metrics = task.evaluate(pred, batch["examples"][idx]["raw"])
        if "nll" in task.metrics:
            sample_metrics["nll"] = nll[idx].item()
        batch_metrics.append(sample_metrics)
        batch["examples"][idx]["generation"] = pred
        batch["examples"][idx]["metrics"] = sample_metrics

    return batch_metrics


@torch.no_grad()
def task_evaluation(
    model: nn.Module,
    tokenizer: Tokenizer,
    task_name: str,
    cfg: ValidConf,
    dump_dir: str,
    world_rank: int,
    world_size: int,
    seed: int,
):
    model.eval()
    metric_list = defaultdict(list)

    context_parallel_size = get_context_parallel_world_size()
    assert (
        context_parallel_size == 1
    ), f"task evaluation does not support context parallel: {context_parallel_size} > 1"

    task_name, task = get_task(cfg.task_root, task_name, tokenizer)
    task_dir = os.path.join(cfg.task_root, task_name)

    eval_path = os.path.join(task_dir, task.eval_file)
    dataloader = EvalTaskDataloader(
        path=eval_path,
        batch_size=cfg.batch_size,
        task=task,
        world_rank=world_rank,
        world_size=world_size,
        seed=seed,
        cfg=cfg,
    )
    batches = list(dataloader.batch_iterator())

    logger.info(
        f"Evaluating on {task_name} w. {len(batches)} batches of size {cfg.batch_size}..."
    )

    len_data = reduce_scalar(len(batches), op="max")
    extra = int(len_data) - len(batches)
    if extra > 0:
        batches.extend([{} for _ in range(extra)])

    check_sync_freq = max(len_data // 10, 1)
    for i, batch in enumerate(batches):
        if i % check_sync_freq == 0:
            check_batch_for_sync(batch)

        logger.debug(f"task {task_name} - batch #{i+1} / {len(batches)}")
        if isinstance(task, ChoiceTask):
            batch_metrics = eval_choice(model, batch, task)
        elif isinstance(task, GenerationTask):
            batch_metrics = eval_generation(
                cfg=cfg, model=model, batch=batch, task=task, tokenizer=tokenizer,
            )
        else:
            raise RuntimeError(f"Unknown task type: {task}")

        if cfg.save_eval:
            save_cur_data(
                data=batch.get("examples", []),
                dataset_name=f"{task_name}-{task.eval_file[:-6]}",
                dump_dir=dump_dir,
                world_rank=world_rank,
            )
        for m in batch_metrics:
            for key, value in m.items():
                metric_list[key].append(value)

    metrics = avg_dist_dict(task.metrics, metric_list)

    if cfg.save_eval:
        gather_and_save_data(
            dataset_name=f"{task_name}-{task.eval_file[:-6]}",
            dump_dir=dump_dir,
            world_rank=world_rank,
            world_size=world_size,
        )

    # pass@k for coding tasks
    if (
        cfg.save_eval
        and world_rank == 0
        and isinstance(task, (HumanEvalTask, MBPPTask))
    ):
        save_dir = Path(dump_dir) / "eval_results"
        save_path = save_dir / f"{task_name}-{task.eval_file}"
        assert save_path.exists()
        metrics.update(task.pass_k_acc(batch_paths=[save_path]))

    return metrics


def execute_evals(
    model: nn.Module,
    tokenizer: Tokenizer,
    cfg: ValidConf,
    multi_segments: bool,
    dump_dir: str,
    world_rank: int,
    world_size: int,
    is_master: bool,
    seed: Optional[int] = None,
):
    assert 0 <= world_rank < world_size
    if seed is not None:
        # eval mode
        torch_seed = seed + world_rank
        logger.info(f"Initializing torch seed to {torch_seed}")
        set_random_seed(torch_seed)

    model.eval()

    scores = {}
    # PPL evaluations
    logger.info("Running PPL evaluations ...")
    for path in cfg.ppl_files:
        logger.info(f"Evaluating PPL on {path} ...")
        ppl, n_tokens = eval_ppl(model, tokenizer, path, cfg, multi_segments, world_rank, world_size)
        logger.info(f"PPL on {path}: {ppl}, w. {n_tokens} tokens.")
        scores[f"ppl/{path}"] = ppl
        torch.cuda.empty_cache()

    # task evaluations
    logger.info("Running tasks evaluations ...")
    check_available_tasks(cfg.tasks)
    for task in cfg.tasks:
        fname = os.path.join(dump_dir, f"{task}.pkl")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if os.path.exists(fname):
            logger.info(
                f"Found {task} metrics file: {fname}, load metrics and skip evaluation."
            )
            # triggering creation of task instance as some static fields must be initialized for later use
            get_task(cfg.task_root, task_name=task, tokenizer=tokenizer)
            with open(fname, "rb") as fin:
                metrics = pickle.load(fin)
        else:
            metrics = task_evaluation(
                model=model,
                tokenizer=tokenizer,
                task_name=task,
                cfg=cfg,
                dump_dir=dump_dir,
                world_rank=world_rank,
                world_size=world_size,
                seed=seed if seed is not None else 42,  # default seed = 42
            )
            if is_master:
                with open(fname, "wb") as fout:
                    pickle.dump(metrics, fout)

        log = " - ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        logger.info(f"Results on {task}: {log}")
        for k, v in metrics.items():
            scores[f"task/{task}/{k}"] = v
        torch.cuda.empty_cache()

    # MMLU
    if "mmlu" in cfg.task_list.split(","):
        metrics = get_mmlu_scores(scores)
        log = " - ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        logger.info(f"Results on mmlu: {log}")
        for k, v in metrics.items():
            scores[f"task/mmlu/{k}"] = v

    if "arabic_mmlu" in cfg.task_list.split(","):
        metrics = get_arabic_mmlu_scores(scores)
        log = " - ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        logger.info(f"Results on arabic_mmlu: {log}")
        for k, v in metrics.items():
            scores[f"task/arabic_mmlu/{k}"] = v

    logger.info("===== Finished all evaluations.")
    return scores
