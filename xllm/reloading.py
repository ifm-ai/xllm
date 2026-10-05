import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.distributed.checkpoint as dist_ckpt

from xllm.config import ModelConf, TokenizerConf, logger
from xllm.checkpointing import (
    sharded_model_dir,
    full_model_dir,
)
from xllm.data.dataset_streamer.tokenizer import Tokenizer, build_tokenizer
from xllm.models import build_model
from xllm.distributed import (
    FullyShardedDataParallel,
    init_signal_handler,
    init_torch_distributed,
    get_data_parallel_world_size,
    get_model_parallel_world_size,
    get_context_parallel_world_size,
    get_hybrid_shard_data_parallel_group,
    initialize_model_parallel,
)
from xllm.utils import (
    get_default_inference_half,
    get_parallel_ranks,
    get_model_state,
    set_model_state
)


@dataclass
class ReloadedConf:
    world_size: int
    dp_world_size: int
    mp_world_size: int
    cp_world_size: int
    dtype: str
    cfg: Dict[str, Any]

    def new_mp_world_size(self, model_parallel_size: Optional[int]):
        return (
            model_parallel_size
            if model_parallel_size is not None
            else self.mp_world_size
        )

    def new_cp_world_size(self, context_parallel_size: Optional[int]):
        return (
            context_parallel_size
            if context_parallel_size is not None
            else self.cp_world_size
        )

    def __post_init__(self):
        assert self.dtype == "fp32"


def reload_config(ckpt_dir: Path) -> ReloadedConf:
    cfg_path = ckpt_dir / "config.json"
    with cfg_path.open("r") as fp:
        cfg = json.load(fp)
    old_mp = cfg["model_parallel_size"]
    old_cp = cfg["context_parallel_size"]
    old_world_size = cfg["slurm"]["world_size"]
    assert 0 < old_mp <= old_world_size
    assert old_world_size % (old_mp * old_cp) == 0
    old_ddp = old_world_size // old_mp // old_cp
    old_dtype = "fp32"  # FSDP training in mixed precision

    return ReloadedConf(
        world_size=old_world_size,
        dp_world_size=old_ddp,
        mp_world_size=old_mp,
        cp_world_size=old_cp,
        cfg=cfg,
        dtype=old_dtype,
    )


def reload_config_and_tokenizer(ckpt_dir: Path, tokenizer_path: Optional[str] = None) -> Tuple[ReloadedConf, Tokenizer, ModelConf]:
    reloaded = reload_config(ckpt_dir)
    cfg = reloaded.cfg

    tokenizer_cfg: TokenizerConf = TokenizerConf.from_dict(cfg["tokenizer"])
    old_tokenizer_path = tokenizer_cfg.path
    new_tokenizer_path: str = tokenizer_path if tokenizer_path is not None else old_tokenizer_path
    assert Path(new_tokenizer_path).exists(), new_tokenizer_path
    tokenizer_cfg.path = new_tokenizer_path

    tokenizer = build_tokenizer(cfg=tokenizer_cfg)
    model_cfg: ModelConf = ModelConf.from_dict(cfg["model"])
    model_cfg.fused_block = False
    model_cfg.init_mode = 'none'
    # old ckpt don't have vocab_size set
    if model_cfg.vocab_size == -1:
        model_cfg.vocab_size = tokenizer.vocab_size
    assert model_cfg.vocab_size == tokenizer.vocab_size, (
        tokenizer.vocab_size,
        model_cfg.vocab_size,
    )
    return reloaded, tokenizer, model_cfg


def init_distributed_mode(model_parallel_size: int, context_parallel_size: int, timeout: int):
    # initialize signal handler
    init_signal_handler()
    # initialize distributed mode / model parallel
    logger.info("Starting init of torch.distributed...")
    is_slurm, global_rank, world_size = init_torch_distributed(timeout)
    logger.info("Done init of torch.distributed.")

    logger.info("Starting init of model parallel...")
    initialize_model_parallel(model_parallel_size, context_parallel_size, timeout=timeout)
    logger.info("Done init of model parallel.")

    # print env info
    if is_slurm:
        logger.info(f"ENV: {os.environ}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"NCCL version: {torch.cuda.nccl.version()}")  # type: ignore

    return global_rank, world_size


def reload_model(
    checkpoint_dir: str,
    full_state: bool,
    model_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None,
    dtype: str = get_default_inference_half(),
    tokenizer_path: Optional[str] = None,
    timeout: int = 1800,
) -> Tuple[nn.Module, Tokenizer, ReloadedConf]:
    ckpt_dir: Path = Path(checkpoint_dir)

    reloaded, tokenizer, model_cfg = reload_config_and_tokenizer(ckpt_dir, tokenizer_path=tokenizer_path)
    new_mp = reloaded.new_mp_world_size(model_parallel_size)
    new_cp = reloaded.new_cp_world_size(context_parallel_size)

    global_rank, world_size = init_distributed_mode(new_mp, new_cp, timeout)

    assert new_mp == get_model_parallel_world_size(), f"{new_mp} != {get_model_parallel_world_size()}"
    assert new_cp == get_context_parallel_world_size(), f"{new_cp} != {get_context_parallel_world_size()}"

    if new_mp == reloaded.mp_world_size:
        logger.info(f"Reloading FSDP model from sharded state -- Path={ckpt_dir} -- MP={new_mp}")
        model = build_model(
            model_cfg, dtype=dtype,
            fully_sharded_size=None,
            fp32_reduce_scatter=(dtype != 'fp32'),
            reshard_after_forward=True,
            forward_prefetch=False,
            tokenizer=tokenizer
        )
        reload_fsdp_from_state(ckpt_dir, full_state, model, reloaded, global_rank)
    else:
        raise NotImplementedError
    return model, tokenizer, reloaded


def reload_fsdp_from_state(
    ckpt_dir: Path,
    full_state: bool,
    model: FullyShardedDataParallel,
    reloaded: ReloadedConf,
    global_rank: int,
):
    dp_rank, _, mp_rank = get_parallel_ranks(global_rank)
    ddp_world_size = get_data_parallel_world_size()
    cp_world_size = get_context_parallel_world_size()
    mp_world_size = get_model_parallel_world_size()

    old_ddp = reloaded.dp_world_size
    old_mp = reloaded.mp_world_size
    old_cp = reloaded.cp_world_size

    assert mp_world_size == old_mp

    logger.info(f"Starting reloading from shards ({old_ddp}, {old_cp}, {old_mp}) -> ({ddp_world_size}, {cp_world_size}, {mp_world_size})")

    model_path = ckpt_dir / (full_model_dir(mp_rank) if full_state else sharded_model_dir(mp_rank))
    assert model_path.exists(), f"Checkpoint for model shard {global_rank} doesn't exists: {model_path}"

    logger.info(f"Reloading model checkpoint from {model_path} ...")
    cpu_offload = not full_state
    hsdp_group = get_hybrid_shard_data_parallel_group()
    reloaded_model_state_dict = get_model_state(model, full_state=full_state, cpu_offload=cpu_offload)
    dist_ckpt.load(reloaded_model_state_dict, checkpoint_id=model_path, process_group=hsdp_group)
    set_model_state(model, reloaded_model_state_dict, full_state=full_state, cpu_offload=cpu_offload)
    del reloaded_model_state_dict
    logger.info("Reloaded model.")
