import math
from typing import List, Dict, Any, Tuple
from timeit import default_timer as timer
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
import gc
import os
import json
import numpy as np
import torch
import torch.distributed.checkpoint as dist_ckpt

from xllm.config import (
    TokenizerConf,
    SlurmConf,
    ModelConf,
    OptimConf,
    TrainerConf,
    ValidConf,
    DataLoaderConfig,
)
from xllm.checkpointing import (
    get_latest_checkpoint_paths_for_sharded_states,
    get_base_model_checkpoint_path,
    get_last_restart_step,
    make_checkpointer,
)
from xllm.data.build import build_data_loader
from xllm.data.dataloader import MultiSourceDataLoader
from xllm.data.dataset_streamer.tokenizer import build_tokenizer
from xllm.distributed import (
    init_signal_handler,
    init_torch_distributed,
    initialize_model_parallel,
    get_data_parallel_group,
    get_hybrid_shard_data_parallel_group,
)
from xllm.distributed.utils import reduce_scalar
from xllm.distributed.slurm import get_global_rank
from xllm.logger import initialize_logger, add_logger_file_handler
from xllm.cluster_check import check_cluster
from xllm.utils import (
    mkdir,
    setup_env,
    log_host,
    clip_grad_norm_,
    log_restart_infos,
    set_random_seed,
    num_parameters,
    check_random_for_sync,
    check_batch_for_sync,
    get_model_state,
    get_sharded_optimizer_state,
    set_model_state,
    set_sharded_optimizer_state,
    collect_provenance,
)
from xllm.models import build_model
from xllm.optim import build_optimizer, rescale_grads
from xllm.monitor import GPUMonitor, Timings
from xllm.metrics import MetricLogger, build_mlogger
from xllm.configuration import cfg_from_cli
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

logger = getLogger()


@dataclass
class State:
    step: int
    scale: float
    scale_updates: int
    clip_cumulative: int


def manual_seed(cfg: TrainerConf):
    logger.info(
        f"Initializing torch seed to {cfg.seed}"
    )
    set_random_seed(cfg.seed)


def initialize_run(cfg: TrainerConf):
    global_rank = get_global_rank()
    # init dump dir / dump parameters / logger file handler
    assert cfg.dump_dir != "", "Please specify dump dir."
    dump_dir = Path(cfg.dump_dir)
    mkdir([dump_dir], global_rank == 0, exist_ok=True)

    # log restart infos before doing anything
    restart_infos = log_restart_infos(
        dump_dir=cfg.dump_dir,
        global_rank=global_rank,
        step=get_last_restart_step(cfg.dump_dir, global_rank),
    )
    logger.warning(f"restart infos: {restart_infos}")

    # initialize signal handler
    init_signal_handler()

    # initialize distributed mode / model parallel
    logger.info("Starting init of torch.distributed...")
    slurm_cfg = init_torch_distributed(cfg.nccl_timeout)
    cfg.slurm.set_values(*slurm_cfg)
    logger.info("Done init of torch.distributed.")

    logger.info("Starting init of model parallel...")
    initialize_model_parallel(cfg.model_parallel_size, cfg.context_parallel_size, cfg.nccl_timeout)
    logger.info("Done init of model parallel.")

    logger.info(
        f"Global rank: {cfg.slurm.global_rank} -- "
        f"model   parallel rank: {cfg.model_parallel_rank}/{cfg.model_parallel_size} -- "
        f"context parallel rank: {cfg.context_parallel_rank}/{cfg.context_parallel_size} -- "
        f"data    parallel rank: {cfg.data_parallel_rank}/{cfg.data_parallel_size}"
    )

    if cfg.slurm.is_master:
        with open(Path(cfg.dump_dir) / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f, sort_keys=True, indent=4)
        add_logger_file_handler(os.path.join(cfg.dump_dir, "train.log"))

    # print env info
    if cfg.slurm.is_slurm_job:
        secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        safe_env = {
            k: ("<redacted>" if any(marker in k.upper() for marker in secret_markers) else v)
            for k, v in os.environ.items()
        }
        logger.info(f"ENV: {safe_env}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"NCCL version: {torch.cuda.nccl.version()}")  # type: ignore


def set_batch_size(cfg: TrainerConf):
    if cfg.global_batch_size is None:
        assert cfg.batch_size is not None
        cfg.global_batch_size = cfg.data_parallel_size * cfg.batch_size
    else:
        assert cfg.batch_size is None
        assert cfg.global_batch_size % cfg.data_parallel_size == 0, \
            f"global batch size ({cfg.global_batch_size}) is not divisible by data parallel size ({cfg.data_parallel_size})."
        cfg.batch_size = cfg.global_batch_size // cfg.data_parallel_size


def get_optim_state(
    model,
    optimizer,
    dcp,
    scheduler,
    state: State,
    data_iterator: MultiSourceDataLoader,
    timings: Timings
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    optim_state_dict = get_sharded_optimizer_state(model, optimizer, dcp)
    training_state_dict = {
        "scheduler": scheduler.state_dict(),
        "step": state.step,
        "scale": state.scale,
        "scale_updates": state.scale_updates,
        "clip_cumulative": state.clip_cumulative,
        "data_state": data_iterator.get_state(),
        "timings": timings.to_dict(),
    }
    return optim_state_dict, training_state_dict


def set_optim_state(
    scheduler,
    reloaded_training_state_dict,
    state: State,
    data_iterator: MultiSourceDataLoader,
    timings: Timings,
    opt_cfg: OptimConf
):
    # setting LR manually to support overwriting during training from the same state
    reloaded_training_state_dict["scheduler"]["base_lrs"] = [opt_cfg.lr]
    scheduler.load_state_dict(reloaded_training_state_dict["scheduler"])
    state.step, state.scale, state.scale_updates, state.clip_cumulative = (
        reloaded_training_state_dict["step"],
        reloaded_training_state_dict["scale"],
        reloaded_training_state_dict["scale_updates"],
        reloaded_training_state_dict["clip_cumulative"],
    )
    timings.load_state_dict(reloaded_training_state_dict.get("timings", {}))  # ok if timings not here
    data_iterator.set_state(reloaded_training_state_dict["data_state"])
    logger.info(
        f"Reloaded training state. "
        f"Last step: {state.step} - Last scale: {state.scale} - Last clip_cumulative: {state.clip_cumulative}"
    )
    if scheduler._step_count - 1 != state.step:
        raise RuntimeError(f"Step mismatch: {scheduler._step_count - 1}/{state.step}")


def main(cfg: TrainerConf):
    gc.disable()
    t_start = timer()

    # initialize distributed mode / signal handler / model parallel
    initialize_run(cfg)

    # setup wandb
    if has_wandb and cfg.slurm.is_master and cfg.log_wandb:
        # wandb.require(experiment="service")
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                   name=os.path.basename(cfg.dump_dir),
                   config={
                       "training": cfg.to_dict(),
                       "provenance": collect_provenance(cfg),
                   })

    # random seed & enable tf32
    manual_seed(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True

    # setup batch size
    set_batch_size(cfg)

    # tokenizer / data iterator
    tokenizer = build_tokenizer(cfg.tokenizer)
    data_iterator: MultiSourceDataLoader = build_data_loader(tokenizer, cfg)

    # only process 0 prints
    if cfg.slurm.global_rank > 0 and cfg.disable_workers_print:
        logger.info(f"No print for worker {cfg.slurm.global_rank}")
        logger.disabled = True

    # model / optimizer paths
    model_path, optim_path, training_state_path = get_latest_checkpoint_paths_for_sharded_states(
        cfg.dump_dir, cfg.slurm.global_rank, cfg.fully_sharded_size, cfg.dcp_for_optimizer
    )
    assert (model_path is None) == (optim_path is None)

    logger.info("Start building of model...")
    cfg.model.vocab_size = tokenizer.vocab_size
    model = build_model(
        cfg.model, dtype=cfg.dtype,
        fully_sharded_size=cfg.fully_sharded_size,
        fp32_reduce_scatter=cfg.fp32_reduce_scatter,
        reshard_after_forward=cfg.reshard_after_forward,
        forward_prefetch=cfg.forward_prefetch, tokenizer=tokenizer
    )
    torch.cuda.empty_cache()
    gc.collect(2)
    logger.info(model)
    # logging number parameters
    logger.info("Computing total number of parameters...")
    num_parameters(model, cfg)
    total_params, activated_params, embed_params = model.num_parameters()
    logger.info(f"Total num. parameters: {total_params}.")
    logger.info(f"Activated  parameters: {activated_params}.")
    logger.info(f"Embedding  parameters: {embed_params}.")
    # dump dir
    logger.info(f"Experiment directory: {cfg.dump_dir}")
    tflops_per_token = model.tflops_per_token(cfg.seq_len)

    # build optimizer / scheduler
    optimizer, scheduler = build_optimizer(model, cfg.optim, cfg.steps, cfg.dtype)

    # setup loggers
    enable_loggers = cfg.slurm.is_master and not cfg.disable_logging
    enable_wandb = cfg.slurm.is_master and cfg.log_wandb and has_wandb
    to_log = ["train", "optim", "timings", "cluster_checks"]
    if cfg.async_eval_ngpus < 1:
        to_log.append("eval")

    mloggers: Dict[str, MetricLogger] = {
        k: build_mlogger(cfg.dump_dir, k, enable_loggers, enable_wandb) for k in to_log
    }

    model.train()
    state = State(step=0, scale=1.0, scale_updates=0, clip_cumulative=0)
    gpu_monitor = GPUMonitor()
    timings = Timings()
    checkpointer = make_checkpointer(
        dump_dir=cfg.dump_dir,
        global_rank=cfg.slurm.global_rank,
        world_size=cfg.slurm.world_size,
        keep_last=cfg.keep_n_last_checkpoints,
        keep_checkpoint_every_step=cfg.eval_freq if cfg.keep_eval_checkpoints else -1,
        cfg=cfg,
        async_checkpointing=cfg.async_checkpointing,
    )

    # reload model if available
    if model_path is not None:
        logger.info(f"Reloading model checkpoint from {model_path} ...")
        hsdp_group = get_hybrid_shard_data_parallel_group()
        # reload sharded model state
        reloaded_model_state_dict = get_model_state(model, full_state=False)
        dist_ckpt.load(reloaded_model_state_dict, checkpoint_id=model_path, process_group=hsdp_group)
        set_model_state(model, reloaded_model_state_dict, full_state=False)
        del reloaded_model_state_dict
        logger.info("Reloaded model.")

        # reload optimizer if available
        assert optim_path is not None
        logger.info(f"Reloading optimizer checkpoint from {optim_path} ...")
        if cfg.dcp_for_optimizer:
            reloaded_optim_state_dict = get_sharded_optimizer_state(model, optimizer, cfg.dcp_for_optimizer)
            dist_ckpt.load(reloaded_optim_state_dict, checkpoint_id=optim_path, process_group=hsdp_group)
        else:
            reloaded_optim_state_dict = torch.load(optim_path, map_location="cpu", weights_only=False)
        set_sharded_optimizer_state(model, optimizer, reloaded_optim_state_dict, cfg.dcp_for_optimizer)
        del reloaded_optim_state_dict
        logger.info("Reloaded optimizer.")

        assert training_state_path is not None
        logger.info(f"Reloading training state checkpoint from {training_state_path} ...")
        reloaded_training_state_dict = torch.load(training_state_path, map_location="cpu", weights_only=False)
        set_optim_state(scheduler, reloaded_training_state_dict, state, data_iterator, timings, cfg.optim)
    elif cfg.base_model_dir is not None:
        base_model_path, base_optim_path = get_base_model_checkpoint_path(
            cfg.base_model_dir, cfg.slurm.global_rank, cfg.fully_sharded_size, cfg.dcp_for_optimizer
        )
        logger.info(f"Reloading base model from {base_model_path} ...")
        hsdp_group = get_hybrid_shard_data_parallel_group()
        # reload base model state
        reloaded_model_state_dict = get_model_state(model, full_state=False)
        dist_ckpt.load(reloaded_model_state_dict, checkpoint_id=base_model_path, process_group=hsdp_group)
        set_model_state(model, reloaded_model_state_dict, full_state=False)
        del reloaded_model_state_dict
        logger.info("Reloaded base model.")

        if cfg.reload_base_optim_state:
            assert base_optim_path is not None, base_optim_path
            logger.info(f"Reloading base optimizer checkpoint from {base_optim_path} ...")
            if cfg.dcp_for_optimizer:
                reloaded_optim_state_dict = get_sharded_optimizer_state(model, optimizer, cfg.dcp_for_optimizer)
                dist_ckpt.load(reloaded_optim_state_dict, checkpoint_id=base_optim_path, process_group=hsdp_group)
            else:
                reloaded_optim_state_dict = torch.load(base_optim_path, map_location="cpu", weights_only=True)
            set_sharded_optimizer_state(model, optimizer, reloaded_optim_state_dict, cfg.dcp_for_optimizer)
            del reloaded_optim_state_dict

    # check gpu nodes
    check_cluster(
        dump_dir=Path(cfg.dump_dir), global_rank=cfg.slurm.global_rank, check_level=cfg.cluster_check_level, step=0
    )

    # build batch iterator
    logger.info("Building batch iterator ...")
    batch_iterator = iter(data_iterator)

    logger.info("Checking torch randomness...")
    check_random_for_sync(1024)
    logger.info(f"Pass torch randomness checking.")

    logger.info(
        f"Re-initializing CUDA seed to {cfg.seed} + {cfg.slurm.global_rank}"
    )
    seed = cfg.seed + cfg.slurm.global_rank
    torch.cuda.manual_seed(seed)

    # training starts
    logger.info(f"Training starts: batch size={cfg.global_batch_size} ({cfg.batch_size}) ...")

    timings.starting += timer() - t_start

    t0, last_nw = timer(), 0
    losses: List[float] = []
    aux_losses: List[float] = [] if cfg.moe_router_load_balancing_type is not None else None
    grad_norms: List[float] = []
    padding_ratios: List[float] = []
    truncation_ratios: List[float] = []
    gc.collect()
    torch.cuda.empty_cache()

    while state.step < cfg.steps:
        torch.cuda.reset_peak_memory_stats()
        # data loading
        t1 = timer()
        batch = next(batch_iterator)
        x = torch.from_numpy(batch.x).cuda()
        y = torch.from_numpy(batch.y).cuda()
        mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda()
        if cfg.dataloader.packing_type == "bestfit":
            padding_ratios.append(batch.padding_ratio)
            truncation_ratios.append(batch.truncation_ratio)

        if cfg.sync_check_freq > 0 and state.step % cfg.sync_check_freq == 0:
            logger.info("Checking batch randomness at step: {}, ...".format(state.step))
            check_batch_for_sync({"x": x, "y": y, "mask": mask})
            logger.info(f"Pass batch randomness checking.")

        # fwd-bwd
        s_fwd_bwd = timer()
        timings.data_loading += s_fwd_bwd - t1

        last_nw += y.nelement()

        tok_loss, aux_loss, _ = model(
            tokens=x, targets=y, token_mask=mask, multi_segments=cfg.multi_segments,
            moe_router_load_balancing_type=cfg.moe_router_load_balancing_type,
            attn_stability_control=cfg.attn_stability_control, deterministic=cfg.deterministic
        )

        if mask is None:
            num_tokens = torch.tensor(tok_loss.numel(), dtype=torch.int64, device=tok_loss.device)
        else:
            num_tokens = mask.to(torch.int64).sum()
        # reduce num tokens across data parallel ranks
        torch.distributed.all_reduce(num_tokens, group=get_data_parallel_group())
        num_tokens_per_rank = num_tokens / (cfg.data_parallel_size * cfg.context_parallel_size)
        train_loss = tok_loss.sum() / (num_tokens_per_rank + 1e-6).to(tok_loss)
        # calculate loss for logging
        loss = train_loss.detach() / cfg.context_parallel_size

        if aux_loss is not None:
            num_aux_loss = model.num_layers - model.num_dense_layers
            aux_loss = aux_loss / num_aux_loss
            train_loss = train_loss + cfg.moe_aux_loss_coeff * aux_loss

        if cfg.loss_rescaling:
            train_loss = state.scale * train_loss

        train_loss.backward()

        # grad clip
        s_clip = timer()
        timings.forward_backward += s_clip - s_fwd_bwd

        clip_max_norm = cfg.optim.clip * state.scale
        grad_norm = clip_grad_norm_(fsdp_module=model, max_norm=clip_max_norm)

        # update scale / status
        overflow = False
        if cfg.loss_rescaling:
            # detect inf and nan
            if grad_norm > 500. or grad_norm != grad_norm:
                state.scale /= 2
                state.scale_updates += 1
                if state.scale < 0.1:
                    raise FloatingPointError((
                        'Minimum loss scale reached ({}). Your loss is probably exploding. '
                        'Try lowering the learning rate, using gradient clipping or increasing the batch size.'
                    ).format(state.scale))

                overflow = True
            elif grad_norm < 0.05 and state.scale < 4096:
                state.scale *= 2
                state.scale_updates += 1
        else:
            assert state.scale == 1.0

        if overflow:
            logger.warning('Overflow detected, setting loss scale to: ' + str(state.scale))
            continue

        # update step
        state.step += 1

        if grad_norm > clip_max_norm:
            state.clip_cumulative += 1

        # optim step
        s_optimize = timer()
        timings.grad_clip += s_optimize - s_clip

        rescale_grads(model, scale=state.scale)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # logging
        s_logging = timer()
        timings.optimize += s_logging - s_optimize

        curr_loss = loss.item()
        losses.append(curr_loss)
        if aux_loss is not None:
            aux_loss = aux_loss.item()
            aux_losses.append(aux_loss)
        grad_norms.append(grad_norm)
        curr_lr = float(optimizer.param_groups[0]["lr"])
        assert math.isfinite(curr_loss), f"loss is not finite:{curr_loss}"

        n_tokens = float(cfg.global_batch_size) * cfg.seq_len / 1e9 * state.step

        optim_metrics = {
            "step": state.step,
            "lr": curr_lr,
            "avg_loss": reduce_scalar(curr_loss, op='mean'),
            "max_loss": reduce_scalar(curr_loss, op='max'),
        }
        if aux_loss is not None:
            optim_metrics.update({
                "avg_aux_loss": reduce_scalar(aux_loss, op='mean'),
                "max_aux_loss": reduce_scalar(aux_loss, op='max'),
            })
        optim_metrics.update({
            "g_norm": grad_norm / state.scale,
            "clip_cumulative": state.clip_cumulative,
            "n_tokens": n_tokens,
        })
        if cfg.loss_rescaling:
            optim_metrics.update({
                "scale": state.scale,
                "scale_updates": state.scale_updates,
            })

        mloggers["optim"].log(optim_metrics, state.step)

        if state.step % cfg.gc_collect_freq == 0:
            gc.collect()

        if state.step % cfg.log_freq == 0:
            delta = timer() - t0

            gpu_info = gpu_monitor.get_stats()
            agg_gpu_info = gpu_monitor.aggregate_gpu_info(gpu_info)
            avg_loss, max_loss = np.mean(losses).item(), max(losses)
            if aux_losses is not None and len(aux_losses) > 0:
                avg_aux_loss, max_aux_loss = np.mean(aux_losses).item(), max(aux_losses)
            else:
                avg_aux_loss, max_aux_loss = None, None
            avg_gnorm, max_gnorm = np.mean(grad_norms).item(), max(grad_norms)
            wps = last_nw * cfg.data_parallel_size / cfg.slurm.world_size / delta
            throughput = tflops_per_token * wps

            # agg across gpus
            global_avg_loss = reduce_scalar(avg_loss, op='mean')
            global_max_loss = reduce_scalar(max_loss, op='max')
            avg_ppl = math.exp(global_avg_loss)
            if avg_aux_loss is not None:
                global_avg_aux_loss = reduce_scalar(avg_aux_loss, op='mean')
                global_max_aux_loss = reduce_scalar(max_aux_loss, op='max')
            else:
                global_avg_aux_loss = None
                global_max_aux_loss = None

            aux_info = f"aux loss: {avg_aux_loss:.3f} ({max_aux_loss:.3f}), " if avg_aux_loss is not None else ""
            global_aux_info = f"aux loss: {global_avg_aux_loss:.3f} ({global_max_aux_loss:.3f}), " if avg_aux_loss is not None else ""
            scale_info = f"scale: {state.scale}, " if cfg.loss_rescaling else ""
            logging_info = (
                f"step: {state.step:7d} ({100 * state.step / cfg.steps:.1f}%), "
                f"lr: {curr_lr:.2e}, "
                f"batch size: {cfg.global_batch_size} ({cfg.batch_size}), "
                f"tokens: {n_tokens:.1f}B, "
                f"loss: {avg_loss:.3f} ({max_loss:.3f}), "
                f"{aux_info}"
                f"gnorm: {avg_gnorm:.3f} ({max_gnorm:.3f}), "
                f"gpu usage: {gpu_info['gpu_usage']:5.1f}%, "
                f"gpu mem: {gpu_info['used_memory_gb']:.1f}GB, "
                f"{scale_info}"
                f"t: {delta:.2f}s, "
                f"wps: {wps:.0f}, "
                f"throughput: {throughput:.1f},"
                f" || "
                f"llm loss: {global_avg_loss:.3f} ({global_max_loss:.3f}), "
                f"ppl: {avg_ppl:.2f}, "
                f"{global_aux_info}"
                f"mem_retries: {int(agg_gpu_info['gpus/num_alloc_retries_MAX'])}"
            )
            if cfg.dataloader.packing_type == "bestfit":
                padding_ratio = reduce_scalar(np.mean(padding_ratios).item(), op='mean')
                truncation_ratio = reduce_scalar(np.mean(truncation_ratios).item(), op='mean')
                logging_info += (
                    f", padding ratio: {padding_ratio * 100:.3f}% ({cfg.seq_len}), "
                    f"truncation ratio: {truncation_ratio * 100:.3f}% ({cfg.global_batch_size})"
                )
            else:
                padding_ratio, truncation_ratio = None, None
            logger.info(logging_info)

            metrics = {
                "step": state.step,
                "wps": wps,
                "throughput": throughput,
                "loss": global_avg_loss,
                "loss (max)": global_max_loss,
                "ppl": avg_ppl,
                "gnorm": avg_gnorm,
                "gnorm (max)": max_gnorm,
            }
            if cfg.dataloader.packing_type == "bestfit":
                metrics.update({
                    "padding ratio": padding_ratio,
                    "truncation ratio": truncation_ratio,
                })
            if global_avg_aux_loss is not None:
                metrics.update({
                    "aux loss": global_avg_aux_loss,
                    "aux loss (max)": global_max_aux_loss,
                })
            mloggers["train"].log(metrics, state.step)
            mloggers["cluster_checks"].log(agg_gpu_info, state.step)
            timings.logging += timer() - s_logging
            s_logging = timer()  # update logging time
            mloggers["timings"].log(timings.get_stats(), state.step)

            checkpointer.check_ok()  # check if background checkpointing is ok

            last_nw = 0
            losses.clear()
            if aux_losses is not None:
                aux_losses.clear()
            grad_norms.clear()
            padding_ratios.clear()
            truncation_ratios.clear()
            t0 = timer()

        # checkpoint
        s_checkpointing = timer()
        timings.logging += s_checkpointing - s_logging

        should_checkpoint = state.step % cfg.dump_freq == 0 or state.step == cfg.steps
        should_eval = cfg.eval_freq > 0 and (state.step % cfg.eval_freq == 0 or state.step == cfg.steps)

        if should_checkpoint:
            gc.collect()
            torch.cuda.empty_cache()
            sharded_model_state = get_model_state(model, full_state=False)
            # import pdb; pdb.set_trace()
            optim_state, training_state = get_optim_state(
                model, optimizer, cfg.dcp_for_optimizer, scheduler, state, data_iterator, timings
            )

            checkpointer.save_latest_checkpoint(
                model_state=sharded_model_state,
                optimizer_state=optim_state,
                training_state=training_state,
                step=state.step,
            )
            t0 = timer()

        # evaluation
        s_eval = timer()
        timings.checkpointing += s_eval - s_checkpointing

        if should_eval:
            checkpointer.wait_for_all()
            if cfg.async_eval_ngpus > 0:
                if cfg.slurm.is_master:
                    from xllm.eval.launch_eval import launch_async_eval
                    launch_async_eval(cfg=cfg, step=state.step)
            else:
                from xllm.eval.launch_eval import launch_sync_eval
                model.eval()
                scores = launch_sync_eval(cfg, model, tokenizer, step=state.step)
                logger.info(f"ALL RESULTS: {scores}")
                mloggers["eval"].log(scores, state.step)
                model.train()

            t0 = timer()

        timings.eval += timer() - s_eval

    # end of training
    logger.info(f"Reached {cfg.steps} steps.")
    data_iterator.close()
    checkpointer.close()
    for mlogger in mloggers.values():
        mlogger.close()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":

    # initialize logger / setup environment
    initialize_logger()
    logger.info("Done setting logger.")
    setup_env()
    log_host()
    logger.info("Done setting env.")

    cfg = TrainerConf(
        tokenizer=TokenizerConf(),
        slurm=SlurmConf(),
        model=ModelConf(),
        optim=OptimConf(),
        valid=ValidConf(),
        dataloader=DataLoaderConfig(),
    )
    cfg = cfg_from_cli(cfg)
    main(cfg)
