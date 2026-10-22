"""Time a restore from a DCP checkpoint on Lustre (option B's restore leg): save model + optimizer with xLLM's
SyncCheckpointer (DCP for both), drop everything, rebuild, and time dist_ckpt.load of model and optimizer."""
import os, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch, torch.distributed as dist
import torch.distributed.checkpoint as dcp
from xllm.checkpointing import SyncCheckpointer, get_checkpoint_dir, sharded_model_dir, sharded_optim_dir
from xllm.config import ModelConf, OptimConf
from xllm.distributed import initialize_model_parallel, get_hybrid_shard_data_parallel_group
from xllm.models import build_model
from xllm.optim import build_optimizer, rescale_grads
from xllm.utils import clip_grad_norm_, get_model_state, get_sharded_optimizer_state, set_model_state, set_sharded_optimizer_state
RANK, WORLD, LOCAL = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ["SLURM_LOCALID"])
VOCAB, SEQ = 32000, 2048
class Tok: vocab_size, bos_id, eos_id, pad_id = VOCAB, 1, 2, 0
def gmax(x):
    t = torch.tensor([float(x)], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.item()
def build():
    torch.manual_seed(0)
    m = build_model(ModelConf(arch="transformer", ddp_backend="fsdp2", init_mode="gaussian", init_std=0.02, num_layers=32,
        model_dim=4096, num_heads=32, head_dim=128, ffn_hidden_dim=14336, chunk_size=SEQ, vocab_size=VOCAB, apply_rmsnorm=True,
        qknorm=True, swiglu=True, rope_head_dim=128), dtype="bf16", fully_sharded_size=None, fp32_reduce_scatter=True,
        reshard_after_forward=True, forward_prefetch=False, tokenizer=Tok())
    o, s = build_optimizer(m, OptimConf(lr=1e-4, warmup=2, clip=1.0), 100, "bf16")
    return m, o, s
def step(m, o, s, i):
    g = torch.Generator(device="cuda").manual_seed(1000 * i + RANK)
    tok = torch.randint(3, VOCAB, (1, SEQ + 1), device="cuda", generator=g)
    loss, _, _ = m(tok[:, :-1], multi_segments=False, targets=tok[:, 1:]); loss.mean().backward()
    clip_grad_norm_(fsdp_module=m, max_norm=1.0); rescale_grads(m, 1.0); o.step(); s.step(); o.zero_grad()
def main():
    torch.cuda.set_device(LOCAL); dist.init_process_group("nccl", rank=RANK, world_size=WORLD); initialize_model_parallel(1, 1, 600)
    work = Path(sys.argv[1]); cfg = SimpleNamespace(dcp_for_optimizer=True, fully_sharded_size=None, to_json=lambda: "{}")
    m, o, s = build()
    for i in range(2): step(m, o, s, i)
    t = time.perf_counter()
    SyncCheckpointer(dump_dir=str(work), cfg=cfg, global_rank=RANK, world_size=WORLD).save_latest_checkpoint(
        model_state=get_model_state(m, full_state=False), optimizer_state=get_sharded_optimizer_state(m, o, True),
        training_state={"step": 2}, step=2)
    dist.barrier(); t_save = gmax(time.perf_counter() - t)
    del m, o, s; torch.cuda.empty_cache()
    for rep in range(2):
        os.system("sync")
        m, o, s = build(); dist.barrier(); t = time.perf_counter()
        ck = get_checkpoint_dir(work, 2); hsdp = get_hybrid_shard_data_parallel_group()
        ms = get_model_state(m, full_state=False); dcp.load(ms, checkpoint_id=ck / sharded_model_dir(0), process_group=hsdp)
        set_model_state(m, ms, full_state=False)
        os_ = get_sharded_optimizer_state(m, o, True); dcp.load(os_, checkpoint_id=ck / sharded_optim_dir(0), process_group=hsdp)
        set_sharded_optimizer_state(m, o, os_, True); torch.cuda.synchronize(); dist.barrier()
        t_load = gmax(time.perf_counter() - t)
        if RANK == 0:
            print(f"DISKRESTORE size=8b world={WORLD} rep={rep} save={t_save:.1f}s load_model+optim_from_lustre={t_load:.1f}s "
                  f"(rep 1 may hit page cache)", flush=True)
        del m, o, s, ms, os_; torch.cuda.empty_cache()
    dist.destroy_process_group()
if __name__ == "__main__": main()
