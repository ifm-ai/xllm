"""
Health check of an ft spare node before it is admitted. First NVML's health
counters of every GPU (uncorrectable ECC errors, a pending or failed row
remap: a GPU that needs a reset). Then child processes, one per GPU, so the
agent itself never holds a CUDA context: each child allocates memory and runs
a matmul on its GPU, then all of them run
xLLM's collective benchmark across the node's GPUs over NCCL. The check across
nodes is the recovery's first training step, which a newly admitted node must
pass like any other.

Usage (by the agent): run_spare_check(gpus, timeout) -> bool
"""

import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger()


def nvml_healthy() -> bool:
    """False when a GPU's counters say it needs a reset or a replacement; True without NVML (the GPU checks still run)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        return True
    for i, h in enumerate(handles):
        try:
            errors = pynvml.nvmlDeviceGetTotalEccErrors(h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                                        pynvml.NVML_VOLATILE_ECC)
        except Exception:  # ECC off or not supported (or an older pynvml)
            errors = 0
        try:
            _, _, pending, failed = pynvml.nvmlDeviceGetRemappedRows(h)
        except Exception:  # no row remapping on this GPU
            pending = failed = 0
        if errors or pending or failed:
            logger.error(f"ft: GPU {i} is not healthy (uncorrectable ECC errors {errors}, row remap pending "
                         f"{bool(pending)}, failed {bool(failed)})")
            return False
    return True


def run_spare_check(gpus: int, timeout: float, port: int) -> bool:
    if not nvml_healthy():
        logger.info("ft: spare health check FAILED (NVML)")
        return False
    procs = []
    for local in range(gpus):
        env = dict(os.environ, RANK=str(local), LOCAL_RANK=str(local), WORLD_SIZE=str(gpus),
                   LOCAL_WORLD_SIZE=str(gpus), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        env.pop("FT_AGENT_PID", None)
        procs.append(subprocess.Popen([sys.executable, "-m", "xllm.ft.spare_check"], env=env))
    t0 = time.time()
    ok = True
    for p in procs:
        try:
            ok = p.wait(timeout=max(0.0, timeout - (time.time() - t0))) == 0 and ok
        except subprocess.TimeoutExpired:
            ok = False
            break
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    logger.info(f"ft: spare health check {'passed' if ok else 'FAILED'} in {time.time() - t0:.1f} s")
    return ok


def child() -> int:
    import torch
    import torch.distributed as dist

    local = int(os.environ["LOCAL_RANK"])
    gpus = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    a = torch.randn(4096, 4096, device="cuda")
    if not torch.isfinite(a @ a).all():
        return 1
    torch.cuda.synchronize()
    if gpus == 1:  # nothing to test over NCCL on a single-GPU node
        return 0
    from xllm.cluster_check.comms_bench import collective_bench

    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    check = collective_bench(ngpu_per_node=gpus)
    dist.destroy_process_group()
    return 0 if check.global_bandwidth > 0 else 1


if __name__ == "__main__":
    sys.exit(child())
