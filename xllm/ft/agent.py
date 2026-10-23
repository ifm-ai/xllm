"""
Per-node agent for in-band fault tolerance (ft).

One agent runs on each node and starts that node's trainers, one per GPU. The
agent of node 0 is the leader: it hosts the control store (heartbeats, the
current generation, requests) and is the only one that decides. On a failure
(a trainer exits with an error, or makes no progress past the hang deadline),
the leader bumps the generation; every agent kills its trainers, waits until
its GPUs' memory is released and starts the next generation in the same
allocation, with a fresh rendezvous port. The job ends with exit code 75
(requeue) when recovery is not possible here: a node stopped sending
heartbeats, a GPU stayed busy after the kill, or generations keep failing
before any training step.

Usage, one agent per node (e.g. srun --ntasks-per-node=1 ...):
    python -m xllm.ft.agent train.py --ft.enabled true --ft.mtbf_sys 86400 ...
The agent reads its settings from the trainer's own --ft_* arguments.
"""

import argparse
import dataclasses
import datetime
import glob
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional

from torch.distributed import TCPStore

from xllm.ft.inject import InjectionReader
from xllm.ft.progress import PROGRESS_ENV, Progress, progress_path
from xllm.ft.replica import NO_COMMON_STEP_EXIT

logger = logging.getLogger()

EXIT_FINISHED = 0
EXIT_REQUEUE = 75
WARMUP_STEPS = 10  # steps whose gaps are measured before the auto hang deadline is armed
FINISH_ACK_TIMEOUT = 60.0
KILL_TIMEOUT = 60.0
HANG_FLOOR_S = 30.0  # the auto step deadline never goes below this: step-time jitter is not a hang  # a trainer still alive this long after SIGKILL is stuck in the kernel: the node is unusable
RELEASE_SLACK_BYTES = 256 * 2**20  # GPU memory above the pre-spawn baseline still counted as released


@dataclasses.dataclass
class AgentConf:
    slot_dir: str
    node_deadline: float
    hb_interval: float
    hb_deadline: float
    hb_margin: float
    retries: int
    release_timeout: float
    nccl_timeout: int


def agent_conf(train_args: List[str]) -> AgentConf:
    """Reads the ft settings from the trainer's arguments, with TrainerConf's defaults."""
    from xllm.config import TrainerConf
    from xllm.ft.config import FtConf

    defaults = {f.name: f.default for f in dataclasses.fields(TrainerConf)}
    defaults.update({f"ft.{f.name}": f.default for f in dataclasses.fields(FtConf)})
    names = {
        "slot_dir": "ft.slot_dir", "node_deadline": "ft.node_deadline", "hb_interval": "ft.hb_interval",
        "hb_deadline": "ft.hb_deadline", "hb_margin": "ft.hb_margin", "retries": "ft.retries",
        "release_timeout": "ft.release_timeout", "nccl_timeout": "nccl_timeout",
    }
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for field in dataclasses.fields(AgentConf):
        parser.add_argument(f"--{names[field.name]}", type=field.type, default=defaults[names[field.name]])
    known, _ = parser.parse_known_args(train_args)
    return AgentConf(**{f.name: getattr(known, names[f.name]) for f in dataclasses.fields(AgentConf)})


class GpuMemory:
    """Used memory per local GPU through NVML. Without NVML the release poll only relies on kill(), which already
    waited for every trainer PID to exit."""

    def __init__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as e:  # NVML missing (non-NVIDIA) or unusable
            logger.warning(f"ft agent: NVML unavailable ({e}); GPU release is checked by process exit only")
            self.nvml, self.handles = None, []

    def count(self) -> Optional[int]:
        return len(self.handles) if self.nvml else None

    def used(self) -> List[int]:
        return [self.nvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles] if self.nvml else []


class Agent:
    def __init__(self, train_args: List[str]):
        self.train_args = train_args
        self.conf = agent_conf(train_args)
        self.node = int(os.environ.get("SLURM_NODEID", "0"))
        self.nnodes = int(os.environ.get("SLURM_NNODES", "1"))
        self.is_leader = self.node == 0
        self.gpu = GpuMemory()
        self.gpus = self.gpu.count() or int(os.environ.get("FT_GPUS_PER_NODE", "8"))
        # the leader is node 0, the first host of the allocation: FT_LEADER_ADDR is set by the sbatch script
        # (outside the container); scontrol is only a fallback where it is available
        self.leader_addr = os.environ.get("FT_LEADER_ADDR") or self._first_host()
        job = int(os.environ.get("SLURM_JOB_ID", "0"))
        self.store_port = int(os.environ.get("FT_STORE_PORT", 20000 + job % 20000))
        self.procs: List[subprocess.Popen] = []
        self.baseline = self.gpu.used()
        self.teardown_s: Optional[float] = None
        self.startup_s: Optional[float] = None  # spawn to first training step, measured in the last generation
        self.retries_used = 0  # consecutive failed generations that got no further than the previous failure
        self.failed_at = 0  # highest step reached by the last failed generation
        local_ranks = [self.node * self.gpus + i for i in range(self.gpus)]
        self.injections = InjectionReader(node=self.node, ranks=local_ranks)
        self.store = self._connect(self.is_leader)
        self.hb_seen = {n: (-1, time.monotonic()) for n in range(self.nnodes)}
        self.stopping = threading.Event()
        self.hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.hb_thread.start()
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)  # the sbatch script owns the requeue

    def _connect(self, host: bool) -> TCPStore:
        return TCPStore(
            self.leader_addr, self.store_port, world_size=None, is_master=host,
            timeout=datetime.timedelta(seconds=max(60.0, 6 * self.conf.node_deadline)), wait_for_workers=False,
        )

    def _heartbeat(self):
        """Counter heartbeat for the agent's whole life, on its own connection, also while waiting or releasing GPUs."""
        store = None
        while not self.stopping.is_set():
            try:
                store = store or self._connect(False)
                store.add(f"hb/{self.node}", 1)
            except Exception as e:  # reconnect next time; a lasting failure shows up as a lost node
                logger.warning(f"ft agent {self.node}: heartbeat failed ({e})")
                store = None
            time.sleep(self.conf.hb_interval)

    def lost_node(self) -> Optional[int]:
        """Leader: a node whose heartbeat stopped for longer than the node deadline and that did not finish."""
        now = time.monotonic()
        for n in range(self.nnodes):
            count = int(self.store.add(f"hb/{n}", 0))
            if count != self.hb_seen[n][0]:
                self.hb_seen[n] = (count, now)
            else:
                # an agent that never sent a heartbeat may still be starting (a slow container start): it gets the
                # startup deadline, not the node deadline
                deadline = self.conf.node_deadline if count > 0 else self.startup_deadline()
                if now - self.hb_seen[n][1] <= deadline:
                    continue
                state = self.store.get(f"state/{n}").decode() if self.store.check([f"state/{n}"]) else "?"
                if not state.startswith("finished"):
                    logger.error(f"ft leader: node {n} sent no heartbeat for {deadline:.0f} s ({state})")
                    return n
        return None

    @staticmethod
    def _first_host() -> str:
        hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        return hosts.split()[0].decode()

    # ------------------------------------------------------------------ trainers
    def spawn(self, gen: int):
        self.procs = []
        for local in range(self.gpus):
            rank = self.node * self.gpus + local
            path = progress_path(self.conf.slot_dir, rank)
            Progress(path, create=True)
            env = dict(os.environ)
            env.pop("TORCHELASTIC_USE_AGENT_STORE", None)  # each generation brings up its own rendezvous store
            env.update(
                RANK=str(rank), SLURM_PROCID=str(rank), WORLD_SIZE=str(self.nnodes * self.gpus),
                LOCAL_RANK=str(local), LOCAL_WORLD_SIZE=str(self.gpus), MASTER_ADDR=self.leader_addr,
                MASTER_PORT=str(self.store_port + 1 + gen), FT_AGENT_PID=str(os.getpid()), FT_GEN=str(gen),
                FT_RESTORE=self.restore_mode(gen), **{PROGRESS_ENV: str(path)},
            )
            self.procs.append(subprocess.Popen([sys.executable, "-m", "xllm.ft.child"] + self.train_args, env=env))
        self.state("running", gen)
        logger.info(f"ft agent {self.node}: generation {gen}, {self.gpus} trainers started")

    def restore_mode(self, gen: int) -> str:
        """memory or disk, decided by the leader for each generation; the first start reads the disk checkpoint."""
        key = f"mode/{gen}"
        return self.store.get(key).decode() if gen > 0 and self.store.check([key]) else "disk"

    def kill(self, timeout: float = KILL_TIMEOUT) -> bool:
        """SIGKILL every trainer and wait for them; False when one is still alive after the timeout."""
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)
        t0 = time.monotonic()
        for p in self.procs:
            try:
                p.wait(timeout=max(0.0, timeout - (time.monotonic() - t0)))
            except subprocess.TimeoutExpired:
                logger.error(f"ft agent {self.node}: trainer {p.pid} still alive {timeout:.0f} s after SIGKILL")
                return False
        return True

    def release_poll(self) -> bool:
        """Wait until every local GPU is back to its pre-spawn memory use; False on timeout."""
        timeout = self.conf.release_timeout
        if timeout <= 0:
            timeout = 120.0 if self.teardown_s is None else max(30.0, (1 + self.conf.hb_margin) * self.teardown_s)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            used = self.gpu.used()
            if all(u <= b + RELEASE_SLACK_BYTES for u, b in zip(used, self.baseline)):
                self.teardown_s = time.monotonic() - t0
                return True
            time.sleep(0.1)
        logger.error(f"ft agent {self.node}: GPU memory not released after {timeout:.0f} s: {self.gpu.used()}")
        return False

    def cleanup(self):
        """Remove this node's slots and progress files: never trusted after the job step ends."""
        job = os.environ.get("SLURM_JOB_ID", "local")
        for f in glob.glob(os.path.join(self.conf.slot_dir, f"ft_{job}_*")):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def _on_sigterm(self, signum, frame):
        logger.warning(f"ft agent {self.node}: SIGTERM, stopping trainers")
        self.kill()
        self.cleanup()
        os._exit(EXIT_REQUEUE)

    # ------------------------------------------------------------------ control store
    def state(self, name: str, gen: int):
        self.store.set(f"state/{self.node}", f"{name}:{gen}")

    def gen(self) -> int:
        return int(self.store.get("gen"))

    def request(self, gen: int, reason: str):
        if not self.store.check([f"req/{gen}"]):
            logger.warning(f"ft agent {self.node}: generation {gen} failed: {reason}")
            self.store.set(f"req/{gen}", f"node {self.node}: {reason}")

    def wait_go(self, gen: int) -> bool:
        """Generation gen starts once every agent has released its GPUs (the leader decides)."""
        deadline = time.monotonic() + self.wait_deadline()
        if self.is_leader:
            while gen > 0 and int(self.store.add(f"free/{gen}", 0)) < self.nnodes:
                lost = self.lost_node()
                if lost is not None:
                    self.store.set("end", f"node {lost} lost")  # no spare yet: requeue
                    return False
                if self.store.check(["end"]):
                    return False
                if time.monotonic() > deadline:
                    self.store.set("end", f"not every node released its GPUs for generation {gen}")
                    return False
                time.sleep(self.conf.hb_interval)
            self.store.set(f"go/{gen}", "1")
        else:
            while not self.store.check([f"go/{gen}"]):
                if self.store.check(["end"]) or time.monotonic() > deadline:
                    return False
                time.sleep(self.conf.hb_interval)
        return True

    def wait_deadline(self) -> float:
        """Deadline of a wait between generations: (1 + m) times the last teardown, at least 120 s."""
        teardown = self.teardown_s if self.teardown_s is not None else 0.0
        return max(120.0, (1 + self.conf.hb_margin) * (teardown + KILL_TIMEOUT))

    # ------------------------------------------------------------------ monitoring
    def monitor(self, gen: int) -> str:
        """Returns "finished", "restart" or "end" (requeue)."""
        progress = [Progress(progress_path(self.conf.slot_dir, self.node * self.gpus + i), create=False)
                    for i in range(self.gpus)]
        spawned = time.monotonic()
        last = [(-1, spawned)] * self.gpus
        gaps: List[float] = []
        warm = False
        last_reached = 0
        while True:
            if self.store.check(["end"]):
                return "end"
            if self.gen() > gen:
                return "restart"
            codes = [p.poll() for p in self.procs]
            if all(c == 0 for c in codes):
                return "finished"
            failed = [(i, c) for i, c in enumerate(codes) if c not in (None, 0)]
            if failed:
                if any(c == NO_COMMON_STEP_EXIT for _, c in failed):
                    self.store.set(f"nocommon/{gen}", "1")  # memory cannot restore: next generation from disk
                self.request(gen, f"trainer exit codes {failed}")
            deadline = self.hang_deadline(gaps)
            now = time.monotonic()
            for i, prog in enumerate(progress):
                step, paused = prog.read()
                if codes[i] == 0:
                    continue  # this trainer finished
                if step == 0:  # still starting: only the startup deadline applies
                    limit = self.startup_deadline()
                    if limit is not None and now - spawned > limit:
                        self.request(gen, f"rank {self.node * self.gpus + i} did not start training within {limit:.0f} s")
                    continue
                if step != last[i][0]:
                    if last[i][0] > 0 and not paused:
                        gaps.append(now - last[i][1])
                    last[i] = (step, now)
                elif paused:
                    last[i] = (step, now)  # the counter is paused during eval and checkpointing
                elif deadline is not None and now - last[i][1] > deadline:
                    self.request(gen, f"rank {self.node * self.gpus + i} made no progress for {deadline:.0f} s")
            reached = max(p.read()[0] for p in progress)
            if reached > last_reached:
                last_reached = reached
                self.store.set(f"reached/{gen}/{self.node}", str(reached))
            self.inject(reached)
            if not warm and all(p.read()[0] > 0 for p in progress):
                warm = True
                self.startup_s = time.monotonic() - spawned
                self.store.set(f"warm/{gen}/{self.node}", "1")
            if self.is_leader:
                outcome = self.lead(gen)
                if outcome is not None:
                    return outcome
            time.sleep(self.conf.hb_interval)

    def inject(self, step: int):
        """Carry out the test injections addressed to this node (tests/ft_inject.py); the rest come with later lanes."""
        self.injections.poll(self.store)
        for inj in self.injections.due(step):
            logger.warning(f"ft agent {self.node}: injecting {inj}")
            if inj.kind == "kill_rank":
                self.procs[inj.rank - self.node * self.gpus].send_signal(signal.SIGKILL)
            elif inj.kind == "hang_rank":
                self.procs[inj.rank - self.node * self.gpus].send_signal(signal.SIGSTOP)
            elif inj.kind == "kill_process":
                os._exit(9)  # the agent dies; its trainers follow through PR_SET_PDEATHSIG
            elif inj.kind == "kill_node":
                os.killpg(os.getpgid(0), signal.SIGKILL)
            else:
                logger.warning(f"ft agent {self.node}: injection {inj.kind} not handled yet")

    def startup_deadline(self) -> Optional[float]:
        """Before a generation's first step: (1 + m) times the last measured startup, at least 60 s; unknown at first."""
        if self.startup_s is None:
            return 0.9 * self.conf.nccl_timeout
        return min(max(60.0, (1 + self.conf.hb_margin) * self.startup_s), 0.9 * self.conf.nccl_timeout)

    def hang_deadline(self, gaps: List[float]) -> Optional[float]:
        """Configured, or (1 + m) times the longest measured step gap once WARMUP_STEPS gaps are seen."""
        if self.conf.hb_deadline > 0:
            return self.conf.hb_deadline
        if len(gaps) < WARMUP_STEPS * self.gpus:
            return 0.9 * self.conf.nccl_timeout
        measured = (1 + self.conf.hb_margin) * max(gaps)
        return min(max(measured, HANG_FLOOR_S), 0.9 * self.conf.nccl_timeout)

    def lead(self, gen: int) -> Optional[str]:
        lost = self.lost_node()
        if lost is not None:
            self.store.set("end", f"node {lost} lost")  # no spare yet: requeue
            return "end"
        if self.store.check([f"req/{gen}"]):
            # a generation that failed without getting past the previous failure (or before every node trained
            # a step) is a retry: a failure that repeats at the same step must not restart forever
            warm = all(self.store.check([f"warm/{gen}/{n}"]) for n in range(self.nnodes))
            reached = max(int(self.store.get(f"reached/{gen}/{n}")) if self.store.check([f"reached/{gen}/{n}"]) else 0
                          for n in range(self.nnodes))
            progressed = warm and reached > self.failed_at
            self.retries_used = 0 if progressed else self.retries_used + 1
            self.failed_at = max(self.failed_at, reached)
            reason = self.store.get(f"req/{gen}").decode()
            if self.retries_used > self.conf.retries:
                logger.error(f"ft leader: {self.retries_used} generations in a row failed without progress ({reason})")
                self.store.set("end", reason)
                return "end"
            # memory first; disk when memory has no common step or R memory restarts in a row failed
            no_common = self.store.check([f"nocommon/{gen}"])
            mode = "disk" if no_common or self.retries_used >= self.conf.retries else "memory"
            logger.warning(f"ft leader: restarting all trainers from {mode}, generation {gen} -> {gen + 1} ({reason})")
            self.store.set(f"mode/{gen + 1}", mode)
            self.store.set("gen", str(gen + 1))
            return "restart"
        return None

    # ------------------------------------------------------------------ main loop
    def finish(self) -> int:
        self.state("finished", self.gen())
        self.store.add("done", 1)
        if self.is_leader:
            while int(self.store.add("done", 0)) < self.nnodes:
                lost = self.lost_node()  # a node lost now never reports done: not a finished run
                if lost is not None:
                    self.store.set("end", f"node {lost} lost before it finished")
                    return EXIT_REQUEUE
                time.sleep(self.conf.hb_interval)
            self.store.set("finished", "1")
            t0 = time.monotonic()
            while int(self.store.add("ack", 0)) < self.nnodes - 1 and time.monotonic() - t0 < FINISH_ACK_TIMEOUT:
                time.sleep(self.conf.hb_interval)
        else:
            while not self.store.check(["finished"]):  # raises if the leader is gone before it finished
                if self.store.check(["end"]):
                    return EXIT_REQUEUE
                time.sleep(self.conf.hb_interval)
            try:
                self.store.add("ack", 1)
            except Exception:  # the leader already closed the store: the run did finish
                pass
        logger.info(f"ft agent {self.node}: run finished")
        return EXIT_FINISHED

    def run(self) -> int:
        if self.is_leader:
            self.store.set("gen", "0")
        gen = 0
        try:
            while True:
                if not self.wait_go(gen):
                    return EXIT_REQUEUE
                self.spawn(gen)
                outcome = self.monitor(gen)
                if outcome == "finished":
                    return self.finish()
                self.state("exiting", gen)
                if not self.kill():
                    self.store.set("end", f"node {self.node}: a trainer survived SIGKILL")
                    return EXIT_REQUEUE
                if outcome == "end":
                    return EXIT_REQUEUE
                gen = self.gen()
                if not self.release_poll():
                    self.store.set("end", f"node {self.node}: GPU memory not released")
                    return EXIT_REQUEUE
                self.store.add(f"free/{gen}", 1)
        except Exception as e:  # the control store went away (leader lost) or anything unexpected
            logger.error(f"ft agent {self.node}: {type(e).__name__}: {e}")
            self.kill()
            return EXIT_REQUEUE
        finally:
            self.cleanup()
            self.stopping.set()
            self.hb_thread.join(timeout=5 * self.conf.hb_interval)


def main():
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s ft-agent {socket.gethostname()} %(message)s")
    code = Agent(sys.argv[1:]).run()
    # exit without running C++ destructors of the store objects (the leader's store may already be gone)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
