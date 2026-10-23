"""
Per-node agent for in-band fault tolerance (ft).

One agent runs on each node of the allocation. The first nodes are training
nodes: each agent starts its node's trainers, one per GPU. The last ft_spares
nodes are idle spares: their agents pass a health check and wait; refill-spare
jobs started by the SLURM helper join the pool the same way.

The agent of physical node 0 is the leader: it hosts the control store
(heartbeats, the current generation, the map from logical training nodes to
physical nodes, the spare pool) and is the only one that decides. On a failure
(a trainer exits with an error, makes no progress past the hang deadline, or
its GPU reports a critical Xid) the leader bumps the generation; every agent
kills its trainers, waits until its GPUs' memory is released and the next
generation starts in the same allocation, from the in-memory copies, with a
fresh rendezvous port. A training node that stops sending heartbeats, or whose
GPU needs a reset, is replaced by a spare, which takes over its logical node
and receives its state from the node holding its copy; a refill spare is then
requested. The job ends with exit code 75 (requeue) when recovery is not
possible here: no spare arrives in time, a GPU stays busy after the kill,
generations keep failing without progress, or the leader is lost.

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
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from torch.distributed import TCPStore

from xllm.ft import dumper
from xllm.ft.gpu_faults import GpuFaultLane
from xllm.ft.inject import InjectionReader
from xllm.ft.progress import PROGRESS_ENV, Progress, progress_path
from xllm.ft.replica import DUMP_LOAD_EXIT, NO_COMMON_STEP_EXIT, dump_files, meta_path, slot_path, valid_steps
from xllm.ft.scheduler import SlurmMailbox, mailbox_dir
from xllm.ft.sizing import SIZING_EXIT
from xllm.ft.spare_check import run_spare_check

logger = logging.getLogger()

EXIT_FINISHED = 0
EXIT_REQUEUE = 75
WARMUP_STEPS = 10  # steps whose gaps are measured before the auto hang deadline is armed
FINISH_ACK_TIMEOUT = 60.0
KILL_TIMEOUT = 60.0  # a trainer still alive this long after SIGKILL is stuck in the kernel: the node is unusable
HANG_FLOOR_S = 30.0  # the auto step deadline never goes below this: step-time jitter is not a hang
REFILL_WAIT_FLOOR_S = 300.0  # waiting for a spare never gives up earlier than this
NEW_NODE_FAILURES = 3  # new nodes in a row failing their first step: the cause is not the nodes, retries apply
SPARE_CHECK_TIMEOUT = 300.0
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
    xid_replace: str
    spares: int
    spare_health: bool
    refill_wait: float
    dump_dir: str
    dump_enabled: bool
    dump_keep: int
    ft_dump_dir: Optional[str]


def agent_conf(train_args: List[str]) -> AgentConf:
    """Reads the ft settings from the trainer's arguments, with TrainerConf's defaults."""
    from xllm.config import TrainerConf
    from xllm.ft.config import FtConf

    defaults = {f.name: f.default for f in dataclasses.fields(TrainerConf)}
    defaults.update({f"ft.{f.name}": f.default for f in dataclasses.fields(FtConf)})
    names = {
        "slot_dir": "ft.slot_dir", "node_deadline": "ft.node_deadline", "hb_interval": "ft.hb_interval",
        "hb_deadline": "ft.hb_deadline", "hb_margin": "ft.hb_margin", "retries": "ft.retries",
        "release_timeout": "ft.release_timeout", "nccl_timeout": "nccl_timeout", "xid_replace": "ft.xid_replace",
        "spares": "ft.spares", "spare_health": "ft.spare_health", "refill_wait": "ft.refill_wait_s",
        "dump_dir": "dump_dir", "dump_enabled": "ft.dump_enabled", "dump_keep": "ft.dump_keep",
        "ft_dump_dir": "ft.dump_dir",
    }
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for field in dataclasses.fields(AgentConf):
        kind = (lambda v: str(v).lower() in ("1", "true", "yes")) if field.type is bool else field.type
        kind = str if field.type == Optional[str] else kind
        parser.add_argument(f"--{names[field.name]}", type=kind, default=defaults[names[field.name]],
                            dest=names[field.name])
    known, _ = parser.parse_known_args(train_args)
    conf = AgentConf(**{f.name: getattr(known, names[f.name]) for f in dataclasses.fields(AgentConf)})
    assert conf.hb_deadline < conf.nccl_timeout, "ft.hb_deadline: the hang deadline must fire before the NCCL timeout"
    return conf


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
        self.refill = os.environ.get("FT_REFILL") == "1"  # a refill-spare job of its own, joining the main job
        self.is_leader = not self.refill and int(os.environ.get("SLURM_NODEID", "0")) == 0
        self.gpu = GpuMemory()
        self.gpus = self.gpu.count() or int(os.environ.get("FT_GPUS_PER_NODE", "8"))
        # the leader is physical node 0, the first host of the allocation: FT_LEADER_ADDR is set by the sbatch
        # script (outside the container); scontrol is only a fallback where it is available
        self.leader_addr = os.environ.get("FT_LEADER_ADDR") or self._first_host()
        job = int(os.environ.get("FT_MAIN_JOB_ID", os.environ.get("SLURM_JOB_ID", "0")))
        self.main_job = job
        self.store_port = int(os.environ.get("FT_STORE_PORT", 20000 + job % 20000))
        self.procs: List[subprocess.Popen] = []
        self.baseline = self.gpu.used()
        self.teardown_s: Optional[float] = None
        self.startup_s: Optional[float] = None  # spawn to first training step, measured in the last generation
        self.retries_used = 0  # consecutive failed generations that got no further than the previous failure
        self.new_failures = 0  # consecutive new nodes that failed their first step (barred without a retry)
        self.warm_gen = -1  # the last generation in which every node trained a step (it resets new_failures)
        self.failed_at = 0  # highest step reached by the last failed generation
        self.logical: Optional[int] = None  # the training node this agent runs, or None while it is a spare
        self.injections = InjectionReader(node=-1, ranks=[])
        self.faults: List = []  # (gpu, xid) reported by the GPU-fault lane's thread
        self.faults_lock = threading.Lock()
        self.xids_to_replace = {int(x) for x in self.conf.xid_replace.split(",") if x}
        self.gpu_faults = GpuFaultLane(self._on_gpu_fault)
        self.store = self._connect(self.is_leader) if self.is_leader else self._join()
        if self.is_leader:
            nnodes = int(os.environ.get("SLURM_NNODES", "1"))
            if self.conf.spares == 0:
                logger.warning("ft leader: ft.spares=0: no spare node; a lost node waits for a refill node, "
                               "else the job requeues")
            if socket.gethostname() in self.barred_file_hosts():
                logger.error(f"ft leader: this node ({socket.gethostname()}) was barred in an earlier run of the job")
            # before ntrain, which the other agents wait for: they see the bar list as soon as they start
            self.store.set("barred", ",".join(self.barred_file_hosts()))  # nodes barred in earlier runs of this job
            self.store.set("ntrain", str(nnodes - self.conf.spares))
            for logical in range(nnodes - self.conf.spares):
                self.store.set(f"map/{logical}", str(logical))
            self.store.set("epoch", f"{job}:{os.environ.get('SLURM_RESTART_COUNT', '0')}")
            self.store.add("next_phys", nnodes)
            self.store.set("gen", "0")
        self.ntrain = int(self.store.get("ntrain"))
        os.environ["FT_EPOCH"] = self.store.get("epoch").decode()  # slot names and run identity of the main job
        self.phys = int(os.environ.get("SLURM_NODEID", "0")) if not self.refill else self.store.add("next_phys", 1) - 1
        self.store.set(f"host/{self.phys}", socket.gethostname())
        self.dump_root = Path(self.conf.ft_dump_dir or os.path.join(self.conf.dump_dir, "ft_checkpoints"))
        if self.is_leader:
            dumper.clean_staging(self.dump_root)
        self.hb_seen: Dict[int, tuple] = {}
        self.dump_abort = threading.Event()  # set by the dump_kill test injection
        self.stopping = threading.Event()
        self.hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.hb_thread.start()
        if self.conf.dump_enabled:
            threading.Thread(target=self._dumps, daemon=True).start()
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)  # the sbatch script owns the requeue

    def _join(self) -> TCPStore:
        """
        First connection to the leader's control store: the leader's node may start much later than this one (a
        slow container start, minutes), so keep trying until the startup deadline before giving up.
        """
        deadline = time.monotonic() + 0.9 * self.conf.nccl_timeout
        while True:
            try:
                return self._connect(False)
            except Exception as e:  # DistNetworkError / DistStoreError: the store is not up yet
                if time.monotonic() > deadline:
                    raise
                logger.warning(f"ft agent: control store at {self.leader_addr}:{self.store_port} not reachable yet "
                               f"({type(e).__name__}); retrying")

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
                store.add(f"hb/{self.phys}", 1)
            except Exception as e:  # reconnect next time; a lasting failure shows up as a lost node
                logger.warning(f"ft agent {self.phys}: heartbeat failed ({e})")
                store = None
            time.sleep(self.conf.hb_interval)

    def lost_node(self, nodes: List[int]) -> Optional[int]:
        """Leader: a physical node whose heartbeat stopped for longer than the node deadline and that did not finish."""
        now = time.monotonic()
        for n in nodes:
            count = int(self.store.add(f"hb/{n}", 0))
            if count != self.hb_seen.get(n, (None,))[0]:
                self.hb_seen[n] = (count, now)
            else:
                # an agent that never sent a heartbeat may still be starting (a slow container start): it gets the
                # startup deadline, not the node deadline
                deadline = self.conf.node_deadline if count > 0 else self.startup_deadline()
                if now - self.hb_seen[n][1] <= deadline:
                    continue
                state = self.store.get(f"state/{n}").decode() if self.store.check([f"state/{n}"]) else "?"
                if not state.startswith("finished"):
                    logger.error(f"ft leader: physical node {n} sent no heartbeat for {deadline:.0f} s ({state})")
                    return n
        return None

    def heard_from(self, n: int) -> bool:
        """Leader: no heartbeat silence longer than the node deadline seen for this node (quiet lost_node)."""
        count, seen = int(self.store.add(f"hb/{n}", 0)), self.hb_seen.get(n)
        return seen is None or count != seen[0] or time.monotonic() - seen[1] <= self.conf.node_deadline

    # ------------------------------------------------------------------ agent dumps and the disk fallback
    def disk_candidate(self) -> str:
        """Leader: the newest complete agent dump not marked bad, if newer than xLLM's checkpoint ('' = xLLM)."""
        try:
            bad = dumper.bad_dumps(self.dump_root)
            dumps = [(step, d) for step, d in dumper.committed_dumps(self.dump_root, self.ntrain * self.gpus)
                     if (d.name, step) not in bad and not self.store.check([f"baddump/{d.name}"])]
            if dumps and dumps[0][0] > dumper.newest_xllm_checkpoint_step(self.conf.dump_dir):
                return str(dumps[0][1])
        except OSError as e:  # the shared filesystem is down: the trainers' own load then fails or waits
            logger.error(f"ft leader: cannot list the agent dumps ({e}); falling back to xLLM's checkpoints")
        return ""

    def set_mode(self, gen: int, mode: str):
        """Leader: the restore mode of generation gen, and for disk the candidate to load."""
        self.store.set(f"mode/{gen}", mode)  # a disk candidate is chosen in wait_go, right before the start

    def _dumps(self):
        """Background agent dumps from the RAM slots, on their own store connection (see xllm/ft/dumper.py)."""
        store, done = None, set()
        while not self.stopping.is_set():
            time.sleep(1.0)
            try:
                store = store or self._connect(False)
                logical = self.logical
                if logical is None or self.procs == [] or any(p.poll() is not None for p in self.procs):
                    continue
                ranks = list(range(logical * self.gpus, (logical + 1) * self.gpus))
                newest = dumper.newest_common_own_step(self.conf.slot_dir, ranks)
                if newest is not None:
                    store.set(f"newest/{logical}", str(newest))
                current = store.get("dump/current").decode() if store.check(["dump/current"]) else ""
                if current and (current, logical) not in done:
                    step, gen, _ = (int(x) for x in current.split(":"))
                    staging = dumper.staging_dir(self.dump_root, gen, step)
                    done.add((current, logical))
                    try:
                        ok = all(dumper.write_rank(self.conf.slot_dir, staging, r, gen, step, self.dump_abort)
                                 for r in ranks)
                    except OSError as e:
                        logger.warning(f"ft agent {self.phys}: dump of step {step} failed ({e})")
                        ok = False
                    store.set(f"dumped/{current}/{logical}" if ok else f"dumpfail/{current}", "1")
                if self.is_leader:
                    self._lead_dumps(store, current)
            except Exception as e:  # dumps are best effort: training and recovery never depend on them
                logger.warning(f"ft agent {self.phys}: dump step failed ({type(e).__name__}: {e})")
                store = None

    def _lead_dumps(self, store, current: str):
        if current:
            step, gen, _ = (int(x) for x in current.split(":"))
            staging = dumper.staging_dir(self.dump_root, gen, step)
            # a failed round, or one of an earlier generation whose nodes changed since (a part may be gone)
            now = int(store.get("gen"))
            replaced = any(store.check([f"new/{g}/{lg}"]) for g in range(gen + 1, now + 1) for lg in range(self.ntrain))
            if store.check([f"dumpfail/{current}"]) or replaced:
                logger.warning(f"ft leader: agent dump round of step {step} dropped")
                shutil.rmtree(staging, ignore_errors=True)
                store.set("dump/current", "")
            elif all(store.check([f"dumped/{current}/{lg}"]) for lg in range(self.ntrain)):
                in_use = self.get_or(f"dumpdir/{int(store.get('gen'))}", "")  # the dump this generation loads
                if dumper.commit(self.dump_root, staging, gen, step, self.ntrain * self.gpus, self.conf.dump_keep,
                                 protect=in_use):
                    store.set("dump/last", str(step))
                else:  # incomplete or mixed markers: drop this round, the next one starts clean
                    shutil.rmtree(staging, ignore_errors=True)
                store.set("dump/current", "")
            return
        newest = [store.get(f"newest/{lg}").decode() if store.check([f"newest/{lg}"]) else None
                  for lg in range(self.ntrain)]
        if None in newest:
            return
        step = min(int(n) for n in newest)
        last = int(store.get("dump/last")) if store.check(["dump/last"]) else -1
        if step > last:  # each round has its own id, so a retried step is never mistaken for a finished one
            store.set("dump/current", f"{step}:{int(store.get('gen'))}:{store.add('dump/round', 1)}")

    def get_or(self, key: str, default: str) -> str:
        """A read that never waits: TCPStore.get on a missing key blocks for the whole store timeout."""
        return self.store.get(key).decode() if self.store.check([key]) else default

    # ------------------------------------------------------------------ map and spare pool (control store)
    def mapped(self) -> Dict[int, int]:
        """logical training node -> physical node"""
        return {logical: int(self.store.get(f"map/{logical}")) for logical in range(self.ntrain)}

    def my_logical(self) -> Optional[int]:
        return next((logical for logical, phys in self.mapped().items() if phys == self.phys), None)

    def spares(self) -> List[int]:
        """Admitted spares, in the order they joined."""
        used = set(self.mapped().values())
        return [p for p in range(int(self.store.add("next_phys", 0)))
                if p not in used and self.store.check([f"spare/{p}"]) and self.store.get(f"spare/{p}") == b"ready"]

    def barred(self) -> List[str]:
        return [h for h in self.store.get("barred").decode().split(",") if h] if self.store.check(["barred"]) else []

    def barred_file(self) -> Path:
        return Path(self.conf.dump_dir) / f".barred.{self.main_job}"  # this job's only: a later job starts clean

    def barred_file_hosts(self) -> List[str]:
        """The bar list outlives the control store: it is kept in <dump_dir>/.barred.<job> across requeues."""
        try:
            return sorted({h for h in self.barred_file().read_text().split() if h})
        except OSError:
            return []

    def bar(self, phys: int, reason: str):
        """A node that failed is never admitted again, and refill requests exclude it."""
        host = self.get_or(f"host/{phys}", "")
        logger.error(f"ft leader: barring {host or 'an unregistered host'} (physical node {phys}): {reason}")
        if host:  # a node that died before registering its host cannot be excluded from refills by name
            self.store.set("barred", ",".join(sorted(set(self.barred()) | {host})))
            try:
                with open(self.barred_file(), "a") as f:
                    f.write(host + "\n")
            except OSError as e:
                logger.warning(f"ft leader: could not persist the bar of {host} ({e})")
        self.store.set(f"spare/{phys}", "barred")

    def request_refill(self):
        """Ask the SLURM helper for a 1-node refill spare, in the background."""
        def ask():
            try:
                job = SlurmMailbox(mailbox_dir(self.conf.dump_dir)).request_node(exclude=self.barred())
                logger.info(f"ft leader: refill spare requested (job {job})")
            except Exception as e:
                logger.warning(f"ft leader: refill spare request failed ({e})")
        threading.Thread(target=ask, daemon=True).start()

    def take_spare(self, target_gen: int, logical: int, phys: int, reason: str) -> bool:
        """Leader: map the first spare to a failed logical node for generation target_gen; False if none arrives."""
        deadline = time.monotonic() + max(REFILL_WAIT_FLOOR_S, self.conf.refill_wait)
        while not self.spares():
            if time.monotonic() > deadline:
                self.store.set("end", f"logical node {logical} lost ({reason}) and no spare arrived")
                return False
            time.sleep(self.conf.hb_interval)
        spare = self.spares()[0]
        self.store.set(f"map/{logical}", str(spare))
        self.store.set(f"spare/{spare}", "used")
        self.store.set(f"new/{target_gen}/{logical}", "1")  # its warm-up failure bars it and costs no retry
        logger.warning(f"ft leader: physical node {spare} replaces logical node {logical} (was {phys}): {reason}")
        self.request_refill()
        return True

    def replace(self, gen: int, logical: int, phys: int, reason: str, mode: str) -> str:
        """Leader: put a spare in place of a failed training node and start the next generation."""
        if phys == self.phys:  # the leader hosts the control store: its node cannot be handed over
            self.store.set("end", f"the leader's node must be replaced ({reason})")
            return "end"
        if not self.take_spare(gen + 1, logical, phys, reason):
            return "end"
        self.set_mode(gen + 1, mode)
        self.store.set("gen", str(gen + 1))
        return "restart"

    def account(self, gen: int, reason: str) -> Optional[str]:
        """
        Retry accounting shared by every failure (r_used): a generation that failed without getting past the
        previous failure, or before every node trained a step, is a retry. Returns the restore mode of the next
        generation (memory first, disk when memory has no common step or R restarts in a row made no progress),
        or None when the job step must end.
        """
        warm = all(self.store.check([f"warm/{gen}/{lg}"]) for lg in range(self.ntrain))
        reached = max(int(self.get_or(f"reached/{gen}/{lg}", "0")) for lg in range(self.ntrain))
        progressed = warm and reached > self.failed_at
        self.retries_used = 0 if progressed else self.retries_used + 1
        if progressed:
            self.new_failures = 0
        self.failed_at = max(self.failed_at, reached)
        if self.retries_used > self.conf.retries:
            logger.error(f"ft leader: {self.retries_used} generations in a row failed without progress ({reason})")
            self.store.set("end", reason)
            return None
        no_common = self.store.check([f"nocommon/{gen}"])
        return "disk" if no_common or self.retries_used >= self.conf.retries else "memory"

    @staticmethod
    def _first_host() -> str:
        hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        return hosts.split()[0].decode()

    # ------------------------------------------------------------------ trainers
    def spawn(self, gen: int, logical: int):
        self.procs = []
        self.logical = logical
        ranks = [logical * self.gpus + i for i in range(self.gpus)]
        self.injections.node, self.injections.ranks = logical, ranks
        master = self.get_or(f"host/{self.mapped()[0]}", self.leader_addr)  # rank 0 follows logical node 0
        replaced = self.store.get(f"replaced/{gen}").decode() if self.store.check([f"replaced/{gen}"]) else ""
        for local, rank in enumerate(ranks):
            path = progress_path(self.conf.slot_dir, rank)
            Progress(path, create=True)
            env = dict(os.environ)
            env.pop("TORCHELASTIC_USE_AGENT_STORE", None)  # each generation brings up its own rendezvous store
            env.update(
                RANK=str(rank), SLURM_PROCID=str(rank), WORLD_SIZE=str(self.ntrain * self.gpus),
                LOCAL_RANK=str(local), LOCAL_WORLD_SIZE=str(self.gpus), MASTER_ADDR=master,
                MASTER_PORT=str(self.store_port + 1 + gen), FT_AGENT_PID=str(os.getpid()), FT_GEN=str(gen),
                FT_RESTORE=self.restore_mode(gen), FT_REPLACED=replaced, FT_DUMP=self.get_or(f"dumpdir/{gen}", ""),
                **{PROGRESS_ENV: str(path)},
            )
            # the trainers' output as the agent's (python -u); xllm/ft/child.py passes it on from its own write_through
            unbuffered = ["-u"] if sys.stdout.write_through else []
            self.procs.append(subprocess.Popen([sys.executable, *unbuffered, "-m", "xllm.ft.child"] + self.train_args,
                                               env=env))
        self.state("running", gen)
        logger.info(f"ft agent {self.phys}: generation {gen}, logical node {logical}, {self.gpus} trainers started")

    def restore_mode(self, gen: int) -> str:
        """memory or disk, decided by the leader for each generation; the first start reads the disk checkpoint."""
        key = f"mode/{gen}"
        return self.store.get(key).decode() if gen > 0 and self.store.check([key]) else "disk"

    def has_own_copies(self, logical: int) -> bool:
        """Every local rank holds a valid own copy of this run (else the node restores from its holder)."""
        return all(valid_steps(lambda s, r=r: slot_path(self.conf.slot_dir, "own", r, s), r)
                   for r in range(logical * self.gpus, (logical + 1) * self.gpus))

    def holds_copies(self, logical: int) -> bool:
        """Any valid copy of this run on this node: an own one of a local rank, or one held for the previous node."""
        held = ((logical - 1) % self.ntrain) * self.gpus  # this node holds the copies of the previous node's ranks
        return any(valid_steps(lambda s, r=r: slot_path(self.conf.slot_dir, kind, r, s), r)
                   for kind, first in (("own", logical * self.gpus), ("peer", held))
                   for r in range(first, first + self.gpus))

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
                logger.error(f"ft agent {self.phys}: trainer {p.pid} still alive {timeout:.0f} s after SIGKILL")
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
        logger.error(f"ft agent {self.phys}: GPU memory not released after {timeout:.0f} s: {self.gpu.used()}")
        return False

    def cleanup(self):
        """Remove this node's slots and progress files: never trusted after the job step ends."""
        job = os.environ["FT_EPOCH"].split(":")[0]
        for f in glob.glob(os.path.join(self.conf.slot_dir, f"ft_{job}_*")):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def _on_sigterm(self, signum, frame):
        logger.warning(f"ft agent {self.phys}: SIGTERM, stopping trainers")
        self.kill()
        self.cleanup()
        os._exit(EXIT_REQUEUE)

    def _on_gpu_fault(self, gpu: int, xid: Optional[int]):
        with self.faults_lock:
            self.faults.append((gpu, xid))

    def check_gpu_faults(self, gen: int):
        """A critical Xid or a lost GPU is a crash of this node's trainers; some need the node replaced."""
        with self.faults_lock:
            faults, self.faults = self.faults, []
        for gpu, xid in faults:
            what = "GPU lost" if xid is None else f"Xid {xid}"
            if xid is None or xid in self.xids_to_replace:
                logger.error(f"ft agent {self.phys}: {what} on GPU {gpu}: this node must be replaced")
                self.store.set(f"bad/{self.phys}", f"{what} on GPU {gpu}")
            self.request(gen, f"{what} on GPU {gpu}")

    # ------------------------------------------------------------------ control store
    def state(self, name: str, gen: int):
        self.store.set(f"state/{self.phys}", f"{name}:{gen}")

    def gen(self) -> int:
        return int(self.store.get("gen"))

    def request(self, gen: int, reason: str):
        if not self.store.check([f"req/{gen}"]):
            logger.warning(f"ft agent {self.phys}: generation {gen} failed: {reason}")
            self.store.set(f"req/{gen}", f"logical node {self.logical}: {reason}")
            self.store.set(f"reqnode/{gen}", str(self.logical))

    def wait_go(self, gen: int) -> bool:
        """Generation gen starts once the training nodes of the last one released their GPUs (the leader decides)."""
        deadline = time.monotonic() + self.wait_deadline()
        if self.is_leader:
            new = [lg for lg in range(self.ntrain) if self.store.check([f"new/{gen}/{lg}"])]
            while gen > 0 and not all(self.store.check([f"free/{gen}/{lg}"]) for lg in range(self.ntrain)
                                      if lg not in new):
                mapped = self.mapped()
                lost = self.lost_node([p for lg, p in mapped.items() if lg not in new and p != self.phys])
                if lost is not None:  # lost before this generation started: re-map it, no new generation
                    self.store.set(f"spare/{lost}", "lost")
                    logical = next(lg for lg, p in mapped.items() if p == lost)
                    if not self.take_spare(gen, logical, lost, f"physical node {lost} lost during a restart"):
                        return False
                    new.append(logical)
                    continue
                if self.store.check(["end"]):
                    return False
                if time.monotonic() > deadline:
                    self.store.set("end", f"not every node released its GPUs for generation {gen}")
                    return False
                time.sleep(self.conf.hb_interval)
            if gen > 0 and self.restore_mode(gen) == "memory" and all(
                    lg in new or self.store.check([f"nocopy/{gen}/{lg}"]) for lg in range(self.ntrain)):
                # no node holds any copy yet (a failure before the first copy): memory cannot restore, go to disk
                logger.warning(f"ft leader: no in-memory copy exists; generation {gen} restores from disk")
                self.set_mode(gen, "disk")
            # nodes without their own copies (new ones, or not copied since they joined) restore from their holder
            fresh = [lg for lg in range(self.ntrain) if lg in new or self.store.check([f"fresh/{gen}/{lg}"])]
            self.store.set(f"replaced/{gen}", ",".join(str(lg) for lg in fresh))
            if gen == 0 or self.restore_mode(gen) == "disk":  # the newest dump committed by now, set before go
                self.store.set(f"dumpdir/{gen}", self.disk_candidate())
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
        logical = self.logical
        progress = [Progress(progress_path(self.conf.slot_dir, logical * self.gpus + i), create=False)
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
            self.check_gpu_faults(gen)
            failed = [(i, c) for i, c in enumerate(codes) if c not in (None, 0)]
            if failed:
                if all(c == SIZING_EXIT for _, c in failed):
                    return "refused"
                if any(c == NO_COMMON_STEP_EXIT for _, c in failed):
                    self.store.set(f"nocommon/{gen}", "1")  # memory cannot restore: next generation from disk
                if any(c == DUMP_LOAD_EXIT for _, c in failed):
                    self.store.set(f"dumpfail/{gen}", "1")
                    bad = self.get_or(f"dumpdir/{gen}", "")  # this dump cannot be loaded: the next candidate is used
                    if bad:
                        self.store.set(f"baddump/{Path(bad).name}", "1")  # even when .bad cannot be written
                        dumper.mark_bad(self.dump_root, Path(bad))
                    self.store.set(f"nocommon/{gen}", "1")
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
                        self.request(gen, f"rank {logical * self.gpus + i} did not start training within {limit:.0f} s")
                    continue
                if step != last[i][0]:
                    if last[i][0] > 0 and not paused:
                        gaps.append(now - last[i][1])
                    last[i] = (step, now)
                elif paused:
                    last[i] = (step, now)  # the counter is paused during eval and checkpointing
                elif deadline is not None and now - last[i][1] > deadline:
                    self.request(gen, f"rank {logical * self.gpus + i} made no progress for {deadline:.0f} s")
            reached = max(p.read()[0] for p in progress)
            if reached > last_reached:
                last_reached = reached
                self.store.set(f"reached/{gen}/{logical}", str(reached))
            self.inject(reached)
            if not warm and all(p.read()[0] > 0 for p in progress):
                warm = True
                self.startup_s = time.monotonic() - spawned
                self.store.set(f"warm/{gen}/{logical}", "1")
            if self.is_leader:
                outcome = self.lead(gen)
                if outcome is not None:
                    return outcome
            time.sleep(self.conf.hb_interval)

    def inject(self, step: int):
        """Carry out the test injections addressed to this node (tests/ft_inject.py)."""
        self.injections.poll(self.store)
        for inj in self.injections.due(step, self.store):
            logger.warning(f"ft agent {self.phys}: injecting {inj}")
            if inj.kind == "kill_rank":
                self.procs[inj.rank - self.logical * self.gpus].send_signal(signal.SIGKILL)
            elif inj.kind == "hang_rank":
                self.procs[inj.rank - self.logical * self.gpus].send_signal(signal.SIGSTOP)
            elif inj.kind == "kill_process":
                os._exit(9)  # the agent dies; its trainers follow through PR_SET_PDEATHSIG
            elif inj.kind == "kill_node":
                os.killpg(os.getpgid(0), signal.SIGKILL)
            elif inj.kind == "xid":
                self.gpu_faults.inject(inj.gpu, inj.xid)
            elif inj.kind == "gpu_lost":
                self.gpu_faults.inject(inj.gpu, None)
            elif inj.kind == "fail_spare_health":
                self.store.set(f"fail_health/{self.phys}", "1")
            elif inj.kind == "torn_write":
                self.torn_write(inj.rank)
            elif inj.kind == "bitflip":
                self.bitflip(inj.rank)
            elif inj.kind == "dump_kill":
                self.dump_abort.set()

    def torn_write(self, rank: int):
        """Injection: the trainer dies halfway through overwriting its newest own slot (metadata removed first,
        half of the payload overwritten, a stale .tmp left), as a crash in the middle of a copy would leave it."""
        steps = valid_steps(lambda s: slot_path(self.conf.slot_dir, "own", rank, s), rank)
        if steps:
            path = slot_path(self.conf.slot_dir, "own", rank, steps[max(steps)])
            meta_path(path).unlink()
            with open(path, "r+b") as f:
                f.write(bytes(path.stat().st_size // 2))
            meta_path(path).with_suffix(".tmp").write_bytes(b"torn")
        self.procs[rank - self.logical * self.gpus].send_signal(signal.SIGKILL)

    def bitflip(self, rank: int):
        """Injection: flip one bit of this rank's payload in the newest committed agent dump (its crc32 must catch it)."""
        dumps = dumper.committed_dumps(self.dump_root, None)
        if dumps:
            payload = dump_files(dumps[0][1], rank)[0]
            with open(payload, "r+b") as f:
                f.seek(payload.stat().st_size // 2)
                byte = f.read(1)
                f.seek(-1, os.SEEK_CUR)
                f.write(bytes([byte[0] ^ 1]))

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
        if self.new_failures and gen != self.warm_gen and all(
                self.store.check([f"warm/{gen}/{lg}"]) for lg in range(self.ntrain)):
            self.new_failures, self.warm_gen = 0, gen  # every node trained: earlier bars were the nodes' fault
        mapped = self.mapped()
        lost = self.lost_node([p for p in mapped.values() if p != self.phys])
        if lost is not None:
            self.store.set(f"spare/{lost}", "lost")  # not waited for at the end; re-admitted if it comes back
            logical = next(lg for lg, p in mapped.items() if p == lost)
            mode = self.account(gen, f"physical node {lost} lost")
            return "end" if mode is None else self.replace(gen, logical, lost, f"physical node {lost} lost", mode)
        dead_spare = self.lost_node(self.spares())
        if dead_spare is not None:  # never hand a dead spare to a training node
            self.store.set(f"spare/{dead_spare}", "lost")
        if self.store.check([f"req/{gen}"]):
            reason = self.store.get(f"req/{gen}").decode()
            bad = [(lg, p) for lg, p in mapped.items() if self.store.check([f"bad/{p}"])]
            if bad:  # a GPU that needs a reset: the node is barred and replaced
                logical, phys = bad[0]
                self.bar(phys, self.store.get(f"bad/{phys}").decode())
                mode = self.account(gen, reason)
                return "end" if mode is None else self.replace(gen, logical, phys, reason, mode)
            failed = int(self.get_or(f"reqnode/{gen}", "-1"))
            restore = self.store.check([f"nocommon/{gen}"]) or self.store.check([f"dumpfail/{gen}"])
            if failed in mapped and self.store.check([f"new/{gen}/{failed}"]) and not self.store.check(
                    [f"warm/{gen}/{failed}"]) and not restore and self.new_failures < NEW_NODE_FAILURES:
                # a newly admitted node failed before its first step: bar it, take the next spare, no retry used
                # (not for a restore decision, 76/77; and only a few times in a row: then the cause is elsewhere)
                self.new_failures += 1
                self.bar(mapped[failed], f"failed its first step after joining ({reason})")
                mode = "disk" if self.store.check([f"nocommon/{gen}"]) else "memory"
                return self.replace(gen, failed, mapped[failed], reason, mode)
            mode = self.account(gen, reason)
            if mode is None:
                return "end"
            logger.warning(f"ft leader: restarting all trainers from {mode}, generation {gen} -> {gen + 1} ({reason})")
            self.set_mode(gen + 1, mode)
            self.store.set("gen", str(gen + 1))
            return "restart"
        return None

    # ------------------------------------------------------------------ main loop
    def finish(self) -> int:
        self.state("finished", self.gen())
        self.store.add("done", 1)
        if self.is_leader:
            while int(self.store.add("done", 0)) < self.ntrain:
                lost = self.lost_node(list(self.mapped().values()))  # a node lost now never reports done
                if lost is not None:
                    self.store.set("end", f"physical node {lost} lost before it finished")
                    return EXIT_REQUEUE
                time.sleep(self.conf.hb_interval)
            self.store.set("finished", "1")
            t0 = time.monotonic()
            self.write_finished_marker()
            # every registered agent that can still ack: not barred or lost, and not silent past the node deadline
            others = len([p for p in range(int(self.store.add("next_phys", 0))) if p != self.phys
                          and self.store.check([f"host/{p}"]) and self.get_or(f"spare/{p}", "") not in ("barred", "lost")
                          and p in self.hb_seen and self.heard_from(p)])  # never polled: never waited for
            while int(self.store.add("ack", 0)) < others and time.monotonic() - t0 < FINISH_ACK_TIMEOUT:
                time.sleep(self.conf.hb_interval)
        else:
            self.wait_finished()
        logger.info(f"ft agent {self.phys}: run finished")
        return EXIT_FINISHED

    def write_finished_marker(self):
        """The sbatch script requeues the job unless this file exists (scripts/ft_train.sbatch)."""
        job, restart = os.environ["FT_EPOCH"].split(":")
        marker = Path(self.conf.dump_dir) / f"ft_finished.{job}.{restart}"
        try:
            marker.write_text(f"finished at generation {self.gen()}\n")
        except OSError as e:  # the run did finish; without the marker the script requeues once more, from disk
            logger.error(f"ft leader: could not write {marker} ({e})")

    def wait_finished(self):
        while not self.store.check(["finished"]):  # raises if the leader is gone before it finished
            if self.store.check(["end"]):
                raise RuntimeError("the job step ends")
            time.sleep(self.conf.hb_interval)
        try:
            self.store.add("ack", 1)
        except Exception:  # the leader already closed the store: the run did finish
            pass

    def park(self) -> Optional[int]:
        """
        Spare: pass the health check, join the pool and wait; returns the generation that maps this node, or None
        when the run finished or this node is barred (fenced: it just ends, the run goes on without it).
        """
        status = self.store.get(f"spare/{self.phys}") if self.store.check([f"spare/{self.phys}"]) else b""
        if status != b"barred" and socket.gethostname() in self.barred():  # barred in an earlier run
            self.store.set(f"spare/{self.phys}", "barred")
            status = b"barred"
        if status == b"barred":
            logger.warning(f"ft agent {self.phys}: this node is barred; it leaves the run")
            return None
        if status != b"ready":
            healthy = not self.conf.spare_health or (
                not self.store.check([f"fail_health/{self.phys}"])
                and run_spare_check(self.gpus, SPARE_CHECK_TIMEOUT, self.store_port + 10000 + self.phys))
            if not healthy:
                self.store.set(f"spare/{self.phys}", "barred")
                logger.warning(f"ft agent {self.phys}: spare health check failed; this node leaves the run")
                return None
            self.store.set(f"spare/{self.phys}", "ready")
        self.state("spare", self.gen())
        logger.info(f"ft agent {self.phys}: spare, waiting")
        while True:
            if self.store.check(["finished"]):
                self.wait_finished()
                return None
            if self.store.check(["end"]):
                raise RuntimeError("the job step ends")
            gen = self.gen()
            if self.my_logical() is not None:  # mapped, also into the generation being started (no gen change)
                return gen
            time.sleep(self.conf.hb_interval)

    def run(self) -> int:
        gen = 0
        main_restart = os.environ.get("FT_MAIN_RESTART")
        try:
            if self.refill and main_restart is not None and main_restart != os.environ["FT_EPOCH"].split(":")[1]:
                logger.warning(f"ft agent {self.phys}: refill job of an earlier run of the main job (restart "
                               f"{main_restart}); not joining")
                self.store.set(f"spare/{self.phys}", "lost")  # never counted as a spare or as an agent to wait for
                return EXIT_FINISHED
            while True:
                logical = self.my_logical()
                if logical is None:  # a spare (or a fenced node that was declared dead): wait in the pool
                    gen = self.park()
                    if gen is None:
                        return EXIT_FINISHED
                    continue
                if gen == 0 and socket.gethostname() in self.barred():
                    # barred in an earlier run of this job and given back by SLURM: replace it right away
                    if self.is_leader:  # the leader's node cannot be handed over: end before any trainer starts
                        self.store.set("end", f"the leader's node {socket.gethostname()} was barred in an earlier run")
                        return EXIT_REQUEUE
                    self.store.set(f"bad/{self.phys}", "barred in an earlier run of this job")
                    self.request(gen, f"physical node {self.phys} was barred in an earlier run")
                    while self.my_logical() == logical and not self.store.check(["end"]):  # no trainers here
                        time.sleep(self.conf.hb_interval)
                    if self.store.check(["end"]):
                        return EXIT_REQUEUE
                    continue  # replaced: park() refuses the barred node
                if not self.wait_go(gen):
                    return EXIT_REQUEUE
                self.spawn(gen, logical)
                outcome = self.monitor(gen)
                if outcome == "finished":
                    return self.finish()
                if outcome == "refused":  # a configuration error: the job ends and is not requeued
                    self.kill()
                    self.store.set("end", "the ft sizing gate refused ft (see the trainers' log)")
                    return SIZING_EXIT
                self.state("exiting", gen)
                if not self.kill():
                    self.store.set("end", f"physical node {self.phys}: a trainer survived SIGKILL")
                    return EXIT_REQUEUE
                if outcome == "end":
                    return EXIT_REQUEUE
                gen = self.gen()
                if self.my_logical() != logical:  # replaced (declared dead or bad GPU): fenced, join the pool
                    continue
                if not self.release_poll():
                    self.store.set("end", f"physical node {self.phys}: GPU memory not released")
                    return EXIT_REQUEUE
                if not self.has_own_copies(logical):
                    self.store.set(f"fresh/{gen}/{logical}", "1")
                if not self.holds_copies(logical):
                    self.store.set(f"nocopy/{gen}/{logical}", "1")
                self.store.set(f"free/{gen}/{logical}", "1")
        except Exception as e:  # the control store went away (leader lost), barred, or anything unexpected
            logger.error(f"ft agent {self.phys}: {type(e).__name__}: {e}")
            self.kill()
            return EXIT_REQUEUE
        finally:
            self.cleanup()
            self.stopping.set()
            self.hb_thread.join(timeout=5 * self.conf.hb_interval)


def main():
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s ft-agent {socket.gethostname()} %(message)s")
    agent = Agent(sys.argv[1:])
    code = agent.run()
    if agent.refill:  # a refill spare's own 1-node job just ends; only the main job's script decides requeues
        code = EXIT_FINISHED
    # exit without running C++ destructors of the store objects (the leader's store may already be gone)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
