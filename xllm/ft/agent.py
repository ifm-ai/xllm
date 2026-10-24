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

from torch.distributed import DistError, TCPStore

from xllm.ft import dumper
from xllm.ft.gpu_faults import GpuFaultLane
from xllm.ft.inject import InjectionReader
from xllm.ft.leader import Lease, Liveness, lease_dir
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
TERM_PORT_OFFSET = 2000  # the store of term t > 0 listens on store_port + 2000 + t (rendezvous: store_port + 1 + gen)
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
    lease_ttl: float = 60.0
    lease_renew: float = 6.0


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
        "ft_dump_dir": "ft.dump_dir", "lease_ttl": "ft.lease_ttl", "lease_renew": "ft.lease_renew",
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


def takeover_wait(leader: Dict, state: Dict, phys: int, lease_ttl: float) -> float:
    """
    Seconds of leader silence after which this agent claims the next term: the designated standby after
    1.5 lease_ttl, every other training node one lease_ttl later per place in line (a dead standby is covered);
    never for a spare or the old leader itself.
    """
    line = [leader.get("standby")] + sorted(int(p) for p in state.get("map", {}).values())
    line = [p for i, p in enumerate(line) if p not in line[:i] and p not in (-1, None, leader.get("phys"))]
    if phys not in line or leader.get("ended"):  # a leader that ended the step: nothing to take over
        return float("inf")
    return lease_ttl * (1.5 + line.index(phys))


class LeaderLost(Exception):
    """The control store of this agent's leader is gone, or a newer leader (higher term) took over."""


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
        self.port = self.store_port  # the current leader's store: term 0 on store_port, term t on +2000+t
        self.term = 0  # the leader's term this agent follows
        self.newer_term = 0  # the highest term read from the shared FS (the standby thread keeps it current)
        self.store_epoch = 0  # bumped at every switch to a new leader's store: the side threads reconnect
        self.known_lost: set = set()  # nodes the leader knows are gone (a lost leader, a node that did not rejoin)
        self.takeover: Optional[Dict] = None  # set by the standby thread when this agent won the next term
        self.renewed_at = time.monotonic()  # the leader's last lease renewal (see frozen)
        self.spare_ready = False
        self.finished_locally = False  # this node's trainers all exited 0: the run reached its last step
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
        self.lease_lock = threading.Lock()  # the lease thread's publish vs an end's (mark_ended)
        self.bad_reason = ""  # why this node must be replaced: told again to a new leader (the old store is gone)
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
        self.lease = Lease(lease_dir(self.conf.dump_dir, *os.environ["FT_EPOCH"].split(":")))
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
        if self.is_leader:
            self.store.set("term", "0")
            self._start_lease(0)
        else:
            threading.Thread(target=self._watch_leader, daemon=True).start()
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
                logger.warning(f"ft agent: control store at {self.leader_addr}:{self.port} not reachable yet "
                               f"({type(e).__name__}); retrying")
                time.sleep(5.0)
                # a late start: the first leader may already be gone and a standby lead a newer term
                restart = os.environ.get("FT_MAIN_RESTART", os.environ.get("SLURM_RESTART_COUNT", "0"))
                leader = Lease(lease_dir(self.conf.dump_dir, str(self.main_job), restart)).leader() or {}
                if int(leader.get("term", 0)) > self.term and leader.get("host"):
                    self.leader_addr, self.port = leader["host"], int(leader["port"])
                    self.term = self.newer_term = int(leader["term"])

    def _connect(self, host: bool) -> TCPStore:
        if not host:  # a quick probe first: a store client waits its whole timeout on a closed port
            socket.create_connection((self.leader_addr, self.port), timeout=self.conf.lease_renew).close()
        return TCPStore(
            self.leader_addr, self.port, world_size=None, is_master=host,
            timeout=datetime.timedelta(seconds=max(60.0, 6 * self.conf.node_deadline)), wait_for_workers=False,
        )

    def _heartbeat(self):
        """Counter heartbeat for the agent's whole life, on its own connection, also while waiting or releasing GPUs."""
        store, epoch = None, self.store_epoch
        while not self.stopping.is_set():
            try:
                if epoch != self.store_epoch:  # a new leader: its store from now on
                    store, epoch = None, self.store_epoch
                store = store or self._connect(False)
                store.add(f"hb/{self.phys}", 1)
            except Exception as e:  # reconnect next time; a lasting failure shows up as a lost node
                if store is not None:
                    logger.warning(f"ft agent {self.phys}: heartbeat failed ({e})")
                store = None
            time.sleep(self.conf.hb_interval)

    def lost_node(self, nodes: List[int]) -> Optional[int]:
        """Leader: a physical node whose heartbeat stopped for longer than the node deadline and that did not finish."""
        now = time.monotonic()
        for n in nodes:
            if n in self.known_lost and not self.store.check([f"reg/{self.term}/{n}"]):  # unless it came back late
                return n
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
        store, done, epoch = None, set(), self.store_epoch
        while not self.stopping.is_set():
            time.sleep(1.0)
            try:
                if epoch != self.store_epoch:
                    store, epoch = None, self.store_epoch
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
        if self.frozen:
            return
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
                self.bad_reason = f"{what} on GPU {gpu}"
                self.store.set(f"bad/{self.phys}", self.bad_reason)
            self.request(gen, f"{what} on GPU {gpu}")

    # ------------------------------------------------------------------ control store
    def state(self, name: str, gen: int):
        self.store.set(f"state/{self.phys}", f"{name}:{gen}")

    def gen(self) -> int:
        """
        The current generation; raises LeaderLost when a newer leader took over (the term on the shared FS), or when
        this agent won the next term (its leader may be alive but cut off: nothing else would stop this generation).
        """
        if self.newer_term > self.term or self.takeover is not None or self.store.check(["moved"]):
            raise LeaderLost("this agent won the next term" if self.takeover is not None
                             else f"a newer leader (term {self.newer_term})")
        return int(self.store.get("gen"))

    def request(self, gen: int, reason: str):
        if not self.store.check([f"req/{gen}"]):
            logger.warning(f"ft agent {self.phys}: generation {gen} failed: {reason}")
            self.store.set(f"req/{gen}", f"logical node {self.logical}: {reason}")
            self.store.set(f"reqnode/{gen}", str(-1 if self.logical is None else self.logical))

    def wait_go(self, gen: int) -> bool:
        """Generation gen starts once the training nodes of the last one released their GPUs (the leader decides)."""
        deadline = time.monotonic() + self.wait_deadline()
        if self.is_leader:
            new = [lg for lg in range(self.ntrain) if self.store.check([f"new/{gen}/{lg}"])]
            while self.frozen or gen > 0 and not all(self.store.check([f"free/{gen}/{lg}"])
                                                     for lg in range(self.ntrain) if lg not in new):
                if self.frozen:  # no decision until the lease is renewed again (or a newer leader took over)
                    self.gen()
                    self.check_abandoned()
                    time.sleep(self.conf.hb_interval)
                    continue
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
            while not self.store.check([f"go/{gen}"]) and self.gen() == gen:  # moved on: the caller re-reads it
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
            elif inj.kind == "lease_cut":
                self.lease = Lease(Path("/proc/ft_lease_cut"))  # nothing can be created or read there

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
        if self.frozen:
            self.check_abandoned()
            return None
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

    # ------------------------------------------------------------------ leader lease and takeover (xllm/ft/leader.py)
    @property
    def frozen(self) -> bool:
        """A leader whose lease is not renewed for lease_ttl/2 decides nothing (on NFS a renewal may also block)."""
        return time.monotonic() - self.renewed_at > self.conf.lease_ttl / 2

    def check_abandoned(self):
        """Frozen leader: when every other training node stopped heartbeating here, they follow a newer leader."""
        others = [p for p in self.mapped().values() if p != self.phys]
        if others and all(self.lost_node([p]) is not None for p in others):
            self.store.set("moved", "abandoned")
            raise LeaderLost("cut off from the shared FS, and its agents left for a newer leader")

    def _start_lease(self, term: int):
        """Leader: publish the store's address for this term and keep the lease renewed (its own thread)."""
        self.renewed_at = time.monotonic()
        self.lease.claim(term)
        threading.Thread(target=self._lead_lease, args=(term,), daemon=True).start()

    def _lead_lease(self, term: int):
        """
        Leader: renew the lease, keep the published standby and the state (journal) current. A renewal that
        fails for lease_ttl/2 freezes all decisions; a newer term on the FS (this leader was cut off) moves its
        agents to the newer leader.
        """
        store, counter, standby = None, 0, None
        while not self.stopping.is_set() and self.term == term:
            try:
                store = store or self._connect(False)
                counter += 1
                if self.lease.renew(term, counter):
                    self.renewed_at = time.monotonic()
                state = self.lease_state(store, term)
                barred, hosts = set(state["barred"]), state.pop("hosts")
                choice = min((p for p in state["map"].values() if p != self.phys and p not in self.known_lost
                              and hosts.get(p, "") not in barred), default=-1)
                with self.lease_lock:  # never over an end's publish
                    if self.stopping.is_set():
                        return
                    now = self.lease.leader()
                    if now is not None and now.get("term", -1) > term:
                        logger.error(f"ft leader {self.phys}: term {now['term']} was taken over; stepping down")
                        self.newer_term = now["term"]
                        store.set("moved", str(now["term"]))
                        return
                    if choice != standby or counter == 1:
                        standby = choice
                        self.lease.publish({"term": term, "host": self.leader_addr, "port": self.port,
                                            "phys": self.phys, "standby": standby})
                    self.lease.save_state(state)
            except Exception as e:  # best effort: the next round retries
                logger.warning(f"ft leader {self.phys}: lease round failed ({type(e).__name__}: {e})")
                store = None
            time.sleep(self.conf.lease_renew)

    def lease_state(self, store, term: int) -> Dict:
        """The leader's state as a standby needs it (the journal), read from this store."""
        mapped = {lg: int(store.get(f"map/{lg}")) for lg in range(self.ntrain)}
        barred = set(store.get("barred").decode().split(",")) if store.check(["barred"]) else set()
        hosts = {p: store.get(f"host/{p}").decode() for p in mapped.values() if store.check([f"host/{p}"])}
        return {"term": term, "gen": int(store.get("gen")), "map": mapped, "ntrain": self.ntrain,
                "next_phys": int(store.add("next_phys", 0)), "barred": sorted(h for h in barred if h),
                "retries_used": self.retries_used, "failed_at": self.failed_at, "new_failures": self.new_failures,
                "hosts": hosts}

    def mark_ended(self, code: int):
        """Leader that ends the step (anything but a finish): no agent may take over its term and train on."""
        if not self.is_leader or code == EXIT_FINISHED:
            return
        with self.lease_lock:
            leader = self.lease.leader() or {}
            if int(leader.get("term", -1)) == self.term:
                self.lease.publish(dict(leader, ended=True))

    def _watch_leader(self):
        """
        Every agent but the leader: read the published leader, keep newer_term current (the fence), and take over
        when the leader's renewal counter stops: the designated standby after 1.5 lease_ttl, any other training
        node one lease_ttl later per place in line, so that a dead standby is covered too.
        """
        alive, term_seen = Liveness(), None
        while not self.stopping.is_set():
            time.sleep(min(2.0, self.conf.lease_renew))
            leader = self.lease.leader()
            if leader is None or self.takeover is not None or self.is_leader:
                continue
            term = int(leader.get("term", 0))
            self.newer_term = max(self.newer_term, term)
            if term != term_seen:
                alive, term_seen = Liveness(), term
            silent = alive.silent_for(self.lease.counter(term))
            # no journal yet (the leader died before its first save): the map it started with
            state = self.lease.state() or {"map": {str(lg): lg for lg in range(self.ntrain)}}
            wait = takeover_wait(leader, state, self.phys, self.conf.lease_ttl)
            if wait > silent:
                continue
            if self.lease.counter(term + 1) is None and silent > wait + 2 * self.conf.lease_ttl:
                # term + 1 was claimed but never led (its winner died right after the claim): skip it
                self.lease.publish(dict(leader, term=term + 1, host="", port=0, standby=-1))
                continue
            if self.lease.claim(term + 1):
                try:
                    self._become_leader(term + 1, leader, state)
                except Exception as e:  # a term nobody leads: published silent, so the next in line claims the one after
                    logger.error(f"ft agent {self.phys}: could not lead term {term + 1} ({type(e).__name__}: {e})")
                    self.lease.publish({"term": term + 1, "host": "", "port": 0, "phys": self.phys, "standby": -1})

    def _become_leader(self, term: int, old: Dict, state: Dict):
        """This agent won the term: open its store with the last leader's state; the main thread takes over."""
        port = self.store_port + TERM_PORT_OFFSET + term
        store = TCPStore(socket.gethostname(), port, world_size=None, is_master=True,
                         timeout=datetime.timedelta(seconds=max(60.0, 6 * self.conf.node_deadline)),
                         wait_for_workers=False)
        store.set("barred", ",".join(sorted(set(state.get("barred", [])) | set(self.barred_file_hosts()))))
        store.set("ntrain", str(self.ntrain))
        for lg, p in state.get("map", {}).items():
            store.set(f"map/{lg}", str(p))
        store.set("epoch", os.environ["FT_EPOCH"])
        store.add("next_phys", int(state.get("next_phys", self.ntrain + self.conf.spares)))
        store.set("gen", str(state.get("gen", 0)))
        store.set("term", str(term))
        self.retries_used, self.failed_at = state.get("retries_used", 0), state.get("failed_at", 0)
        self.new_failures = state.get("new_failures", 0)
        logger.warning(f"ft agent {self.phys}: the leader (physical node {old.get('phys')}, term {term - 1}) is "
                       f"silent; taking over as the leader of term {term}")
        self.takeover = {"term": term, "store": store, "port": port, "old": int(old.get("phys", -1)), "state": state}

    def rejoin(self) -> bool:
        """
        The leader's store is gone or a newer leader took over: every leader loss starts a new generation, so
        stop the trainers, then follow the newer leader (or become it); False when none appears in time.
        """
        logger.warning(f"ft agent {self.phys}: lost the leader of term {self.term}; waiting for the next one")
        self.kill()
        if self.is_leader:  # a leader that lost its term (cut off and taken over): an ordinary agent from now on
            self.is_leader = False
            threading.Thread(target=self._watch_leader, daemon=True).start()
        # the standby takes over after 1.5 lease_ttl of silence, the next ones a lease_ttl apart
        deadline = time.monotonic() + self.conf.lease_ttl * 4.5 + self.wait_deadline()
        alive = Liveness()
        while time.monotonic() < deadline:
            if self.takeover is not None:
                return self._take_over()
            leader = self.lease.leader()
            if leader is not None and leader.get("ended") and int(leader.get("term", 0)) == self.term:
                logger.warning(f"ft agent {self.phys}: the leader ended the step")
                return False
            same = leader is not None and int(leader.get("term", 0)) == self.term
            if same and alive.silent_for(self.lease.counter(self.term)) < 3 * self.conf.lease_renew and \
                    time.monotonic() - alive.since < self.conf.lease_ttl:
                try:  # the leader is alive (a network glitch): back to the same store
                    self.store = self._connect(False)
                    self.store_epoch += 1
                    self.reannounce()
                    return True
                except Exception:
                    pass
            if leader is not None and int(leader.get("term", 0)) > self.term:
                try:
                    self.leader_addr, self.port = leader["host"], int(leader["port"])
                    self.store = self._connect(False)
                except Exception:  # its store is not up yet
                    time.sleep(2.0)
                    continue
                self.term = self.newer_term = int(leader["term"])
                self.store_epoch += 1
                self.reannounce()
                return True
            time.sleep(2.0)
        return False

    def reannounce(self):
        """This agent, in the new leader's store: its host, its state as a spare, and its registration."""
        self.store.set(f"host/{self.phys}", socket.gethostname())
        behind = self.phys + 1 - int(self.store.add("next_phys", 0))
        if behind > 0:  # a refill numbered after the last journal save: never number another node the same
            self.store.add("next_phys", behind)
        if self.spare_ready and self.my_logical() is None:
            self.store.set(f"spare/{self.phys}", "ready")
        if self.bad_reason:
            self.store.set(f"bad/{self.phys}", self.bad_reason)
        if self.finished_locally:
            self.store.set(f"fin/{self.term}/{self.phys}", "1")
        self.store.set(f"reg/{self.term}/{self.phys}", "1")

    def _take_over(self) -> bool:
        """
        The new leader, in the main thread: wait for the agents to register (no node is judged on an empty
        heartbeat baseline), then replace the old leader's node like any lost node and start the next generation.
        """
        t = self.takeover
        self.store, self.port, self.term, self.leader_addr = t["store"], t["port"], t["term"], socket.gethostname()
        self.newer_term = self.term
        self.renewed_at = time.monotonic()  # before is_leader: never seen as a frozen leader
        self.is_leader = True
        self.takeover = None  # cleared after is_leader is set: the watcher never sees both unset
        self.store_epoch += 1
        self._start_lease(self.term)
        self.reannounce()
        old = t["old"]
        expected = {int(p) for p in t["state"].get("map", {}).values()} - {old, self.phys}
        deadline = time.monotonic() + self.wait_deadline()
        while time.monotonic() < deadline and not all(self.store.check([f"reg/{self.term}/{p}"]) for p in expected):
            time.sleep(self.conf.hb_interval)
        self.known_lost |= {old} | {p for p in expected if not self.store.check([f"reg/{self.term}/{p}"])}
        self.hb_seen = {}
        if any(self.store.check([f"fin/{self.term}/{p}"]) for p in expected | {self.phys}):
            self.finished_locally = True  # the old leader died while the run was finishing: one finished, all did
            return True
        gen = self.gen()
        mapped = self.mapped()
        logical = next((lg for lg, p in mapped.items() if p == old), None)
        reason = f"the leader of term {self.term - 1} (physical node {old}) was lost"
        if logical is None:  # the old leader trained nothing: just start the next generation
            self.set_mode(gen + 1, "memory")
            self.store.set("gen", str(gen + 1))
            return True
        self.store.set(f"spare/{old}", "lost")
        return self.replace(gen, logical, old, reason, "memory") == "restart"

    def report_free(self, gen: int, logical: int):
        """This node released its GPUs for generation gen; say whether it can restore from its own copies."""
        if not self.has_own_copies(logical):
            self.store.set(f"fresh/{gen}/{logical}", "1")
        if not self.holds_copies(logical):
            self.store.set(f"nocopy/{gen}/{logical}", "1")
        self.store.set(f"free/{gen}/{logical}", "1")

    # ------------------------------------------------------------------ main loop
    def finish(self) -> int:
        self.finished_locally = True  # trainers end in lockstep: every rank reached the last step
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

    def finished_marker(self) -> Path:
        job, restart = os.environ["FT_EPOCH"].split(":")
        return Path(self.conf.dump_dir) / f"ft_finished.{job}.{restart}"

    def write_finished_marker(self):
        """The sbatch script requeues the job unless this file exists (scripts/ft_train.sbatch)."""
        marker = self.finished_marker()
        try:
            marker.write_text(f"finished at generation {self.get_or('gen', '?')}\n")  # not gen(): no fence here
        except OSError as e:  # the run did finish; without the marker the script requeues once more, from disk
            logger.error(f"ft leader: could not write {marker} ({e})")

    def wait_finished(self):
        while not self.store.check(["finished"]):  # raises if the leader is gone before it finished
            if self.finished_marker().exists():  # the leader wrote it and is gone: the run did finish
                return
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
            # a check passed before a takeover counts (a fresh store knows nothing of it)
            healthy = (status == b"" and self.spare_ready) or not self.conf.spare_health or (
                not self.store.check([f"fail_health/{self.phys}"])
                and run_spare_check(self.gpus, SPARE_CHECK_TIMEOUT, self.store_port + 10000 + self.phys))
            if not healthy:
                self.store.set(f"spare/{self.phys}", "barred")
                logger.warning(f"ft agent {self.phys}: spare health check failed; this node leaves the run")
                return None
            self.spare_ready = True
            self.store.set(f"spare/{self.phys}", "ready")
        self.spare_ready = True
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
                try:
                    logical = self.my_logical()
                    if logical is None:  # a spare (or a fenced node that was declared dead): wait in the pool
                        gen = self.park()
                        if gen is None:
                            return EXIT_FINISHED
                        continue
                    if gen == 0 and socket.gethostname() in self.barred():
                        # barred in an earlier run of this job and given back by SLURM: replace it right away
                        if self.is_leader:  # the leader's node cannot be handed over: end before any trainer starts
                            host = socket.gethostname()
                            self.store.set("end", f"the leader's node {host} was barred in an earlier run")
                            return EXIT_REQUEUE
                        self.bad_reason = "barred in an earlier run of this job"
                        self.store.set(f"bad/{self.phys}", self.bad_reason)
                        self.request(gen, f"physical node {self.phys} is barred")
                        while self.my_logical() == logical and not self.store.check(["end"]):  # no trainers here
                            time.sleep(self.conf.hb_interval)
                        if self.store.check(["end"]):
                            return EXIT_REQUEUE
                        continue  # replaced: park() refuses the barred node
                    if not self.wait_go(gen):
                        return EXIT_REQUEUE
                    if self.gen() > gen:  # moved on before it started (a takeover): free at once, then follow it
                        gen = self.gen()
                        if self.my_logical() == logical:
                            self.report_free(gen, logical)
                        continue
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
                    self.report_free(gen, logical)
                except (LeaderLost, DistError, ConnectionError) as e:  # follow (or become) the next leader
                    logger.warning(f"ft agent {self.phys}: {type(e).__name__}: {e}")
                    running = gen
                    if not self.rejoin():
                        logger.error(f"ft agent {self.phys}: no leader to follow, or the new leader ended the step")
                        return EXIT_REQUEUE
                    if self.finished_locally:  # the leader was lost during the finish: end the run, train nothing
                        if self.is_leader:
                            self.write_finished_marker()
                            self.store.set("finished", "1")
                        else:
                            self.store.add("done", 1)
                            self.wait_finished()
                        logger.info(f"ft agent {self.phys}: run finished")
                        return EXIT_FINISHED
                    if not self.is_leader and self.my_logical() is not None:
                        # this node's trainers are gone: never back into the generation that was running
                        # the same leader after a glitch, or a new one not yet started (a node that must go is
                        # replaced anyway: nothing to ask)
                        if self.gen() <= running and not self.bad_reason:
                            self.request(running, f"physical node {self.phys} lost the control store")
                        while self.gen() <= running:
                            if self.store.check(["finished"]) or self.finished_marker().exists():
                                logger.info(f"ft agent {self.phys}: run finished")
                                return EXIT_FINISHED
                            if self.store.check(["end"]):
                                return EXIT_REQUEUE
                            time.sleep(self.conf.hb_interval)
                    gen, logical = self.gen(), self.my_logical()
                    if logical is not None:  # still a training node: its GPUs are free for the next generation
                        if not self.release_poll():
                            self.store.set("end", f"physical node {self.phys}: GPU memory not released")
                            return EXIT_REQUEUE
                        self.report_free(gen, logical)
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
    agent.mark_ended(code)
    if agent.refill:  # a refill spare's own 1-node job just ends; only the main job's script decides requeues
        code = EXIT_FINISHED
    # exit without running C++ destructors of the store objects (the leader's store may already be gone)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
