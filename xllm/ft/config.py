"""
Settings of xLLM's in-band fault tolerance (ft, "hot swap"): TrainerConf.ft, given as --ft.<name> on the command line.
Off by default; none of them is used when enabled is False.
"""

from dataclasses import dataclass
from typing import Optional

from xllm.configuration import Config


@dataclass
class FtConf(Config):
    enabled: bool = False  # Whether trainers run under a per-node agent that restarts them inside the allocation from in-memory copies.
    mtbf_sys: Optional[float] = None  # Expected seconds between failures of the whole job; required when enabled. Sets the auto copy interval.
    k: int = 0  # Steps between in-memory copies. 0 = auto from the measured copy cost, step time and mtbf_sys.
    hb_deadline: float = 0.0  # Seconds without trainer progress before a hang is declared. 0 = auto from measured step, eval and checkpoint times.
    hb_margin: float = 0.5  # Margin m applied to measured times in the auto deadlines: deadline = (1 + m) * measured.
    hb_interval: float = 0.2  # Seconds between agent heartbeats.
    node_deadline: float = 10.0  # Seconds without heartbeats before a node is declared dead.
    release_timeout: float = 0.0  # Seconds to wait for GPU memory to be released after a kill. 0 = auto from the previous teardown.
    xid_replace: str = "48,63,64,74,79,92,94,95,119,120"  # Comma separated critical Xids after which the node is replaced by a spare.
    dump_enabled: bool = True  # Whether agents write background disk dumps from the in-memory copies.
    dump_keep: int = 2  # Number of committed agent dumps to keep.
    dump_dir: Optional[str] = None  # Root of the agent dumps. None = <dump_dir>/ft_checkpoints.
    refill_wait_s: float = 0.0  # Seconds to wait for a refill node before requeueing. 0 = auto from the first measured refill.
    spare_health: bool = True  # Whether spares pass a local GPU and NCCL health check before admission.
    spares: int = 0  # Number of the allocated nodes kept as idle spares (the last ones). 0 = none: a lost node is not replaced.
    retries: int = 2  # Failed recoveries from memory before falling back to disk.
    lease_ttl: float = 60.0  # Seconds after which a standby takes over from a leader whose lease stopped being renewed.
    lease_renew: float = 6.0  # Seconds between the leader's lease renewals (and state flushes) on the shared FS.
    slot_dir: str = "/dev/shm/xllm_ft"  # Directory holding the in-memory copies (must be RAM-backed).
    local_log_dir: str = "/tmp/xllm_ft_logs"  # Per-host local directory for logs and metrics when enabled (a job subfolder is used).

    def __post_init__(self):
        if self.enabled:
            assert self.mtbf_sys is not None and self.mtbf_sys > 0, "ft.enabled requires ft.mtbf_sys > 0"
            assert self.k >= 0 and self.spares >= 0
            assert self.retries >= 0 and self.dump_keep >= 1
            assert self.hb_deadline >= 0 and self.hb_margin >= 0 and self.hb_interval > 0
            assert self.node_deadline > 0 and self.release_timeout >= 0 and self.refill_wait_s >= 0
            assert 0 < self.lease_renew < self.lease_ttl / 4, "ft.lease_renew must be well under ft.lease_ttl"
            assert all(x.isdigit() for x in self.xid_replace.split(",") if x), "ft.xid_replace: comma separated Xids"
