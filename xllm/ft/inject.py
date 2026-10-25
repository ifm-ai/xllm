"""
Fault-injection requests for the ft tests. A request is written to the ft
control store; the per-node agent reads the ones that target its node and
carries them out, so injections need no access to the nodes themselves.
The command-line tool that publishes requests is tests/ft_inject.py.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

logger = logging.getLogger()

COUNT_KEY = "ft/inject/count"
REQUEST_KEY = "ft/inject/{}"

# kind -> required target field ("rank" or "node")
KINDS = {
    "kill_rank": "rank",  # SIGKILL one trainer
    "kill_process": "node",  # SIGKILL the agent process of a node (its trainers die with it)
    "kill_node": "node",  # SIGKILL the agent and its trainers together (killpg), as a node death
    "hang_rank": "rank",  # SIGSTOP one trainer inside a collective
    "torn_write": "rank",  # the trainer dies halfway through its next slot write
    "bitflip": "rank",  # flip one bit in the rank's payload of the newest committed agent dump
    "xid": "node",  # deliver a critical Xid event at the agent's NVML event wrapper
    "gpu_lost": "node",  # the agent's NVML wrapper raises NVML_ERROR_GPU_IS_LOST
    "fail_spare_health": "node",  # the next spare health gate on the node fails
    "dump_kill": "node",  # the agent's dumper is killed in the middle of a write
    "lease_cut": "node",  # every later lease read and write of the agent fails, as if cut off from the shared FS
}


@dataclass
class Injection:
    kind: str
    rank: Optional[int] = None
    node: Optional[int] = None
    step: Optional[int] = None  # act when training reaches this step; None = at once
    xid: Optional[int] = None
    gpu: int = 0
    index: int = -1  # position in the store, set by the reader

    def __post_init__(self):
        assert self.kind in KINDS, f"unknown injection kind: {self.kind}"
        target = KINDS[self.kind]
        assert getattr(self, target) is not None, f"{self.kind} needs --{target}"
        if self.kind == "xid":
            assert self.xid is not None, "xid needs an Xid number"

    def targets(self, node: int, ranks: List[int]) -> bool:
        if KINDS[self.kind] == "node":
            return self.node == node
        return self.rank in ranks

    def to_json(self) -> str:
        fields = asdict(self)
        fields.pop("index")  # where it sits in the store is not part of the request
        return json.dumps(fields)

    @staticmethod
    def from_json(text: str) -> "Injection":
        return Injection(**json.loads(text))


def publish(store, injection: Injection) -> int:
    """Write one request to the store and return its index."""
    index = store.add(COUNT_KEY, 1) - 1
    store.set(REQUEST_KEY.format(index), injection.to_json())
    logger.info(f"Published injection {index}: {injection}")
    return index


@dataclass
class InjectionReader:
    """Reads the requests published since the last poll that target one node."""

    node: int
    ranks: List[int]
    next_index: int = 0
    pending: List[Injection] = field(default_factory=list)
    missing_since: Optional[float] = None
    missing_timeout: float = 30.0  # a request still unwritten after this long is skipped (its publisher died)
    last_raw: bytes = b""

    def poll(self, store, now: Optional[float] = None) -> List[Injection]:
        now = time.time() if now is None else now
        count = store.add(COUNT_KEY, 0)
        last_key = REQUEST_KEY.format(self.next_index - 1)
        restarted = count < self.next_index or (
            self.next_index > 0 and (not store.check([last_key]) or store.get(last_key) != self.last_raw)
        )
        if restarted:  # the last request read is gone or changed: the store was restarted; start over, loudly
            logger.error(f"Injection store was restarted (read {self.next_index}, counter {count}); reading from 0 again")
            self.next_index, self.missing_since = 0, None
        while self.next_index < count:
            key = REQUEST_KEY.format(self.next_index)
            # the counter is bumped before the request is written
            if not store.check([key]):
                self.missing_since = self.missing_since or now
                if now - self.missing_since < self.missing_timeout:
                    break
                logger.error(f"Injection {self.next_index} was never written; skipping it")
                self.next_index, self.missing_since = self.next_index + 1, None
                continue
            self.missing_since = None
            self.last_raw = store.get(key)
            injection = Injection.from_json(self.last_raw.decode())
            injection.index = self.next_index
            self.next_index += 1
            if injection.targets(self.node, self.ranks):
                self.pending.append(injection)
        return self.pending

    def due(self, step: int, store=None) -> List[Injection]:
        """
        Remove and return the pending requests whose step has been reached. With a store, each request is
        carried out once in the whole job: a node that takes over a target later does not repeat it.
        """
        ready = [i for i in self.pending if i.step is None or i.step <= step]
        self.pending = [i for i in self.pending if i not in ready]
        if store is not None:
            ready = [i for i in ready if store.add(f"ft/inject/claimed/{i.index}", 1) == 1]
        return ready
