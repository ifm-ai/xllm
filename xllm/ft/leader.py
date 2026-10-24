"""
The ft leader's lease and state on the shared filesystem, so that a standby agent can take the leader's place.

One folder per run of the job (<dump_dir>/ft_leader/<job>.<restart>) holds:
  leader         where the current leader's control store is: {"term", "host", "port", "standby"}
  claim.<term>   the immutable claim of a term (link(2): exactly one agent wins it)
  renew.<term>   the leader's renewal counter, rewritten every lease_renew seconds
  state.json     the leader's decisions (the map, the generation, the retry counters, ...), the journal
On NFSv3 (attribute and lookup caches can be stale for seconds) every file is replaced by an atomic rename and
read by open() + read(): never by mtime, stat or a directory listing. Liveness is a counter that keeps changing,
judged on the reader's own monotonic clock, never a timestamp compared across nodes.
"""

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def lease_dir(dump_dir: str, job: str, restart: str) -> Path:
    return Path(dump_dir) / "ft_leader" / f"{job}.{restart}"


def _write(path: Path, text: str):
    # per thread: two threads of one agent (the lease thread, a step-down) may write the same file at once
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(path: Path) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


class Lease:
    """The lease files of one run; every call is best effort and never raises on a filesystem error."""

    def __init__(self, folder: Path):
        self.folder = folder

    def claim(self, term: int) -> bool:
        """Claim a term: True for exactly one caller, whatever the others do at the same time."""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            mine = self.folder / f".claim.{term}.{socket.gethostname()}.{os.getpid()}"
            mine.write_text(socket.gethostname())
            try:
                os.link(mine, self.folder / f"claim.{term}")
                return True
            except FileExistsError:
                return False
            finally:
                mine.unlink(missing_ok=True)
        except OSError:
            return False

    def renew(self, term: int, counter: int) -> bool:
        try:
            _write(self.folder / f"renew.{term}", str(counter))
            return True
        except OSError:
            return False

    def counter(self, term: int) -> Optional[int]:
        text = _read(self.folder / f"renew.{term}")
        try:
            return int(text) if text is not None else None
        except ValueError:
            return None

    def publish(self, leader: Dict[str, Any]) -> bool:
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            _write(self.folder / "leader", json.dumps(leader))
            return True
        except OSError:
            return False

    def leader(self) -> Optional[Dict[str, Any]]:
        text = _read(self.folder / "leader")
        try:
            return json.loads(text) if text else None
        except ValueError:  # a torn write is impossible (a rename of a per-writer file), a foreign file is not
            return None

    def save_state(self, state: Dict[str, Any]) -> bool:
        try:
            _write(self.folder / "state.json", json.dumps(state))
            return True
        except OSError:
            return False

    def state(self) -> Optional[Dict[str, Any]]:
        text = _read(self.folder / "state.json")
        try:
            return json.loads(text) if text else None
        except ValueError:
            return None


class Liveness:
    """A leader is alive while its renewal counter keeps changing, on this reader's own monotonic clock."""

    def __init__(self):
        self.seen: Optional[int] = None
        self.since = time.monotonic()

    def silent_for(self, counter: Optional[int]) -> float:
        now = time.monotonic()
        if counter is not None and counter != self.seen:
            self.seen, self.since = counter, now
        return now - self.since
