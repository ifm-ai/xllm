"""
Scheduler client of the ft agent. Requests and releases of refill-spare nodes go
through a mailbox directory served by the host-side helper
scripts/ft_slurm_helper.sh, because the agent runs inside the container where the
SLURM commands may not be available. A shared-filesystem outage only blocks
requests and releases.
"""

import os
import time
import uuid
from pathlib import Path
from typing import List, Optional


def mailbox_dir(dump_dir: str) -> Path:
    return Path(dump_dir) / "ft_slurm" / os.environ.get("SLURM_JOB_ID", "local")


class SlurmMailbox:
    def __init__(self, mailbox: Path, timeout: float = 120.0):
        self.mailbox = mailbox
        self.timeout = timeout
        mailbox.mkdir(parents=True, exist_ok=True)

    def _ask(self, line: str) -> str:
        rid = uuid.uuid4().hex
        tmp = self.mailbox / f"tmp.{rid}"
        tmp.write_text(line + "\n")
        os.replace(tmp, self.mailbox / f"req.{rid}")  # the helper only sees complete requests
        reply = self.mailbox / f"rep.{rid}"
        t0 = time.monotonic()
        while not reply.exists():
            if time.monotonic() - t0 > self.timeout:
                (self.mailbox / f"req.{rid}").unlink(missing_ok=True)  # the helper must not act on it later
                raise TimeoutError(f"no answer from the SLURM helper to '{line}' after {self.timeout:.0f} s")
            time.sleep(0.2)
        answer = reply.read_text().strip()
        reply.unlink()
        if not answer.startswith("ok "):
            raise RuntimeError(f"SLURM helper: {answer}")
        return answer[3:]

    def request_node(self, exclude: Optional[List[str]] = None) -> str:
        """Submits a 1-node refill-spare job, never on the excluded (barred) hosts; returns its job id."""
        return self._ask(f"request_node {','.join(exclude) if exclude else '-'}")

    def release_node(self, jobid: str):
        """Cancels a refill-spare job (static spares are barred and parked instead, never cancelled)."""
        self._ask(f"release_node {jobid}")
