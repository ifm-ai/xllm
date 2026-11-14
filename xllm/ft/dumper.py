"""
Agent-side disk dumps of the in-memory copies (ft).

The agents write durable checkpoints from their nodes' RAM slots, off the GPUs
and off the training thread. The leader picks the dump step: the newest step
for which every logical node holds an own copy. Each agent copies its ranks'
slots of that step into a staging dir (invisible to the checkpoint globs),
fsyncs every file and writes one marker per rank with the run, generation,
step, size and crc32. When every rank's marker is there, the leader writes
config.json and renames the staging dir to checkpoint_<step>: a dump is either
complete or invisible. The next dump starts when the previous one is
committed, so the cadence follows the write bandwidth and training never
waits. Dumps live in their own root, separate from xLLM's checkpoints.
"""

import json
import logging
import os
import shutil
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

from xllm.ft.replica import dump_files, lease_path, read_meta, run_epoch, slot_path

logger = logging.getLogger()

CHUNK = 64 * 2**20


def staging_dir(root: Path, gen: int, step: int) -> Path:
    job, restart = run_epoch()
    return root / "checkpoints" / f".staging-{job}.{restart}-{gen}-{step:08d}"


def bad_dumps(root: Path) -> set:
    """Dumps that failed to load, persisted across requeues as <root>/.bad lines '<dir name> <step>'."""
    try:
        lines = (root / ".bad").read_text().splitlines()
    except FileNotFoundError:
        return set()
    bad = set()
    for line in lines:  # a torn last line (the writer died) is skipped
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            bad.add((parts[0], int(parts[1])))
    return bad


def mark_bad(root: Path, dump_dir: Path):
    """Best effort: on a shared-FS error the dump stays a candidate, and the retry accounting bounds the retries."""
    try:
        step = json.loads((dump_dir / "config.json").read_text())["step"] if (dump_dir / "config.json").exists() else -1
        with open(root / ".bad", "a") as f:
            f.write(f"{dump_dir.name} {step}\n")
    except OSError as e:
        logger.error(f"ft: could not mark the agent dump {dump_dir} bad ({e})")
        return
    logger.error(f"ft: agent dump {dump_dir} marked bad")


def committed_dumps(root: Path, world: Optional[int]) -> List[Tuple[int, Path]]:
    """Complete dumps for this world size (None: any), newest first."""
    out = []
    for d in (root / "checkpoints").glob("checkpoint_*"):
        config = d / "config.json"
        if config.exists():
            try:
                info = json.loads(config.read_text())
                step, dump_world = int(info["step"]), int(info["world"])
            except (ValueError, KeyError, TypeError):
                logger.warning(f"ft: {config} is unreadable; the dump is skipped")
                continue
            if world is None or dump_world == world:
                out.append((step, d))
    return sorted(out, reverse=True)


def own_slot_of_step(slot_dir: str, rank: int, step: int) -> Optional[Path]:
    for slot in (0, 1):
        path = slot_path(slot_dir, "own", rank, slot)
        meta = read_meta(path)
        if meta is not None and meta["epoch"] == run_epoch() and meta["rank"] == rank and meta["step"] == step:
            return path
    return None


def newest_common_own_step(slot_dir: str, ranks: List[int]) -> Optional[int]:
    """The newest step for which every one of these ranks holds a valid own copy."""
    newest = None
    for rank in ranks:
        steps = []
        for slot in (0, 1):
            meta = read_meta(slot_path(slot_dir, "own", rank, slot))
            if meta is not None and meta["epoch"] == run_epoch() and meta["rank"] == rank:
                steps.append(meta["step"])
        if not steps:
            return None
        newest = max(steps) if newest is None else min(newest, max(steps))
    return newest


def _fsync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_rank(slot_dir: str, staging: Path, rank: int, gen: int, step: int, abort=None) -> bool:
    """
    Copies one rank's own slot of this step into the staging dir, under a lease taken before the slot is
    checked, so the trainer does not overwrite it meanwhile; False if the slot already moved on. `abort` (a
    threading.Event, set by the dump_kill test injection) stops the copy halfway with an OSError.
    """
    leases = [lease_path(slot_path(slot_dir, "own", rank, slot)) for slot in (0, 1)]
    for lease in leases:
        lease.touch()
    try:
        return _write_rank(slot_dir, staging, rank, gen, step, abort)
    finally:
        for lease in leases:
            lease.unlink(missing_ok=True)


def _write_rank(slot_dir: str, staging: Path, rank: int, gen: int, step: int, abort=None) -> bool:
    src = own_slot_of_step(slot_dir, rank, step)
    if src is None:
        return False
    staging.mkdir(parents=True, exist_ok=True)
    payload, meta_file, ok_file = dump_files(staging, rank)
    meta = src.with_suffix(".meta").read_bytes()
    crc, size = 0, 0
    with open(src, "rb") as fin, open(payload, "wb") as fout:
        while chunk := fin.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
            fout.write(chunk)
            if abort is not None and abort.is_set():
                abort.clear()
                raise OSError("dump aborted halfway (test injection)")
        fout.flush()
        os.fsync(fout.fileno())
    if own_slot_of_step(slot_dir, rank, step) is None:  # overwritten by a newer copy while we read it
        return False
    with open(meta_file, "wb") as f:
        f.write(meta)
        f.flush()
        os.fsync(f.fileno())
    job, restart = run_epoch()
    with open(ok_file, "w") as f:
        json.dump({"epoch": f"{job}.{restart}", "gen": gen, "step": step, "bytes": size, "crc32": crc,
                   "meta_crc32": zlib.crc32(meta)}, f)
        f.flush()
        os.fsync(f.fileno())
    return True


def commit(root: Path, staging: Path, gen: int, step: int, world: int, keep: int, protect: str = "") -> bool:
    """Leader: publish a complete dump, then drop the oldest beyond `keep` (never `protect`, a dump in use)."""
    job, restart = run_epoch()
    for rank in range(world):
        payload, _, ok_file = dump_files(staging, rank)
        if not ok_file.exists():
            return False
        ok = json.loads(ok_file.read_text())
        if (ok["epoch"], ok["gen"], ok["step"]) != (f"{job}.{restart}", gen, step) or payload.stat().st_size != ok["bytes"]:
            return False
    with open(staging / "config.json", "w") as f:
        json.dump({"step": step, "world": world, "epoch": f"{job}.{restart}"}, f)
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(staging)
    target = staging.parent / f"checkpoint_{step:08d}"
    if target.exists():  # a replay reached a step that was already dumped: keep the complete one
        shutil.rmtree(staging)
        return True
    os.replace(staging, target)
    _fsync_dir(staging.parent)
    for _, old in committed_dumps(root, None)[keep:]:  # across world sizes: older runs' dumps age out too
        if str(old) != protect:
            shutil.rmtree(old, ignore_errors=True)
    logger.info(f"ft leader: agent dump of step {step} committed ({target})")
    return True


def clean_staging(root: Path):
    """Leader at start: staging dirs of earlier runs or interrupted dumps are never used."""
    for d in (root / "checkpoints").glob(".staging-*"):
        shutil.rmtree(d, ignore_errors=True)


def newest_xllm_checkpoint_step(dump_dir: str) -> int:
    steps = [int(d.name.split("_")[-1]) for d in Path(dump_dir, "checkpoints").glob("checkpoint_*")
             if d.name.split("_")[-1].isdigit() and (d / "config.json").exists()]
    return max(steps, default=-1)
