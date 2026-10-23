"""
Resources of the local node, discovered at run time so that no cluster constant
is needed: GPU count, size of the RAM-backed filesystem holding the in-memory
copies, and physical RAM.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class NodeResources:
    gpus: int
    shm_bytes: int  # size of the filesystem holding the slot directory
    shm_free_bytes: int
    ram_bytes: int


def existing_ancestor(path: Path) -> Path:
    path = path.absolute()
    while not path.exists():
        path = path.parent
    return path


def discover_node(slot_dir: str) -> NodeResources:
    st = os.statvfs(existing_ancestor(Path(slot_dir)))
    return NodeResources(
        gpus=torch.cuda.device_count(),
        shm_bytes=st.f_frsize * st.f_blocks,
        shm_free_bytes=st.f_frsize * st.f_bavail,
        ram_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    )
