from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os
import re
from logging import getLogger

from xllm.data.dataset_streamer.data_iterator.reader import JSONLReader
from xllm.data.data_types import Instance

logger = getLogger()


class DataIterator(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __next__(self) -> Instance:
        ...

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def set_state(self, state: Optional[Dict[str, Any]]):
        ...

    @abstractmethod
    def close(self):
        ...


class JSONLFileIterator(DataIterator):
    """A thin synchronous wrapper around JSONLReader."""

    def __init__(
        self,
        fpath: str,
        world_rank: int,
        world_size: int,
    ):
        self.fpath = fpath
        self.world_rank = world_rank
        self.world_size = world_size
        self.reader = JSONLReader(fpath, world_rank, world_size)

    def start(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> Instance:
        return next(self.reader)

    def get_state(self) -> Dict[str, Any]:
        return {
            "fpath": self.fpath,
            "world_rank": self.world_rank,
            "world_size": self.world_size,
            "reader": self.reader.get_state(),
        }

    def set_state(self, state: Optional[Dict[str, Any]]):
        if not state:
            return

        assert self.fpath == state["fpath"]
        assert self.world_rank == state["world_rank"]
        assert self.world_size == state["world_size"]
        self.reader.set_state(state["reader"])

    def close(self):
        self.reader.close()


class JSONLFolderIterator(DataIterator):
    """
    Given a directory, iterate over all chunked JSONL files in order.
    Returns the records assigned to this rank across all files.

    This iterator intentionally does not maintain a per-source raw prefetch
    queue. The BufferAssembler owns token-budgeted prefetching; keeping another
    fixed-size raw queue here can over-read low-weight datasets and small files.
    """

    def __init__(
        self,
        fdir: str,
        world_rank: int,
        world_size: int,
        infinite: bool = False,
    ):
        self.fdir = fdir
        self.world_rank = world_rank
        self.world_size = world_size
        self.infinite = infinite
        self._assign_data(fdir)
        self.file_idx = 0
        self.repetition = 0
        self.reader: Optional[JSONLReader] = None

    def _assign_data(self, path: str):
        path = path.strip()
        assert os.path.isdir(path), path
        fnames = [x for x in os.listdir(path) if re.fullmatch(r".*chunk\.?\d+.*\.jsonl", x)]
        all_files = [os.path.join(path, fname) for fname in sorted(fnames)]
        if not all_files:
            raise FileNotFoundError(f"No chunk JSONL files found in directory: {path}")

        num_files = len(all_files)

        if self.world_size >= num_files and self.world_size % num_files == 0:
            ranks_per_file = self.world_size // num_files
            file_idx = self.world_rank // ranks_per_file
            self.files = [all_files[file_idx]]
            self.reader_world_rank = self.world_rank % ranks_per_file
            self.reader_world_size = ranks_per_file
            self.assignment_mode = "rank_group_per_file"
        elif num_files > self.world_size and num_files % self.world_size == 0:
            files_per_rank = num_files // self.world_size
            start_idx = self.world_rank * files_per_rank
            end_idx = start_idx + files_per_rank
            self.files = all_files[start_idx:end_idx]
            self.reader_world_rank = 0
            self.reader_world_size = 1
            self.assignment_mode = "file_group_per_rank"
        else:
            self.files = all_files
            self.reader_world_rank = self.world_rank
            self.reader_world_size = self.world_size
            self.assignment_mode = "all_files_modulo_rank"

    def start(self):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> Instance:
        completed_repetitions_without_sample = 0
        while True:
            if self.file_idx >= len(self.files):
                if self.infinite and len(self.files) > 0:
                    self.file_idx = 0
                    self.repetition += 1
                    self.reader = None
                    completed_repetitions_without_sample += 1
                    if completed_repetitions_without_sample >= 2:
                        raise RuntimeError(
                            f"No readable records were assigned to rank {self.world_rank}/{self.world_size} "
                            f"in {self.fdir!r} across a complete repetition"
                        )
                else:
                    raise StopIteration

            if self.reader is None:
                self.reader = JSONLReader(
                    self.files[self.file_idx],
                    self.reader_world_rank,
                    self.reader_world_size,
                )
                logger.debug(f"Starting iteration {self.repetition} over {self.reader.fpath} ...")

            try:
                inst = next(self.reader)
                inst.repetition = self.repetition
                return inst
            except StopIteration:
                self.reader.close()
                self.reader = None
                self.file_idx += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "fdir": self.fdir,
            "world_rank": self.world_rank,
            "world_size": self.world_size,
            "reader_world_rank": self.reader_world_rank,
            "reader_world_size": self.reader_world_size,
            "assignment_mode": self.assignment_mode,
            "infinite": self.infinite,
            "file_idx": self.file_idx,
            "repetition": self.repetition,
            "reader": self.reader.get_state() if self.reader is not None else None,
        }

    def set_state(self, state: Optional[Dict[str, Any]]):
        if not state:
            return
        if self.reader is not None:
            self.reader.close()
            self.reader = None

        assert self.fdir == state["fdir"]
        assert self.world_rank == state["world_rank"]
        assert self.world_size == state["world_size"]
        assert self.reader_world_rank == state.get("reader_world_rank", self.reader_world_rank)
        assert self.reader_world_size == state.get("reader_world_size", self.reader_world_size)
        assert self.infinite == state["infinite"]

        self.file_idx = state["file_idx"]
        self.repetition = int(state.get("repetition", 0))
        reader_state = state.get("reader")
        if reader_state is not None:
            self.reader = JSONLReader(
                self.files[self.file_idx],
                self.reader_world_rank,
                self.reader_world_size,
            )
            self.reader.set_state(reader_state)

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
