from typing import Iterator, Dict, Optional, Any
from abc import abstractmethod
import os
import json
from logging import getLogger

from xllm.data.data_types import Instance

logger = getLogger()


class Reader:
    """An iterator that reads one file and yield one line of string each time."""
    def __init__(self, fpath: str):
        self.fpath = fpath
        self.line_num = 0
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.read())
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.read())
        return next(self._iterator)

    @abstractmethod
    def read(self) -> Iterator[Instance]:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class JSONLReader(Reader):
    def __init__(self, fpath: str, world_rank: int, world_size: int):
        super().__init__(fpath)
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.world_rank = world_rank
        self.world_size = world_size
        self.f = None
        self.file_pos = 0
        self.line_num = 0
        self.open()

    def reset(self):
        self.file_pos = 0
        self.line_num = 0
        self.f.seek(0)

    def read(self) -> Iterator[Instance]:
        while True:
            cur_file_pos = self.file_pos
            cur_line_num = self.line_num
            line = self.f.readline()
            if not line:
                break

            self.file_pos += len(line)
            self.line_num += 1
            if cur_line_num % self.world_size == self.world_rank:
                try:
                    inst = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"{e}: Error when trying to decode line {cur_line_num} in {self.fpath}")
                    continue
                else:
                    yield Instance(
                        raw_data=inst,
                        filename=self.fpath,
                        file_pos=cur_file_pos,
                        line_num=cur_line_num,
                    )
        self.close()
        return

    def set_state(self, state: Optional[Dict[str, Any]]):
        if not state:
            return
        self.line_num = state["line_num"]
        self.file_pos = state["file_pos"]
        logger.warning(
            f"Setting JSONL position on {self.fpath} ({self.world_rank}/{self.world_size}): {self.file_pos}"
        )
        self.f.seek(self.file_pos)

    def get_state(self) -> Dict[str, Any]:
        return {
            "file_pos": self.file_pos,
            "line_num": self.line_num,
        }

    def open(self):
        assert self.fpath.endswith(".jsonl") and os.path.isfile(self.fpath), f"Invalid jsonl file: {self.fpath}"
        self.f = open(self.fpath, "rb")

    def close(self):
        if self.f is not None:
            try:
                self.f.close()
            finally:
                self.f = None
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
