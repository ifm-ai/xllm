from typing import Any, List, Optional
from dataclasses import dataclass
import numpy as np

# -------------------------
# Core data types
# -------------------------
@dataclass
class Instance:
    # Filled by Reader
    raw_data: Any
    filename: str
    file_pos: int
    line_num: int
    repetition: int = 0

    # Ready after Tokenizer. The packer performs the final x/y shift after packing.
    tokens: Optional[List[int]] = None
    target_mask: Optional[List[bool]] = None


@dataclass
class SourceInfo:
    """Records where a sub-sequence inside a PackedSeq came from."""
    dataset: str        # dataset directory (top-level mix key)
    filename: str       # actual JSONL file path
    line_num: int       # line number within that file
    is_truncation: bool = False  # True when a long doc was force-split across bins (BFP)

@dataclass
class PackedSeq:
    x: np.ndarray
    y: np.ndarray
    mask: Optional[np.ndarray] = None
    src_infos: Optional[List[SourceInfo]] = None   # one entry per contributing Instance
    padding_count: int = 0
    has_truncation: bool = False


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    mask: Optional[np.ndarray] = None
    position_ids: Optional[np.ndarray] = None
    src_names: Optional[List[List[str]]] = None
    src_infos: Optional[List[List[SourceInfo]]] = None  # [batch_idx][source_idx]
    padding_ratio: float = -1.0
    truncation_ratio: float = -1.0

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert self.mask is None or (self.mask.shape == self.x.shape and self.mask.dtype == bool)
        assert self.position_ids is None or self.position_ids.shape == self.x.shape
        assert self.src_names is None or len(self.src_names) == len(self.x)
        assert self.src_infos is None or len(self.src_infos) == len(self.x)
