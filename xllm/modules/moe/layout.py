from dataclasses import dataclass

from xllm.distributed import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_expert_outer_rank,
    get_expert_outer_world_size,
)
from xllm.distributed.utils import divide_and_check_no_remainder


@dataclass(frozen=True)
class ExpertLayout:
    """Which routed experts a GPU owns.

    EP = TP x E_out: E_out consecutive data parallel replicas form an expert group. The TP ranks split
    the experts as without EP, and the E_out replicas split each TP rank's share again, so the GPU at
    TP rank tp_rank in replica outer_rank owns the n_local experts [start, end).
    """
    n_experts: int
    tp: int
    e_out: int = 1
    tp_rank: int = 0
    outer_rank: int = 0

    def __post_init__(self):
        divide_and_check_no_remainder(self.n_experts, self.tp * self.e_out)
        assert 0 <= self.tp_rank < self.tp, f"{self.tp_rank}/{self.tp}"
        assert 0 <= self.outer_rank < self.e_out, f"{self.outer_rank}/{self.e_out}"

    @classmethod
    def from_process_groups(cls, n_experts: int) -> "ExpertLayout":
        """The layout of the calling rank."""
        return cls(
            n_experts,
            tp=get_model_parallel_world_size(),
            e_out=get_expert_outer_world_size(),
            tp_rank=get_model_parallel_rank(),
            outer_rank=get_expert_outer_rank(),
        )

    @property
    def n_local(self) -> int:
        return self.n_experts // (self.tp * self.e_out)

    @property
    def start(self) -> int:
        return self.tp_rank * (self.n_experts // self.tp) + self.outer_rank * self.n_local

    @property
    def end(self) -> int:
        return self.start + self.n_local
