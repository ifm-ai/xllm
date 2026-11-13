from typing import Any, Optional, Callable, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange

from xllm.utils import get_init_fn
from xllm.modules.fused_ops import memory_efficient_dropout
from xllm.modules.fused_ops import mgmm
from .layout import ExpertLayout


def _init_affine_weight(
    weight,
    layout: ExpertLayout,
    out_features: int,
    in_features: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
):
    assert weight.dim() == 2, f"weight dim is not 2, {weight.dim()}"

    # draw every expert in order and keep our own, so each expert's init does not depend on the layout
    my_weight_list = []
    for i in range(layout.n_experts):
        w = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
        init_method(w)
        if layout.start <= i < layout.end:
            my_weight_list.append(w)
        else:
            del w

    with torch.no_grad():
        assert len(my_weight_list) == layout.n_local
        torch.cat(my_weight_list, dim=0, out=weight)
    # clear master weights
    del my_weight_list


class _NoRows(torch.autograd.Function):
    """The experts' output when they receive no rows. The weights get zero gradients, so that FSDP reduces
    the same parameters on this rank as on the ranks whose experts received rows."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
        ctx.weight_specs = [(w.shape, w.dtype, w.device) for w in weights]
        return x.new_empty(x.shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return (grad, *(torch.zeros(shape, dtype=dtype, device=device) for shape, dtype, device in ctx.weight_specs))


class Expert(nn.Module):
    def __init__(
        self,
        model_dim: int,
        expert_inter_dim: int,
        num_experts: int,
        backend: str = 'sequential',
        hidden_dropout: float = 0.0,
        init_mode: str = 'gaussian',
        init_std: Optional[float] = None
    ):
        super().__init__()

        self.model_dim = model_dim
        self.inter_dim = expert_inter_dim
        self.n_experts = num_experts
        self.backend = backend
        self.hidden_dropout = hidden_dropout
        self.init_mode = init_mode
        self.init_std = init_std
        self.layout = ExpertLayout.from_process_groups(num_experts)
        self.n_local_experts = self.layout.n_local

        self.weight1 = Parameter(torch.empty(self.n_local_experts * self.inter_dim, self.model_dim))
        self.weight2 = Parameter(torch.empty(self.n_local_experts * self.model_dim, self.inter_dim))
        self.weight3 = Parameter(torch.empty(self.n_local_experts * self.inter_dim, self.model_dim))

        # init w1 & w3
        init_fn = get_init_fn(init_mode, dim=model_dim, std=init_std)
        _init_affine_weight(self.weight1, self.layout, self.inter_dim, self.model_dim, init_fn)
        _init_affine_weight(self.weight3, self.layout, self.inter_dim, self.model_dim, init_fn)
        # init w2
        init_fn = get_init_fn(init_mode, dim=expert_inter_dim, std=init_std)
        _init_affine_weight(self.weight2, self.layout, self.model_dim, self.inter_dim, init_fn)

    def forward(self, x: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:
        if x.size(0) == 0:
            return _NoRows.apply(x, self.weight1, self.weight2, self.weight3)
        w1 = rearrange(self.weight1, '(n s) d -> n s d', n=self.n_local_experts)
        w2 = rearrange(self.weight2, '(n s) d -> n s d', n=self.n_local_experts)
        w3 = rearrange(self.weight3, '(n s) d -> n s d', n=self.n_local_experts)
        h1 = F.silu(mgmm(x, w1, group_sizes, True, self.backend))
        h3 = mgmm(x, w3, group_sizes, True, self.backend)
        hidden = memory_efficient_dropout(h1 * h3, self.hidden_dropout, self.training)
        y = mgmm(hidden, w2, group_sizes, True, self.backend)
        return y

    def run_fused(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Return fn(*args, weight1, weight2, weight3), run as this module's forward.

        For a fused block that computes the experts inside: when the experts are their own FSDP unit,
        this method is registered as a forward method, so FSDP gathers the weights before fn, gathers
        them again before its backward (fn's output is this module's output), and reduces their
        gradients after it (once the gradients of the tensors in args are computed).
        """
        return fn(*args, self.weight1, self.weight2, self.weight3)

    def extra_repr(self) -> str:
        return 'mdim={}, edim={}, expert={} ({}), backend={}, init={} ({})'.format(
            self.model_dim, self.inter_dim, self.n_experts, self.n_local_experts, self.backend, self.init_mode, self.init_std
        )


def routed_expert_parameters(model: nn.Module) -> List[Parameter]:
    """The parameters of the routed experts of a model. With FSDP2 they are on the expert mesh, which is not the
    model's when EP > TP."""
    return [p for module in model.modules() if isinstance(module, Expert) for p in module.parameters()]
