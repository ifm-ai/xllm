# coding=utf-8

# Most parts of the code here are adapted from Fairscale
# repo: https://github.com/facebookresearch/fairscale


from typing import Callable, Optional, List
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from einops import rearrange

from xllm.distributed import get_model_parallel_rank, get_model_parallel_world_size
from xllm.distributed.utils import divide_and_check_no_remainder
from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
    gather_copy_model_parallel_region,
    reduce_scatter_model_parallel_region,
)
from xllm.modules.fused_ops import mgmm


def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    stride: int = 1,
    return_master_weight: bool = False,
) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        else:
            return None

    # Initialize master weight
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)

    if return_master_weight:
        return master_weight
    else:
        del master_weight
        del weight_list
        return None


def _initialize_group_weight(
    weight,
    out_features,
    in_features,
    per_partition_size,
    partition_dim: int,
    num_groups,
    transpose,
    init_method,
    return_master_weight,
) -> Optional[torch.Tensor]:
    assert weight.dim() == 3, f"weight dim is not 3, {weight.dim()}"
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        for i in range(num_groups):
            init_method(weight[i])
        if return_master_weight:
            return weight
        else:
            return None
    # Initialize master weight
    master_weight = torch.empty(num_groups, out_features, in_features, dtype=weight.dtype, requires_grad=False)
    for i in range(num_groups):
        init_method(master_weight[i])

    if transpose:
        master_weight = master_weight.transpose(1, 2).contiguous()

    # Split and copy
    weight_list = torch.split(master_weight, per_partition_size, dim=partition_dim)
    rank = get_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)

    if return_master_weight:
        return master_weight
    else:
        del master_weight
        del weight_list
        return None


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_emb: bool = False,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.normal_,
    ) -> None:
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_emb = scale_emb
        self.embedding_scale = math.sqrt(embedding_dim) if scale_emb else None
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.gather_output = gather_output
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        world_size = get_model_parallel_world_size()
        self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, world_size)

        # Allocate weights.
        self.weight = Parameter(torch.empty(self.num_embeddings, self.embedding_dim_per_partition))
        # And initialize.
        _initialize_affine_weight(
            self.weight,
            self.num_embeddings,
            self.embedding_dim,
            self.embedding_dim_per_partition,
            1,
            init_method,
            stride=1,
            return_master_weight=False,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = F.embedding(
            input_parallel,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if self.scale_emb:
            output_parallel = output_parallel * self.embedding_scale

        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim} ({embedding_dim_per_partition})'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        s += ', scale_emb={scale_emb}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ', gather_output={gather_output}'
        return s.format(**self.__dict__)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        input_is_parallel: If true, the input is a split of model parallel
        disable_input_reduce: If true, disable the reduce call in the backward
                              of the input tensor
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        disable_input_reduce: bool = False,
        gather_output: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.disable_input_reduce = disable_input_reduce
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.empty(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data, dim=0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            if self.disable_input_reduce:
                input_parallel = gather_from_model_parallel_region(input_)
            else:
                input_parallel = gather_copy_model_parallel_region(input_)
        else:
            if self.disable_input_reduce:
                input_parallel = input_
            else:
                input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={} ({}), bias={}, parallel_input={}, reduce_input={}, gather_output={}'.format(
            self.in_features, self.out_features, self.output_size_per_partition, self.bias is not None,
            self.input_is_parallel, not self.disable_input_reduce, self.gather_output
        )


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        parallel_output: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.parallel_output = parallel_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size) if parallel_output else out_features

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.empty(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        if self.parallel_output:
            # Reduce-scatter across all the partitions.
            output_ = reduce_scatter_model_parallel_region(output_parallel)
        else:
            # All-reduce across all the partitions.
            output_ = reduce_from_model_parallel_region(output_parallel)

        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_

        return output

    def extra_repr(self) -> str:
        return 'in_features={} ({}), out_features={} ({}), bias={}, parallel_input={}, parallel_output={}'.format(
            self.in_features, self.input_size_per_partition,
            self.out_features, self.output_size_per_partition,
            self.bias is not None, self.input_is_parallel, self.parallel_output
        )


class GroupColumnParallelLinear(torch.nn.Module):

    """Grouped linear layer with column parallelism.

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        num_groups: the number of groups
        input_is_parallel: If true, the input is a split of model parallel
        disable_input_reduce: If true, disable the reduce call in the backward
                              of the input tensor
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        backend: str = 'sequential',
        input_is_parallel: bool = False,
        disable_input_reduce: bool = False,
        gather_output: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(GroupColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.backend = backend
        self.input_is_parallel = input_is_parallel
        self.disable_input_reduce = disable_input_reduce
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.empty(self.num_groups * self.in_features, self.output_size_per_partition))

        # Initialize weight.
        self.master_weight = _initialize_group_weight(
            rearrange(self.weight, '(n d) v -> n d v', n=self.num_groups),
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            2,
            self.num_groups,
            True,
            init_method,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data, dim=1)

    def forward(self, input_: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            if self.disable_input_reduce:
                input_parallel = gather_from_model_parallel_region(input_)
            else:
                input_parallel = gather_copy_model_parallel_region(input_)
        else:
            if self.disable_input_reduce:
                input_parallel = input_
            else:
                input_parallel = copy_to_model_parallel_region(input_)

        # mgmm
        w = rearrange(self.weight, '(n d) v -> n d v', n=self.num_groups)
        output_parallel = mgmm(input_parallel, w, group_sizes, False, self.backend)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        return ('in_features={}, out_features={} ({}), groups={}, backend={}, '
                'parallel_input={}, reduce_input={}, gather_output={}').format(
            self.in_features, self.out_features, self.output_size_per_partition, self.num_groups,
            self.backend, self.input_is_parallel, not self.disable_input_reduce, self.gather_output
        )

class GroupRowParallelLinear(torch.nn.Module):
    """Grouped Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        backend: str = 'sequential',
        input_is_parallel: bool = False,
        parallel_output: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
    ):
        super(GroupRowParallelLinear, self).__init__()
        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.backend = backend
        self.input_is_parallel = input_is_parallel
        self.parallel_output = parallel_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.empty(self.num_groups * self.out_features, self.input_size_per_partition))

        # Initialize weight.
        self.master_weight = _initialize_group_weight(
            rearrange(self.weight, '(n v) d -> n v d', n=self.num_groups),
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            2,
            self.num_groups,
            False,
            init_method,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data, dim=1)

    def forward(self, input_: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)

        # Matrix multiply.
        w = rearrange(self.weight, '(n v) d -> n v d', n=self.num_groups)
        output_parallel = mgmm(input_parallel, w, group_sizes, True, self.backend)
        if self.parallel_output:
            # Reduce-scatter across all the partitions.
            output = reduce_scatter_model_parallel_region(output_parallel)
        else:
            # All-reduce across all the partitions.
            output = reduce_from_model_parallel_region(output_parallel)

        return output

    def extra_repr(self) -> str:
        return ('in_features={} ({}), out_features={} ({}), groups={}, backend={}, '
                'parallel_input={}, parallel_output={}').format(
            self.in_features, self.input_size_per_partition,
            self.out_features, self.output_size_per_partition, self.num_groups,
            self.backend, self.input_is_parallel, self.parallel_output
        )
