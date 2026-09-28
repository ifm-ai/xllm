from typing import Iterable, Optional, Union, List
import math
import functools
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._init_utils import ProcessGroupType
from torch.distributed.fsdp.fully_sharded_data_parallel import TrainingState, _get_grad_norm
from torch.distributed.fsdp._runtime_utils import _lazy_init

from .initialize import (
    get_context_parallel_world_size,
    get_data_parallel_world_size,
    get_model_parallel_world_size
)


class FullyShardedDataParallel(FSDP):
    """
    A wrapper of torch.FullyShardedDataParallel
    Args:
        module (nn.Module):
            This is the module to be wrapped with FSDP.
        process_group (Optional[Union[ProcessGroup, Tuple[ProcessGroup, ProcessGroup]]]):
            This is the process group over which the model is sharded and thus
            the one used for FSDP's all-gather and reduce-scatter collective
            communications. If ``None``, then FSDP uses the default process
            group. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a tuple of
            process groups, representing the groups over which to shard and
            replicate, respectively. If ``None``, then FSDP constructs process
            groups for the user to shard intra-node and replicate inter-node.
            (Default: ``None``)
        sharding_strategy (Optional[ShardingStrategy]):
            This configures the sharding strategy, which may trade off memory
            saving and communication overhead. See :class:`ShardingStrategy`
            for details. (Default: ``FULL_SHARD``)
        cpu_offload (Optional[CPUOffload]):
            This configures CPU offloading. If this is set to ``None``, then
            no CPU offloading happens. See :class:`CPUOffload` for details.
            (Default: ``None``)
        auto_wrap_policy (Optional[Union[Callable[[nn.Module, bool, int], bool], ModuleWrapPolicy, CustomPolicy]]):
            This specifies a policy to apply FSDP to submodules of ``module``,
            which is needed for communication and computation overlap and thus
            affects performance. If ``None``, then FSDP only applies to
            ``module``, and users should manually apply FSDP to parent modules
            themselves (proceeding bottom-up). For convenience, this accepts
            ``ModuleWrapPolicy`` directly, which allows users to specify the
            module classes to wrap (e.g. the transformer block). Otherwise,
            this should be a callable that takes in three arguments
            ``module: nn.Module``, ``recurse: bool``, and
            ``nonwrapped_numel: int`` and should return a ``bool`` specifying
            whether the passed-in ``module`` should have FSDP applied if
            ``recurse=False`` or if the traversal should continue into the
            module's subtree if ``recurse=True``. Users may add additional
            arguments to the callable. The ``size_based_auto_wrap_policy`` in
            ``torch.distributed.fsdp.wrap.py`` gives an example callable that
            applies FSDP to a module if the parameters in its subtree exceed
            100M numel. We recommend printing the model after applying FSDP
            and adjusting as needed.
        backward_prefetch (Optional[BackwardPrefetch]):
            This configures explicit backward prefetching of all-gathers. If
            ``None``, then FSDP does not backward prefetch, and there is no
            communication and computation overlap in the backward pass. See
            :class:`BackwardPrefetch` for details. (Default: ``BACKWARD_PRE``)
        mixed_precision (Optional[MixedPrecision]):
            This configures native mixed precision for FSDP. If this is set to
            ``None``, then no mixed precision is used. Otherwise, parameter,
            buffer, and gradient reduction dtypes can be set. See
            :class:`MixedPrecision` for details. (Default: ``None``)
        ignored_modules (Optional[Iterable[torch.nn.Module]]): Modules whose
            own parameters and child modules' parameters and buffers are
            ignored by this instance. None of the modules directly in
            ``ignored_modules`` should be :class:`FullyShardedDataParallel`
            instances, and any child modules that are already-constructed
            :class:`FullyShardedDataParallel` instances will not be ignored if
            they are nested under this instance. This argument may be used to
            avoid sharding specific parameters at module granularity when using an
            ``auto_wrap_policy`` or if parameters' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should
            be initialized onto an actual device. As of v1.12, FSDP detects
            modules with parameters or buffers on meta device via ``is_meta``
            and either applies ``param_init_fn`` if specified or calls
            ``nn.Module.reset_parameters()`` otherwise. For both cases, the
            implementation should *only* initialize the parameters/buffers of
            the module, not those of its submodules. This is to avoid
            re-initialization. In addition, FSDP also supports deferred
            initialization via torchdistX's (https://github.com/pytorch/torchdistX)
            ``deferred_init()`` API, where the deferred modules are initialized
            by calling ``param_init_fn`` if specified or torchdistX's default
            ``materialize_module()`` otherwise. If ``param_init_fn`` is
            specified, then it is applied to all meta-device modules, meaning
            that it should probably case on the module type. FSDP calls the
            initialization function before parameter flattening and sharding.
        device_id (Optional[Union[int, torch.device]]): An ``int`` or
            ``torch.device`` giving the CUDA device on which FSDP
            initialization takes place, including the module initialization
            if needed and the parameter sharding. This should be specified to
            improve initialization speed if ``module`` is on CPU. If the
            default CUDA device was set (e.g. via ``torch.cuda.set_device``),
            then the user may pass ``torch.cuda.current_device`` to this.
            (Default: ``None``)
        sync_module_states (bool): If ``True``, then each FSDP module will
            broadcast module parameters and buffers from rank 0 to ensure that
            they are replicated across ranks (adding communication overhead to
            this constructor). This can help load ``state_dict`` checkpoints
            via ``load_state_dict`` in a memory efficient way. See
            :class:`FullStateDictConfig` for an example of this. (Default:
            ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next forward-pass all-gather before the current forward
            computation. This is only useful for CPU-bound workloads, in which
            case issuing the next all-gather earlier may improve overlap. This
            should only be used for static-graph models since the prefetching
            follows the first iteration's execution order. (Default: ``False``)
        limit_all_gathers (bool): If ``True``, then FSDP explicitly
            synchronizes the CPU thread to ensure GPU memory usage from only
            *two* consecutive FSDP instances (the current instance running
            computation and the next instance whose all-gather is prefetched).
            If ``False``, then FSDP allows the CPU thread to issue all-gathers
            without any extra synchronization. (Default: ``True``) We often
            refer to this feature as the "rate limiter". This flag should only
            be set to ``False`` for specific CPU-bound workloads with low
            memory pressure in which case the CPU thread can aggressively issue
            all kernels without concern for the GPU memory usage.
        use_orig_params (bool): Setting this to ``True`` has FSDP use
            ``module`` 's original parameters. FSDP exposes those original
            parameters to the user via :meth:`nn.Module.named_parameters`
            instead of FSDP's internal :class:`FlatParameter` s. This means
            that the optimizer step runs on the original parameters, enabling
            per-original-parameter hyperparameters. FSDP preserves the original
            parameter variables and manipulates their data between unsharded
            and sharded forms, where they are always views into the underlying
            unsharded or sharded :class:`FlatParameter`, respectively. With the
            current algorithm, the sharded form is always 1D, losing the
            original tensor structure. An original parameter may have all,
            some, or none of its data present for a given rank. In the none
            case, its data will be like a size-0 empty tensor. Users should not
            author programs relying on what data is present for a given
            original parameter in its sharded form. ``True`` is required to
            use ``torch.compile()``. Setting this to ``False`` exposes FSDP's
            internal :class:`FlatParameter` s to the user via
            :meth:`nn.Module.named_parameters`. (Default: ``False``)
        ignored_states (Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]):
            Ignored parameters or modules that will not be managed by this FSDP
            instance, meaning that the parameters are not sharded and their
            gradients are not reduced across ranks. This argument unifies with
            the existing ``ignored_modules`` argument, and we may deprecate
            ``ignored_modules`` soon. For backward compatibility, we keep both
            ``ignored_states`` and `ignored_modules``, but FSDP only allows one
            of them to be specified as not ``None``.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        model_parallel_process_group: Optional[ProcessGroup] = None,
        **kwargs
    ):
        super().__init__(module=module, process_group=process_group, **kwargs)
        self.data_parallel_world_size = get_data_parallel_world_size()
        self.context_parallel_world_size = get_context_parallel_world_size()
        self.model_parallel_process_group = model_parallel_process_group
        self.model_parallel_world_size = get_model_parallel_world_size()

        assert self.data_parallel_world_size % self.world_size == 0

    def extra_repr(self) -> str:
        replicas = self.data_parallel_world_size // self.world_size
        repr = (
            f"world_size=({self.world_size}x{replicas}, {self.context_parallel_world_size}, {self.model_parallel_world_size}), "
            f"sharding_strategy={self.sharding_strategy}, "
            f"mixed_precision=({self.mixed_precision.param_dtype}, {self.mixed_precision.reduce_dtype}, {self.mixed_precision.buffer_dtype})"
        )
        return repr

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
    ) -> float:
        """
        Clip all gradients at this point in time. The norm is computed over all
        gradients together, as if they were concatenated into a single vector.
        Gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note:: This is analogous to ``torch.nn.utils.clip_grad_norm_`` but
            handles the partitioning and multiple devices per rank under the
            hood. The default torch util is not applicable here, because each
            rank only has a partial view of all the grads in the model, so
            calling it for FSDP models would lead to different scaling being
            applied per subset of model parameters.

        .. warning:: This needs to be called on all ranks, since synchronization
            primitives will be used.
        """
        _lazy_init(self, self)
        if not self._is_root:
            raise RuntimeError(
                "`clip_grad_norm_()` should only be called on the root FSDP instance"
            )
        if self._zero_scalar is None:
            self._zero_scalar = torch.tensor(0.0, device=self.compute_device)
        self._assert_state(TrainingState.IDLE)

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        sharded_params_set = set()
        nonsharded_params_set = set()  # `NO_SHARD` or not FSDP-managed
        # Make sure to compute the local norm using lists for deterministic
        # iteration order and hence deterministic total norm computation
        sharded_params = []
        nonsharded_params = []
        grads: List[torch.Tensor] = []
        for handle in self._all_handles:
            if handle.uses_sharded_strategy:
                target_set = sharded_params_set
                target_list = sharded_params
            else:
                target_set = nonsharded_params_set
                target_list = nonsharded_params

            if handle._use_orig_params:
                for param in handle.flat_param._params:
                    if param not in target_set:
                        target_set.add(param)
                        target_list.append(param)
                        if param.grad is not None:
                            grads.append(param.grad)
            else:
                if handle.flat_param not in target_set:
                    target_set.add(handle.flat_param)
                    target_list.append(handle.flat_param)
                    if handle.flat_param.grad is not None:
                        grads.append(handle.flat_param.grad)

        for param in self.parameters():
            not_fsdp_managed = (
                    param not in sharded_params_set and param not in nonsharded_params_set
            )
            if not_fsdp_managed:
                nonsharded_params_set.add(param)
                nonsharded_params.append(param)
                if param.grad is not None:
                    grads.append(param.grad)

        # Compute local norms (forced to be in FP32)
        local_sharded_norm = _get_grad_norm(sharded_params, norm_type, self._zero_scalar, self.compute_device)
        local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type, self._zero_scalar, self.compute_device) if nonsharded_params else None

        # Reconstruct the total gradient norm depending on the norm type
        if norm_type == math.inf:
            total_norm = (
                torch.maximum(local_sharded_norm, local_nonsharded_norm)
                if local_nonsharded_norm is not None
                else local_sharded_norm
            )
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.process_group)
            if self.model_parallel_world_size > 1:
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.model_parallel_process_group)
        else:
            total_norm = local_sharded_norm ** norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            # All-reducing the local non-sharded norm would count it an extra
            # world-size-many times
            if local_nonsharded_norm is not None:
                total_norm += local_nonsharded_norm ** norm_type
            if self.model_parallel_world_size > 1:
                dist.all_reduce(total_norm, group=self.model_parallel_process_group)
            total_norm = total_norm ** (1.0 / norm_type)
        if self.cpu_offload.offload_params:
            total_norm = total_norm.cpu()

        total_norm_cpu = total_norm.item()
        if max_norm < total_norm_cpu:
            clip_coef = max_norm / (total_norm + 1e-6)
            for grad in grads:
                grad.mul_(clip_coef.to(grad.device, grad.dtype))

        return total_norm_cpu

    def num_parameters(self):
        return self.module.num_parameters()

    def tflops_per_token(self, seq_len: int):
        return self.module.tflops_per_token(seq_len)
