#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities that can be used with distributed training."""

from typing import Any, Callable, Dict, Optional

import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning_fabric.utilities.distributed import _distributed_available as new_distributed_available
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info


def register_ddp_comm_hook(
    model: DistributedDataParallel,
    ddp_comm_state: Optional[object] = None,
    ddp_comm_hook: Optional[Callable] = None,
    ddp_comm_wrapper: Optional[Callable] = None,
) -> None:
    """Function to register communication hook for DDP model https://pytorch.org/docs/master/ddp_comm_hooks.html.

    Args:
        model:
            DDP model
        ddp_comm_state:
            state is passed to the hook and can be used to maintain
            and update any state information that users would like to
            maintain as part of the training process. Examples: error
            feedback in gradient compression, peers to communicate with
            next in GossipGrad etc.
        ddp_comm_hook:
            hook(state: object, bucket: dist._GradBucket) -> torch.futures.Future

            This callable function is called once the bucket is ready. The
            hook can perform whatever processing is needed and return
            a Future indicating completion of any async work (ex: allreduce).
            If the hook doesn't perform any communication, it can also
            just return a completed Future. The Future should hold the
            new value of grad bucket's tensors. Once a bucket is ready,
            c10d reducer would call this hook and use the tensors returned
            by the Future and copy grads to individual parameters.
        ddp_comm_wrapper:
            communication hook wrapper to support a communication hook such
            as FP16 compression as wrapper, which could be combined with
            ddp_comm_hook

    Examples:

        >>> from torch.distributed.algorithms.ddp_comm_hooks import ( # doctest: +SKIP
        ...     default_hooks as default,
        ...     powerSGD_hook as powerSGD,
        ...     post_localSGD_hook as post_localSGD,
        ... )
        >>>
        >>> # fp16_compress_hook for compress gradients
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     ddp_comm_hook=default.fp16_compress_hook,
        ... )
        >>>
        >>> # powerSGD_hook
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     ddp_comm_state=powerSGD.PowerSGDState(
        ...         process_group=None,
        ...         matrix_approximation_rank=1,
        ...         start_powerSGD_iter=5000,
        ...     ),
        ...     ddp_comm_hook=powerSGD.powerSGD_hook,
        ... )
        >>>
        >>> # post_localSGD_hook
        >>> subgroup, _ = torch.distributed.new_subgroups() # doctest: +SKIP
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     state=post_localSGD.PostLocalSGDState(
        ...         process_group=None,
        ...         subgroup=subgroup,
        ...         start_localSGD_iter=1_000,
        ...     ),
        ...     ddp_comm_hook=post_localSGD.post_localSGD_hook,
        ... )
        >>>
        >>> # fp16_compress_wrapper combined with other communication hook
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     ddp_comm_state=powerSGD.PowerSGDState(
        ...         process_group=None,
        ...         matrix_approximation_rank=1,
        ...         start_powerSGD_iter=5000,
        ...     ),
        ...     ddp_comm_hook=powerSGD.powerSGD_hook,
        ...     ddp_comm_wrapper=default.fp16_compress_wrapper,
        ... )
    """
    if ddp_comm_hook is None:
        return
    # inform mypy that ddp_comm_hook is callable
    ddp_comm_hook: Callable = ddp_comm_hook

    if ddp_comm_wrapper is not None:
        rank_zero_info(
            f"DDP comm wrapper is provided, apply {ddp_comm_wrapper.__qualname__}({ddp_comm_hook.__qualname__})."
        )
        ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)

    rank_zero_debug(f"Registering DDP comm hook: {ddp_comm_hook.__qualname__}.")
    model.register_comm_hook(state=ddp_comm_state, hook=ddp_comm_hook)  # type: ignore[operator]


def _broadcast_object_list(obj: Any, rank: int) -> Any:
    objects = [obj if torch.distributed.get_rank() == rank else None]
    torch.distributed.broadcast_object_list(objects, src=rank)
    return objects[0]


# TODO: Refactor with the Strategy Collectives once finalized.
def _collect_states_on_rank_zero(state: Dict[str, Any]) -> Dict[int, Any]:
    """This distributed utility collects dictionary state across all processes.

    Args:
        state: Dictionary containing the state of the current process

    Returns:
        states: On global rank 0, a dictionary where the primary keys are
            the process rank and the values their associated states. Otherwise, returns None.
    """
    if not new_distributed_available():
        return {0: state}
    return {rank: _broadcast_object_list(state, rank) for rank in range(torch.distributed.get_world_size())}
