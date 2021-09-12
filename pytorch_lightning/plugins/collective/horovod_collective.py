# Copyright The PyTorch Lightning team.
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
from typing import Any, List, Optional, Union

import torch

from pytorch_lightning.plugins.collective import Collective
from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.distributed import group as dist_group
from pytorch_lightning.utilities.distributed import ReduceOp

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class HorovodCollective(Collective):
    """Collective interface for Horovod training type plugins."""

    def __init__(
        self,
        on_gpu: Optional[bool] = False,
        local_rank: int = 0,
    ) -> None:
        self.on_gpu = on_gpu
        self.local_rank = local_rank

    def join(self) -> None:
        """Horovod function that indicates that the rank finished processing data.

        All ranks that did not call join() continue to process allreduce operations. This function blocks the Python thread
        until all ranks join.
        """
        if self.on_gpu:
            hvd.join(self.local_rank)
        else:
            hvd.join()

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if distributed_available():
            self.join()

    def broadcast(self, obj: object, src: int = 0) -> object:
        obj = hvd.broadcast_object(obj, src)
        return obj

    def all_gather(
        self, result: Union[torch.Tensor], group: Optional[Any] = dist_group.WORLD, sync_grads: bool = False
    ) -> List[torch.Tensor]:
        if group is not None and group != dist_group.WORLD:
            raise ValueError("Horovod does not support allgather using a subcommunicator at this time. Unset `group`.")

        if len(result.shape) == 0:
            # Convert scalars to single dimension tensors
            result = result.reshape(1)

        # sync and gather all
        self.join()
        gathered = hvd.allgather(result)
        gathered_result = list(gathered.split(1, dim=0))
        return gathered_result

    def reduce(
        self,
        tensor: Union[torch.Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[torch.Tensor, Any]:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if group is not None:
            raise ValueError("Horovod does not support allreduce using a subcommunicator at this time. Unset `group`.")

        if reduce_op in (None, "avg", "mean"):
            reduce_op = hvd.Average
        elif reduce_op in ("sum", ReduceOp.SUM):
            reduce_op = hvd.Sum
        else:
            raise ValueError(f"unrecognized `reduce_op`: {reduce_op}")

        # sync all processes before reduction
        self.join()
        return hvd.allreduce(tensor, op=reduce_op)

    def reduce_boolean_decision(self, decision: bool) -> bool:
        """Reduce the early stopping decision across all processes."""
        return decision
