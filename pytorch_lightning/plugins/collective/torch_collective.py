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
from typing import Any, Optional, Union

import torch
import torch.distributed

from pytorch_lightning.overrides.torch_distributed import broadcast_object_list
from pytorch_lightning.plugins.collective import Collective
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_8
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available, distributed_available
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.distributed import ReduceOp, sync_ddp_if_available
from pytorch_lightning.utilities.types import _METRIC_COLLECTION


class TorchCollective(Collective):
    """Collective interface for DDP, DDPSpawn, DP and DDP2."""

    def __init__(
        self,
        local_reduce: bool = False,
        rank: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
        device_id: Optional[int] = None,
        world_size: int = 1,
    ) -> None:
        """
        Note:
            DDP and DDPSpawn sync accross multiple nodes/devices, local_reduce = False
            DP run reduce in on node, local_reduce = True
            DDP2 behaves like DP in one node, local_reduce = True

        local_reduce set in Plugins.setup() functions
        """
        self.local_reduce = local_reduce
        self.rank = rank
        self.device = device
        self.device_id = device_id
        self.world_size = world_size

    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if not distributed_available():
            return
        if _TORCH_GREATER_EQUAL_1_8 and torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.device_id)
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        if not distributed_available():
            return obj
        else:
            obj = [obj]
            if self.rank != 0:
                obj = [None] * len(obj)
            broadcast_object_list(obj, 0, group=_group.WORLD)
            return obj[0]

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes."""
        return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def reduce(
        self,
        tensor: Union[torch.Tensor, _METRIC_COLLECTION],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[torch.Tensor, _METRIC_COLLECTION]:
        """Reduces the given tensor (e.g. across GPUs/processes)

        If local_reduce = True (dp and ddp2), reduces tensor from all local processes.

        If local_reduce = False (ddp, ddpspawning and extentions), reduces a tensor from several distributed processes

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if self.local_reduce:

            def mean(t: torch.Tensor) -> torch.Tensor:
                original_dtype = t.dtype
                return t.float().mean().to(original_dtype)

            return apply_to_collection(tensor, torch.Tensor, mean)

        if isinstance(tensor, torch.Tensor):
            tensor = sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def reduce_boolean_decision(self, decision: bool) -> bool:
        if self.local_reduce:
            return decision
        else:
            decision1 = torch.tensor(int(decision), device=self.device)
            decision2 = self.reduce(decision1, reduce_op=ReduceOp.SUM)
            decision = bool(decision2 == self.world_size)
            return decision
