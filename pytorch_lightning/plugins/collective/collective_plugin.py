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
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch

from pytorch_lightning.utilities.distributed import ReduceOp


class CollectivePlugin(ABC):
    """Interface for collective functions.

    Lightning collective supports communications between multiple processes and multiple nodes, provides routines such
    as barrier, broadcast, all_gather, and reduce

    .. note::
        This API is experimental/in-beta and subject to change
    """

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this function.

        Args:
            name: a str pass into barrier. Only torch xla respect this param
        """

    @abstractmethod
    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes.

        Args:
            obj: the object to broadcast
            src: source rank.
        """

    @abstractmethod
    def all_gather(
        self, tensor: torch.Tensor, process_group: Optional[Any] = None, sync_grads: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """Perform a all_gather on all processes.

        Args:
            tensor: the tensor to all_gather
            process_group: the process group to gather results from
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Returns: a tensor (torch distributed) or a list of tensor (horovod)
        """

    @abstractmethod
    def reduce(
        self,
        tensor: Union[torch.Tensor, Any],
        process_group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[torch.Tensor, Any]:
        """Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            process_group: the process group to reduce
            reduce_op: the reduction operation. Defaults to 'mean'.
                Can also be a string 'sum' or ReduceOp.
            *args: plugin-specific positional arguments
            **kwargs: plugin-specific keyword arguments
        """
