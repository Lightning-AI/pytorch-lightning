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


class Collective(ABC):
    """Base class for collective functions for training type plugins."""

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Forces all possibly joined processes to wait for each other."""

    @abstractmethod
    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes."""

    @abstractmethod
    def all_gather(
        self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """Perform a all_gather on all processes."""

    @abstractmethod
    def reduce(
        self,
        tensor: Union[torch.Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[torch.Tensor, Any]:
        """Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            *args: plugin-specific positional arguments
            **kwargs: plugin-specific keyword arguments
        """

    @abstractmethod
    def reduce_boolean_decision(self, decision: bool) -> bool:
        """Reduce the early stopping decision across all processes."""
