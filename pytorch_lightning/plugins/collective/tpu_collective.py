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
import io
import os
from typing import Any, Optional, Union

import torch

from pytorch_lightning.plugins.collective import Collective
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TPU_AVAILABLE:
    import torch_xla.core.xla_env_vars as xenv
    import torch_xla.core.xla_model as xm
    from torch_xla.core.xla_model import rendezvous


class TPUCollective(Collective):
    """Collective interface for TPUSpawning training type plugins."""

    def __init__(
        self,
        device: Union[str, torch.device] = torch.device("xla"),
        root_device: torch.device = torch.device("xla"),
        world_size: int = 1,
    ):
        self.device = device
        self.root_device = root_device
        self.world_size = world_size

    @property
    def is_distributed(self) -> bool:
        # HOST_WORLD_SIZE is None outside the xmp.spawn process
        return os.getenv(xenv.HOST_WORLD_SIZE, None) is not None and self.world_size != 1

    def barrier(self, name: Optional[str] = None) -> None:
        if self.is_distributed:
            rendezvous(name)

    def broadcast(self, obj: object, src: int = 0) -> object:
        if not self.is_distributed:
            return obj
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        data_tensor = torch.tensor(data, device=self.root_device, dtype=torch.float)
        data = xm.all_gather(data_tensor)
        buffer = io.BytesIO(data.cpu().byte().numpy())
        obj = torch.load(buffer)
        return obj

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """
        Function to gather a tensor from several distributed processes
        Args:
            tensor: tensor of shape (batch, ...)
            group: not available with TPUs
            sync_grads: not available with TPUs
        Return:
            A tensor of shape (world_size, batch, ...)
        """
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        return xm.all_gather(tensor)

    def reduce(self, output: Any, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None) -> Any:
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, device=self.device)

        _invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        _invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if _invalid_reduce_op or _invalid_reduce_op_str:
            raise MisconfigurationException(
                "Currently, TPUSpawn TrainingTypePlugin only support `sum`, `mean`, `avg` reduce operation."
            )

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    def reduce_boolean_decision(self, decision: bool) -> bool:
        """Reduce the early stopping decision across all processes."""
        return decision
