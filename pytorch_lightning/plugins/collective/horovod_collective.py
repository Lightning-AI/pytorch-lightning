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
from typing import Any, Optional, Union

import torch

from pytorch_lightning.plugins.collective import Collective
from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.types import _TPU_AVAILABLE

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm
    from torch_xla.core.xla_model import rendezvous
else:
    xm, rendezvous = [None] * 4

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class HorovodCollective(Collective):
    """Collective interface for Horovod training type plugins."""

    def __init__(
        self,
        on_gpu: Optional[bool] = False,
        local_rank: Optional[int] = 0,
    ):
        self._on_gpu = on_gpu
        self._local_rank = local_rank

    def join(self):
        """Horovod function that indicates that the rank finished processing data.

        All ranks that did not call join() continue to process allreduce operations. This function blocks Python thread
        until all ranks join.
        """
        if self.on_gpu:
            hvd.join(self.local_rank)
        else:
            hvd.join()

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
        return self._xm.all_gather(tensor)

    def reduce(self, tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"):
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
