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
from typing import Any

import torch

from pytorch_lightning.overrides.torch_distributed import (
    _object_to_tensor,
    _rank_not_in_group,
    _tensor_to_object,
    broadcast_object_list,
)
from pytorch_lightning.utilities import _SMDIST_AVAILABLE
from pytorch_lightning.utilities.distributed import group as _group

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.distributed as sm_dist


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any, group=_group.WORLD):
        # always wrap into a list so list can be brodcasted.
        obj = [obj]

        if self.rank != 0:
            obj = [None] * len(obj)

        broadcast_object_list(obj, 0, group=group or _group.WORLD)

        return obj[0]


class SMLightningDistributed(LightningDistributed):

    def broadcast(self, obj: Any, group=sm_dist.group.WORLD):
        # always wrap into a list so list can be brodcasted.
        obj = [obj]

        obj = [obj]

        if self.rank != 0:
            obj = [None] * len(obj)

        _broadcast_object_list(obj, self.rank, 0, group=group)

        return obj[0]

    # def _broadcast(self, tensor: torch.Tensor, src: int, group: Optional[Any] = None):
    #     if group is None:
    #         return sm_dist.broadcast(tensor, src=src)
    #     return sm_dist.broadcast(tensor, src=0, group=group)

    # def _emit(self, obj: Any, group=_group.WORLD):
    #     buffer = io.BytesIO()
    #     torch.save(obj, buffer)
    #     data = bytearray(buffer.getbuffer())
    #     length_tensor = torch.tensor([len(data)]).long().to(self.device)
    #     self._broadcast(length_tensor, src=0, group=group)
    #     data_tensor = torch.ByteTensor(data).to(self.device)
    #     self._broadcast(data_tensor, src=0, group=group)

    # def _receive(self, group=_group.WORLD):
    #     length_tensor = torch.tensor([0]).long().to(self.device)
    #     self._broadcast(length_tensor, src=0, group=group)
    #     data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
    #     self._broadcast(data_tensor, src=0, group=group)
    #     buffer = io.BytesIO(data_tensor.cpu().numpy())
    #     obj = torch.load(buffer)
    #     return obj


# Taken from https://github.com/pytorch/pytorch/blob/1.7/torch/distributed/distributed_c10d.py#L1327
def _broadcast_object_list(object_list, rank, src=0, group=None):
    if _rank_not_in_group(group):
        return

    my_rank = rank
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(*[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.LongTensor(len(object_list))

    # group_backend = get_backend(group)
    # is_nccl_backend = group_backend == Backend.NCCL
    # current_device = torch.device("cpu")
    # if is_nccl_backend:
    #     # See note about using torch.cuda.current_device() here in docstring.
    #     # We cannot simply use my_rank since rank == device is not necessarily
    #     # true.
    #     current_device = torch.device('cuda', torch.cuda.current_device())
    #     object_sizes_tensor = object_sizes_tensor.to(current_device)
    #     object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    sm_dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.ByteTensor(torch.sum(object_sizes_tensor).item())

    # if is_nccl_backend:
    #     object_tensor = object_tensor.to(current_device)

    sm_dist.broadcast(object_tensor, src=src, group=group)

    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.ByteTensor)  # type: ignore[call-overload]
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)
