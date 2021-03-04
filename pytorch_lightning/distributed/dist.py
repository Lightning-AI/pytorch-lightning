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
import pickle
from typing import Any

import torch
from torch.distributed.distributed_c10d import (
    _rank_not_in_group,
    Backend,
    broadcast,
    get_backend,
    get_rank,
    GroupMember,
)


# Taken from https://github.com/pytorch/pytorch/blob/1.7/torch/distributed/distributed_c10d.py#L1164
def _object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)  # type: ignore[attr-defined]
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


# Taken from https://github.com/pytorch/pytorch/blob/1.7/torch/distributed/distributed_c10d.py
def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    out = pickle.loads(buf)
    return out


# Taken from https://github.com/pytorch/pytorch/blob/1.7/torch/distributed/distributed_c10d.py#L1327
def _broadcast_object_list(object_list, src=0, group=None):
    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(*[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.LongTensor(len(object_list))

    group_backend = get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    current_device = torch.device("cpu")
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.ByteTensor(torch.sum(object_sizes_tensor).item())

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)
    broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.ByteTensor)  # type: ignore[call-overload]
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any, group=None):
        is_list = isinstance(obj, list)

        if not is_list:
            obj = [obj]

        if self.rank != 0:
            obj = [None for _ in range(len(obj))]

        _broadcast_object_list(obj, 0, group=group or GroupMember.WORLD)

        if not is_list:
            obj = obj[0]

        return obj
