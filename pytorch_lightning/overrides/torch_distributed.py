import logging
import pickle

import torch

from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8

log = logging.getLogger(__name__)

if torch.distributed.is_available():
    from torch.distributed import Backend, broadcast, get_backend, get_rank, GroupMember

# The code underneath is taken from PyTorch ``torch/distributed/distributed_c10d.py``
# and enable broadcasting for PyTorch 1.6 and lower.


# https://github.com/pytorch/pytorch/blob/1.7/torch/distributed/distributed_c10d.py#L160
def _rank_not_in_group(group):
    """
    Helper that checks if the current process's rank is not in a given group.
    """
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


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


if _TORCH_GREATER_EQUAL_1_8 and torch.distributed.is_available():
    from torch.distributed.distributed_c10d import broadcast_object_list
else:
    broadcast_object_list = _broadcast_object_list
