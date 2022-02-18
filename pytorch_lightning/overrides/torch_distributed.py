import io
import logging
import os
import pickle

import torch
from torch._C._distributed_c10d import ProcessGroup

from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


logger = logging.getLogger(__name__)

if torch.distributed.is_available():
    from torch.distributed import Backend, broadcast, get_backend, get_rank, GroupMember

# The code underneath is taken from PyTorch `torch/distributed/distributed_c10d.py`
# the distributed backend and tensor type updates for habana backend is done here before broadcast

# https://github.com/pytorch/pytorch/blob/1.10.1/torch/distributed/distributed_c10d.py#L256
def _rank_not_in_group(group: ProcessGroup):
    """Helper that checks if the current process's rank is not in a given group."""
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


# Taken from https://github.com/pytorch/pytorch/blob/1.10.1/torch/distributed/distributed_c10d.py#L1518
def _object_to_tensor(obj):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


# Taken from https://github.com/pytorch/pytorch/blob/1.10.1/torch/distributed/distributed_c10d.py#L1530
def _tensor_to_object(tensor, tensor_size):
    buf = tensor.numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


# Taken from https://github.com/pytorch/pytorch/blob/1.10.1/torch/distributed/distributed_c10d.py#L1729
def _broadcast_object_list(object_list, src=0, group=None, device=None):
    """Broadcasts picklable objects in ``object_list`` to the whole group. Similar to :func:`broadcast`, but Python
    objects can be passed in. Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> broadcast_objects
        ['foo', 12, {1: 2}]
    """
    if _rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(*[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == Backend.NCCL
    dist_backend = os.environ.get("PL_TORCH_DISTRIBUTED_BACKEND")
    is_hpu_backend = group_backend == torch.distributed.Backend(str(dist_backend))
    current_device = None
    if device is not None:
        if is_nccl_backend and device.type != "cuda":
            raise ValueError("device type must be cuda for nccl backend")
        current_device = device
    else:
        current_device = torch.device("cpu")
        if is_nccl_backend:
            # See note about using torch.cuda.current_device() here in
            # docstring. We cannot simply use my_rank since rank == device is
            # not necessarily true.
            current_device = torch.device("cuda", torch.cuda.current_device())
    if is_nccl_backend:
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    elif is_hpu_backend:
        current_device = torch.device("hpu")
        # Workaround: HPU doesn't not support long tensors for collectives
        if (object_sizes_tensor.type() == "torch.LongTensor") or \
           (object_sizes_tensor.type() == "torch.hpu.LongTensor"):
            object_sizes_tensor = object_sizes_tensor.int()
        else:
            print("unhandled hpu object_sizes_tensor type :: ", object_sizes_tensor.type())
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
        )

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)
    elif is_hpu_backend:
        object_tensor = object_tensor.to(current_device)

    broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device("cpu"):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


if not torch.distributed.is_available():
    # avoid failures on early PyTorch versions for Windows where
    # not all functions used in `broadcast_object_list` are available.
    def _broadcast_noop(obj, *_, **__):
        return obj

    broadcast_object_list = _broadcast_noop
elif _TORCH_GREATER_EQUAL_1_8 and not _HPU_AVAILABLE:
    from torch.distributed.distributed_c10d import broadcast_object_list
else:
    broadcast_object_list = _broadcast_object_list
