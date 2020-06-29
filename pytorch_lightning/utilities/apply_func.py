from abc import ABC
from collections import Mapping, Sequence
from copy import copy
from typing import Any, Callable, Union

import torch

import importlib
TORCHTEXT_AVAILABLE = importlib.util.find_spec("torchtext") is not None
if TORCHTEXT_AVAILABLE:
    from torchtext.data import Batch
else:
    Batch = type(None)


def apply_to_collection(data: Any, dtype: Union[type, tuple], function: Callable, *args, **kwargs) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        the resulting collection

    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    elif isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs)
                          for k, v in data.items()})
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data


class TransferableDataType(ABC):
    """
    A custom type for data that can be moved to a torch device via `.to(...)`.

    Example:

        >>> isinstance(dict, TransferableDataType)
        False
        >>> isinstance(torch.rand(2, 3), TransferableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), TransferableDataType)
        True
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented


def move_data_to_device(batch: Any, device: torch.device):
    """
    Transfers a collection of data to the given device. Any object that defines a method
    ``to(device)`` will be moved and all other objects in the collection will be left untouched.

    Args:
        batch: A tensor or collection of tensors or anything that has a method `.to(...)`.
            See :func:`apply_to_collection` for a list of supported collection types.
        device: The device to which the data should be moved

    Return:
        the same collection but with all contained tensors residing on the new device.

    See Also:
        - :meth:`torch.Tensor.to`
        - :class:`torch.device`
    """
    def batch_to(data):
        # try to move torchtext data first
        if TORCHTEXT_AVAILABLE and isinstance(data, Batch):

            # Shallow copy because each Batch has a reference to Dataset which contains all examples
            device_data = copy(data)
            for field in data.fields:
                # Batch contains output of Field.process(...) which is tensor hence .to(...) exists
                device_field = getattr(data, field).to(device, non_blocking=True)
                setattr(device_data, field, device_field)
            return device_data
        else:
            return data.to(device, non_blocking=True)

    return apply_to_collection(batch, dtype=(TransferableDataType, Batch), function=batch_to)
