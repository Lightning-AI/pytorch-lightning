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
import operator
from abc import ABC
from collections.abc import Mapping, Sequence
from copy import copy
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _compare_version, _TORCHTEXT_AVAILABLE

if _TORCHTEXT_AVAILABLE:
    if _compare_version("torchtext", operator.ge, "0.9.0"):
        from torchtext.legacy.data import Batch
    else:
        from torchtext.data import Batch
else:
    Batch = type(None)


def to_dtype_tensor(value, dtype: torch.dtype = None, device: torch.device = None):
    if device is None:
        raise MisconfigurationException("device (torch.device) should be provided.")
    return torch.tensor(value, dtype=dtype, device=device)


def from_numpy(value, device: torch.device = None):
    if device is None:
        raise MisconfigurationException("device (torch.device) should be provided.")
    return torch.from_numpy(value).to(device)


CONVERSION_DTYPES = [
    # bool -> uint8 as bool -> torch.bool triggers RuntimeError: Unsupported data type for NCCL process group
    (bool, partial(to_dtype_tensor, dtype=torch.uint8)),
    (int, partial(to_dtype_tensor, dtype=torch.int)),
    (float, partial(to_dtype_tensor, dtype=torch.float)),
    (np.ndarray, from_numpy),
]


def _is_namedtuple(obj: object) -> bool:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def apply_to_collection(
    data: Any,
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    include_none: bool = True,
    **kwargs
) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection
    """
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []  # can't use dict, need to preserve order if `OrderedDict`
        for k, v in data.items():
            v = apply_to_collection(v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            if include_none or v is not None:
                out.append((k, v))
        return elem_type(out)

    is_namedtuple = _is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(d, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple else elem_type(out)

    # data is neither of dtype, nor a collection
    return data


def apply_to_collections(
    data1: Optional[Any],
    data2: Optional[Any],
    dtype: Union[type, tuple],
    function: Callable,
    *args,
    wrong_dtype: Optional[Union[type, tuple]] = None,
    **kwargs
) -> Any:
    """
    Zips two collections and applies a function to their items of a certain dtype.

    Args:
        data1: The first collection
        data2: The second collection
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection
    """
    if data1 is None and data2 is not None:
        # in case they were passed reversed
        data1, data2 = data2, None

    elem_type = type(data1)

    if isinstance(data1, dtype) and data2 is not None and (wrong_dtype is None or not isinstance(data1, wrong_dtype)):
        return function(data1, data2, *args, **kwargs)

    if isinstance(data1, Mapping) and data2 is not None:
        # use union because we want to fail if a key does not exist in both
        zipped = {k: (data1[k], data2[k]) for k in data1.keys() | data2.keys()}
        return elem_type({
            k: apply_to_collections(*v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            for k, v in zipped.items()
        })

    is_namedtuple = _is_namedtuple(data1)
    is_sequence = isinstance(data1, Sequence) and not isinstance(data1, str)
    if (is_namedtuple or is_sequence) and data2 is not None:
        assert len(data1) == len(data2), 'Sequence collections have different sizes'
        out = [
            apply_to_collections(v1, v2, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            for v1, v2 in zip(data1, data2)
        ]
        return elem_type(*out) if is_namedtuple else elem_type(out)

    return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)


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
        if _TORCHTEXT_AVAILABLE and isinstance(data, Batch):

            # Shallow copy because each Batch has a reference to Dataset which contains all examples
            device_data = copy(data)
            for field, field_value in data.dataset.fields.items():
                if field_value is None:
                    continue
                device_field = move_data_to_device(getattr(data, field), device)
                setattr(device_data, field, device_field)
            return device_data

        kwargs = dict(non_blocking=True) if isinstance(data, torch.Tensor) else {}
        return data.to(device, **kwargs)

    dtype = (TransferableDataType, Batch) if _TORCHTEXT_AVAILABLE else TransferableDataType
    return apply_to_collection(batch, dtype=dtype, function=batch_to)


def convert_to_tensors(data, device: torch.device = None):
    if device is None:
        raise MisconfigurationException("device (torch.device) should be provided.")

    for src_dtype, conversion_func in CONVERSION_DTYPES:
        data = apply_to_collection(data, src_dtype, partial(conversion_func, device=device))

    def _move_to_device_and_make_contiguous(t: torch.Tensor, device: torch.device):
        return t.to(device).contiguous()

    data = apply_to_collection(data, torch.Tensor, partial(_move_to_device_and_make_contiguous, device=device))
    return data
