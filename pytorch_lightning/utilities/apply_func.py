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
"""Utilities used for collections."""

import dataclasses
import operator
from abc import ABC
from collections import defaultdict, OrderedDict
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _compare_version, _TORCHTEXT_LEGACY
from pytorch_lightning.utilities.warnings import rank_zero_deprecation

if _TORCHTEXT_LEGACY:
    if _compare_version("torchtext", operator.ge, "0.9.0"):
        from torchtext.legacy.data import Batch
    else:
        from torchtext.data import Batch
else:
    Batch = type(None)


_CPU_DEVICES = ("cpu", torch.device("cpu"))


def to_dtype_tensor(
    value: Union[int, float, List[Union[int, float]]], dtype: torch.dtype, device: Union[str, torch.device]
) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


def from_numpy(value: np.ndarray, device: Union[str, torch.device]) -> torch.Tensor:
    return torch.from_numpy(value).to(device)


CONVERSION_DTYPES: List[Tuple[Any, Callable[[Any, Any], torch.Tensor]]] = [
    # bool -> uint8 as bool -> torch.bool triggers RuntimeError: Unsupported data type for NCCL process group
    (bool, partial(to_dtype_tensor, dtype=torch.uint8)),
    (int, partial(to_dtype_tensor, dtype=torch.int)),
    (float, partial(to_dtype_tensor, dtype=torch.float)),
    (np.ndarray, from_numpy),
]


def _is_namedtuple(obj: object) -> bool:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def _is_dataclass_instance(obj: object) -> bool:
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type]]] = None,
    include_none: bool = True,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

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
        out = []
        for k, v in data.items():
            v = apply_to_collection(
                v, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple = _is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple else elem_type(out)

    if _is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            v = None
            if field_init:
                v = apply_to_collection(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                raise MisconfigurationException(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
                    " HINT: is your batch a frozen dataclass?"
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data


def apply_to_collections(
    data1: Optional[Any],
    data2: Optional[Any],
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Any,
) -> Any:
    """Zips two collections and applies a function to their items of a certain dtype.

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

    Raises:
        AssertionError:
            If sequence collections have different data sizes.
    """
    if data1 is None:
        if data2 is None:
            return
        # in case they were passed reversed
        data1, data2 = data2, None

    elem_type = type(data1)

    if isinstance(data1, dtype) and data2 is not None and (wrong_dtype is None or not isinstance(data1, wrong_dtype)):
        return function(data1, data2, *args, **kwargs)

    if isinstance(data1, Mapping) and data2 is not None:
        # use union because we want to fail if a key does not exist in both
        zipped = {k: (data1[k], data2[k]) for k in data1.keys() | data2.keys()}
        return elem_type(
            {
                k: apply_to_collections(*v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
                for k, v in zipped.items()
            }
        )

    is_namedtuple = _is_namedtuple(data1)
    is_sequence = isinstance(data1, Sequence) and not isinstance(data1, str)
    if (is_namedtuple or is_sequence) and data2 is not None:
        assert len(data1) == len(data2), "Sequence collections have different sizes."
        out = [
            apply_to_collections(v1, v2, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            for v1, v2 in zip(data1, data2)
        ]
        return elem_type(*out) if is_namedtuple else elem_type(out)

    if _is_dataclass_instance(data1) and data2 is not None:
        if not _is_dataclass_instance(data2):
            raise TypeError(
                "Expected inputs to be dataclasses of the same type or to have identical fields"
                f" but got input 1 of type {type(data1)} and input 2 of type {type(data2)}."
            )
        if not (
            len(dataclasses.fields(data1)) == len(dataclasses.fields(data2))
            and all(map(lambda f1, f2: isinstance(f1, type(f2)), dataclasses.fields(data1), dataclasses.fields(data2)))
        ):
            raise TypeError("Dataclasses fields do not match.")
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        data = [data1, data2]
        fields: List[dict] = [{}, {}]
        memo: dict = {}
        for i in range(len(data)):
            for field in dataclasses.fields(data[i]):
                field_value = getattr(data[i], field.name)
                fields[i][field.name] = (field_value, field.init)
                if i == 0:
                    memo[id(field_value)] = field_value

        result = deepcopy(data1, memo=memo)

        # apply function to each field
        for ((field_name, (field_value1, field_init1)), (_, (field_value2, field_init2))) in zip(
            fields[0].items(), fields[1].items()
        ):
            v = None
            if field_init1 and field_init2:
                v = apply_to_collections(
                    field_value1,
                    field_value2,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    **kwargs,
                )
            if not field_init1 or not field_init2 or v is None:  # retain old value
                return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                raise MisconfigurationException(
                    "A frozen dataclass was passed to `apply_to_collections` but this is not allowed."
                    " HINT: is your batch a frozen dataclass?"
                ) from e
        return result

    return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)


class TransferableDataType(ABC):
    """A custom type for data that can be moved to a torch device via ``.to(...)``.

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
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented


def move_data_to_device(batch: Any, device: Union[str, torch.device]) -> Any:
    """Transfers a collection of data to the given device. Any object that defines a method ``to(device)`` will be
    moved and all other objects in the collection will be left untouched.

    Args:
        batch: A tensor or collection of tensors or anything that has a method ``.to(...)``.
            See :func:`apply_to_collection` for a list of supported collection types.
        device: The device to which the data should be moved

    Return:
        the same collection but with all contained tensors residing on the new device.

    See Also:
        - :meth:`torch.Tensor.to`
        - :class:`torch.device`
    """

    def batch_to(data: Any) -> Any:
        # try to move torchtext data first
        if _TORCHTEXT_LEGACY and isinstance(data, Batch):
            # TODO: also remove the torchtext dependency with Lightning 1.8
            rank_zero_deprecation(
                "The `torchtext.legacy.Batch` object is deprecated and Lightning will remove support for it in v1.8."
                " We recommend you to migrate away from Batch by following the TorchText README:"
                " https://github.com/pytorch/text#bc-breaking-legacy"
            )
            # Shallow copy because each Batch has a reference to Dataset which contains all examples
            device_data = copy(data)
            for field, field_value in data.dataset.fields.items():
                if field_value is None:
                    continue
                device_field = move_data_to_device(getattr(data, field), device)
                setattr(device_data, field, device_field)
            return device_data

        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        if isinstance(data, torch.Tensor) and device not in _CPU_DEVICES:
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `TransferableDataType` and forgot to return `self`.
        return data

    dtype = (TransferableDataType, Batch) if _TORCHTEXT_LEGACY else TransferableDataType
    return apply_to_collection(batch, dtype=dtype, function=batch_to)


def convert_to_tensors(data: Any, device: Union[str, torch.device]) -> Any:
    for src_dtype, conversion_func in CONVERSION_DTYPES:
        data = apply_to_collection(data, src_dtype, conversion_func, device=device)

    def _move_to_device_and_make_contiguous(t: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
        return t.to(device).contiguous()

    data = apply_to_collection(data, torch.Tensor, _move_to_device_and_make_contiguous, device=device)
    return data
