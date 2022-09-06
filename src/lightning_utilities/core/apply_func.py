# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import dataclasses
from collections import defaultdict, OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union


def is_namedtuple(obj: object) -> bool:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def is_dataclass_instance(obj: object) -> bool:
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
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

    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data):
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
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
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

    is_namedtuple_ = is_namedtuple(data1)
    is_sequence = isinstance(data1, Sequence) and not isinstance(data1, str)
    if (is_namedtuple_ or is_sequence) and data2 is not None:
        assert len(data1) == len(data2), "Sequence collections have different sizes."
        out = [
            apply_to_collections(v1, v2, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            for v1, v2 in zip(data1, data2)
        ]
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data1) and data2 is not None:
        if not is_dataclass_instance(data2):
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
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collections` but this is not allowed."
                ) from e
        return result

    return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
