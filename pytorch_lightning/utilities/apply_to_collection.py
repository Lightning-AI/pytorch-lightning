from collections import Mapping, Sequence
from typing import Any, Callable, Union


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
