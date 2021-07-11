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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Set, Type

import torch

from pytorch_lightning.utilities.apply_func import _is_dataclass_instance, _is_namedtuple
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()


def find_collection_types(data: Any, output: Optional[Set] = None) -> Set[Type]:
    """
    Recursively investigates a collection to detect any elements and collect types.

    Args:
        data: the collection to apply the function to

    Returns:
        The resulting collection types as a set
    """
    if output is None:
        output = set()

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        for v in data.values():
            output.add(type(v))
            find_collection_types(v, output=output)

    is_namedtuple = _is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        for d in data:
            output.add(type(d))
            find_collection_types(d, output=output)

    if _is_dataclass_instance(data):
        for field in data.__dataclass_fields__:
            output.add(type(field))
            find_collection_types(getattr(data, field), output=output)

    # data is neither of dtype, nor a collection
    return output


_PRIMITIVE_TYPES = [bool, int, str, torch.Tensor, set, torch.device, OrderedDict, torch.dtype, list, type(None), dict]


class NotPrimitiveTypes(Exception):
    pass


def validate_state_dict_to_primitive_types(state_dict: Dict, should_raise: bool = True) -> None:
    """
    This helper function enables to catch early non primitive types while serializing a checkpoint.
    """
    state_dict_types = find_collection_types(state_dict)
    for type in state_dict_types:
        if type not in _PRIMITIVE_TYPES and str(type) != "<class 'builtin_function_or_method'>":
            msg = f"Found {type} which isn't a primitive types: {_PRIMITIVE_TYPES}. "
            if should_raise:
                raise NotPrimitiveTypes(msg)
            else:
                msg += "HINT: This might cause issue when re-loading this issue as this object will be pickled."
            warning_cache.warn(msg)
