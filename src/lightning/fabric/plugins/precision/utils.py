# Copyright The Lightning AI team.
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
from collections.abc import Mapping
from typing import Any, Union

import torch
from torch import Tensor


def _convert_fp_tensor(tensor: Tensor, dst_type: Union[str, torch.dtype]) -> Tensor:
    return tensor.to(dst_type) if torch.is_floating_point(tensor) else tensor


# do not support using as a decorator or else `torch.get_default_dtype` will get picked up at initialization
class _DtypeContextManager:
    """A context manager to change the default tensor type when tensors get created.

    See: :func:`torch.set_default_dtype`

    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._previous_dtype: torch.dtype = torch.get_default_dtype()
        self._new_dtype = dtype

    def __enter__(self) -> None:
        torch.set_default_dtype(self._new_dtype)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_default_dtype(self._previous_dtype)


class _ClassReplacementContextManager:
    """A context manager to monkeypatch classes."""

    def __init__(self, mapping: Mapping[str, type]) -> None:
        self._mapping = mapping
        self._originals = {}
        self._modules = {}
        for class_string in mapping:
            module_name, class_name = class_string.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            self._modules[class_string] = module
            self._originals[class_string] = getattr(module, class_name)

    def __enter__(self) -> None:
        for class_string, replacement in self._mapping.items():
            _, class_name = class_string.rsplit(".", 1)
            setattr(self._modules[class_string], class_name, replacement)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for class_string, replacement in self._mapping.items():
            _, class_name = class_string.rsplit(".", 1)
            setattr(self._modules[class_string], class_name, self._originals[class_string])
