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
from typing import Mapping, Optional, Union

from lightning_utilities.core.imports import module_available
from torch import Tensor
from torch.nn import Module, Parameter


def _is_deferred(module: Optional[Module]) -> bool:
    if module is None or not module_available("torchdistx.fake"):
        return False
    from torchdistx.fake import is_fake

    def any_fake(tensors: Mapping[str, Optional[Union[Tensor, Parameter]]]) -> bool:
        return any(is_fake(t) for t in tensors.values() if t is not None)

    is_deferred = any(_is_deferred(m) for m in module.children())
    return is_deferred or any_fake(module._parameters) or any_fake(module._buffers)
