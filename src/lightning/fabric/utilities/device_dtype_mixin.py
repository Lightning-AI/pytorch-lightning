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

from typing import Any, Optional, Union

import torch
from torch.nn import Module
from typing_extensions import Self, override


class _DeviceDtypeModuleMixin(Module):
    __jit_unused_properties__: list[str] = ["device", "dtype"]

    def __init__(self) -> None:
        super().__init__()
        self._dtype: Union[str, torch.dtype] = torch.get_default_dtype()
        self._device = torch.device("cpu")

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]) -> None:
        # necessary to avoid infinite recursion
        raise RuntimeError("Cannot set the dtype explicitly. Please use module.to(new_dtype).")

    @property
    def device(self) -> torch.device:
        device = self._device

        # make this more explicit to always include the index
        if device.type == "cuda" and device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        return device

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        """See :meth:`torch.nn.Module.to`."""
        # this converts `str` device to `torch.device`
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        _update_properties(self, device=device, dtype=dtype)
        return super().to(*args, **kwargs)

    @override
    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
        """Moves all model parameters and buffers to the GPU. This also makes associated parameters and buffers
        different objects. So it should be called before constructing optimizer if the module will live on GPU while
        being optimized.

        Arguments:
            device: If specified, all parameters will be copied to that device. If `None`, the current CUDA device
                index will be used.

        Returns:
            Module: self

        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        _update_properties(self, device=device)
        return super().cuda(device=device)

    @override
    def cpu(self) -> Self:
        """See :meth:`torch.nn.Module.cpu`."""
        _update_properties(self, device=torch.device("cpu"))
        return super().cpu()

    @override
    def type(self, dst_type: Union[str, torch.dtype]) -> Self:
        """See :meth:`torch.nn.Module.type`."""
        _update_properties(self, dtype=dst_type)
        return super().type(dst_type=dst_type)

    @override
    def float(self) -> Self:
        """See :meth:`torch.nn.Module.float`."""
        _update_properties(self, dtype=torch.float)
        return super().float()

    @override
    def double(self) -> Self:
        """See :meth:`torch.nn.Module.double`."""
        _update_properties(self, dtype=torch.double)
        return super().double()

    @override
    def half(self) -> Self:
        """See :meth:`torch.nn.Module.half`."""
        _update_properties(self, dtype=torch.half)
        return super().half()


def _update_properties(
    root: torch.nn.Module, device: Optional[torch.device] = None, dtype: Optional[Union[str, torch.dtype]] = None
) -> None:
    for module in root.modules():
        if not isinstance(module, _DeviceDtypeModuleMixin):
            continue
        # cannot use `module.to()` because we don't actually want to move the model in case there are multiple
        # devices types (such as partial meta parameters)
        if device is not None:
            module._device = device
        if dtype is not None:
            module._dtype = dtype
