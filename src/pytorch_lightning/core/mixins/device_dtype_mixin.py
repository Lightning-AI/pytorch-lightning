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

from typing import Any, Optional, Union

import torch
from torch.nn import Module
from typing_extensions import Self

import pytorch_lightning as pl


class DeviceDtypeModuleMixin(Module):
    __jit_unused_properties__ = ["device", "dtype"]

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
    def device(self) -> Union[str, torch.device]:
        device = self._device

        # make this more explicit to always include the index
        if device.type == "cuda" and device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        return device

    def to(self, *args: Any, **kwargs: Any) -> Self:  # type: ignore[valid-type]
        """Moves and/or casts the parameters and buffers.

        This can be called as
        .. function:: to(device=None, dtype=None, non_blocking=False)
        .. function:: to(dtype, non_blocking=False)
        .. function:: to(tensor, non_blocking=False)
        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.
        See below for examples.

        Note:
            This method modifies the module in-place.

        Args:
            device: the desired device of the parameters
                and buffers in this module
            dtype: the desired floating point type of
                the floating point parameters and buffers in this module
            tensor: Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self

        Example::
            >>> from torch import Tensor
            >>> class ExampleModule(DeviceDtypeModuleMixin):
            ...     def __init__(self, weight: Tensor):
            ...         super().__init__()
            ...         self.register_buffer('weight', weight)
            >>> _ = torch.manual_seed(0)
            >>> module = ExampleModule(torch.rand(3, 4))
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]])
            >>> module.to(torch.double)
            ExampleModule()
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float64)
            >>> cpu = torch.device('cpu')
            >>> module.to(cpu, dtype=torch.half, non_blocking=True)
            ExampleModule()
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float16)
            >>> module.to(cpu)
            ExampleModule()
            >>> module.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float16)
            >>> module.device
            device(type='cpu')
            >>> module.dtype
            torch.float16
        """
        # there is diff nb vars in PT 1.5
        out = torch._C._nn._parse_to(*args, **kwargs)
        self.__update_properties(device=out[0], dtype=out[1])
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:  # type: ignore[valid-type]
        """Moves all model parameters and buffers to the GPU. This also makes associated parameters and buffers
        different objects. So it should be called before constructing optimizer if the module will live on GPU
        while being optimized.

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
        self.__update_properties(device=device)
        return super().cuda(device=device)

    def cpu(self) -> Self:  # type: ignore[valid-type]
        """Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        self.__update_properties(device=torch.device("cpu"))
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) -> Self:  # type: ignore[valid-type]
        """Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        self.__update_properties(dtype=dst_type)
        return super().type(dst_type=dst_type)

    def float(self) -> Self:  # type: ignore[valid-type]
        """Casts all floating point parameters and buffers to ``float`` datatype.

        Returns:
            Module: self
        """
        self.__update_properties(dtype=torch.float)
        return super().float()

    def double(self) -> Self:  # type: ignore[valid-type]
        """Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        """
        self.__update_properties(dtype=torch.double)
        return super().double()

    def half(self) -> Self:  # type: ignore[valid-type]
        """Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        """
        self.__update_properties(dtype=torch.half)
        return super().half()

    def __update_properties(
        self, device: Optional[torch.device] = None, dtype: Optional[Union[str, torch.dtype]] = None
    ) -> None:
        def apply_fn(module: Union["DeviceDtypeModuleMixin", Module]) -> None:
            # TODO: Find why `isinstance(module, DeviceDtypeModuleMixin)` doesn't
            # work when using `init_meta_context`.
            if not isinstance(module, (DeviceDtypeModuleMixin, pl.LightningModule)):
                return
            if device is not None:
                module._device = device
            if dtype is not None:
                module._dtype = dtype

        self.apply(apply_fn)
