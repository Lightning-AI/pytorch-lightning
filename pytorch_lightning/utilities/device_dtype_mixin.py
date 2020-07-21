from typing import Union, Optional

import torch
from torch.nn import Module


class DeviceDtypeModuleMixin(Module):
    _device: ...
    _dtype: Union[str, torch.dtype]

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]):
        # necessary to avoid infinite recursion
        raise RuntimeError('Cannot set the dtype explicitly. Please use module.to(new_dtype).')

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

    @device.setter
    def device(self, new_device: Union[str, torch.device]):
        # Necessary to avoid infinite recursion
        raise RuntimeError('Cannot set the device explicitly. Please use module.to(new_device).')

    def to(self, *args, **kwargs) -> Module:
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
            >>> class ExampleModule(DeviceDtypeModuleMixin):
            ...     def __init__(self, weight: torch.Tensor):
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
        """
        # there is diff nb vars in PT 1.5
        out = torch._C._nn._parse_to(*args, **kwargs)
        device = out[0]
        dtype = out[1]
        if device is not None:
            self.apply(device_apply_fn(device))

        if dtype is not None:
            self.apply(dtype_apply_fn(dtype))

        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[int] = None) -> Module:
        """Moves all model parameters and buffers to the GPU.
        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device: if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        self.apply(device_apply_fn(torch.device('cuda', index=device)))
        return super().cuda(device=device)

    def cpu(self) -> Module:
        """Moves all model parameters and buffers to the CPU.
        Returns:
            Module: self
        """
        self.apply(device_apply_fn(torch.device('cpu')))
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) -> Module:
        """Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        self.apply(dtype_apply_fn(dst_type))
        return super().type(dst_type=dst_type)

    def float(self) -> Module:
        """Casts all floating point parameters and buffers to float datatype.

        Returns:
            Module: self
        """
        self.apply(dtype_apply_fn(torch.float))
        return super().float()

    def double(self) -> Module:
        """Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        """
        self.apply(dtype_apply_fn(torch.double))
        return super().double()

    def half(self) -> Module:
        """Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        """
        self.apply(dtype_apply_fn(torch.half))
        return super().half()





def dtype_apply_fn(dtype):
    return apply_attr('_dtype', dtype)


def device_apply_fn(device):
    return apply_attr('_device', device)


# def update_attributes(module, **kwargs):
#     if not isinstance(module, DeviceDtypeModuleMixin):
#         return
#     for k, v in kwargs:
#         module.__setattr__(k, v)


def apply_attr(name: str, value):

    def apply_fn(module):
        if isinstance(module, DeviceDtypeModuleMixin):
            module.__setattr__(name, value)

    return apply_fn
