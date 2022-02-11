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
from typing import Any, Dict, Generator, Iterator, Optional, Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins import PrecisionPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device


class _LiteOptimizer(LightningOptimizer):
    def state_dict(self) -> Dict[str, Tensor]:
        assert self._strategy is not None
        return self._strategy.optimizer_state(self.optimizer)


class _LiteModule(DeviceDtypeModuleMixin):
    def __init__(self, module: nn.Module, precision_plugin: PrecisionPlugin) -> None:
        """The LiteModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            module: The module to wrap
            precision_plugin: Reference to the precision plugin for handling precision context
        """
        super().__init__()
        self._module = module
        self._precision_plugin = precision_plugin

    @property
    def module(self) -> nn.Module:
        return self._module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward
        method."""
        precision = self._precision_plugin.precision
        precision_to_type = {
            "bf16": torch.bfloat16,
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }
        # TODO (@awaelchli): let the precision plugin handle the conversion
        to_type = precision_to_type[precision]

        def _convert_float_tensor(t: Tensor) -> Tensor:
            return t.to(to_type) if torch.is_floating_point(t) else t

        args, kwargs = apply_to_collection([args, kwargs], function=_convert_float_tensor, dtype=Tensor)

        with self._precision_plugin.forward_context():
            output = self.module(*args, **kwargs)

        to_type = torch.get_default_dtype()
        output = apply_to_collection(output, function=_convert_float_tensor, dtype=Tensor)
        return output


class _LiteDataLoader:
    def __init__(self, dataloader: DataLoader, device: Optional[torch.device] = None) -> None:
        """The LiteDataLoader is a wrapper for the :class:`~torch.utils.data.DataLoader`. It moves the data to the
        device automatically if the device is specified.

        Args:
            dataloader: The dataloader to wrap
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).
        """
        self.__dict__.update(dataloader.__dict__)
        self._dataloader = dataloader
        self._device = device

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        iterator = iter(self._dataloader)
        if self._device is None:
            yield from iterator
            return

        for item in iterator:
            yield move_data_to_device(item, self._device)
