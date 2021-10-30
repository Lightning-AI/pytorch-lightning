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
"""The LightningSystem - an LightningModule """
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torchmetrics import Metric

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

__all__ = ["LightningSystem"]


class LightningSystem(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_modules = OrderedDict()

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if isinstance(value, Parameter):
            raise MisconfigurationException("A `LightningSystem` doesn't support parameters.")
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError("cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                if isinstance(value, Metric):
                    self._metric_modules[name] = value
                elif len(modules) > 0:
                    raise MisconfigurationException(
                        "A `LightningSystem` supports only a single nn.Module expects `torchmetrics.Metric`."
                    )
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as child module '{}' "
                        "(torch.nn.Module or None expected)".format(torch.typename(value), name)
                    )
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if isinstance(buffers, Tensor):
                    raise MisconfigurationException("A `LightningSystem` doesn't support parameters.")
                else:
                    object.__setattr__(self, name, value)

    @property
    def has_module(self) -> bool:
        return len(self.__dict__.get("_modules")) > 0

    @property
    def module_name(self) -> Optional[str]:
        if self.has_module:
            modules: OrderedDict = self.__dict__.get("_modules")
            return list(modules.keys())[0]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict()
        if self.has_module:
            module_name = self.module_name

            def prune_name(k: str) -> str:
                if k.startswith(module_name):
                    return k.replace(module_name + ".", "")
                return k

            return {prune_name(k): v for k, v in state_dict.items()}
        return state_dict

    def load_state_dict(self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True):
        if not self.has_module:
            raise MisconfigurationException("A `LightningSystem doesn't contain any module.")
        module_name = self.module_name

        def add_name(k: str) -> str:
            if "." not in k:
                return module_name + "." + k
            return k

        state_dict = OrderedDict({add_name(k): v for k, v in state_dict.items()})
        return super().load_state_dict(state_dict, strict=strict)
