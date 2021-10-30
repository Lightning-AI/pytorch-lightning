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
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torchmetrics import Metric

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

__all__ = ["LightningSystem"]


class LightningSystem(LightningModule):
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
                print("1", name, value)
                if modules is None:
                    raise AttributeError("cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                if not isinstance(value, Metric):
                    if sum(not isinstance(m, Metric) for m in modules) > 0:
                        raise MisconfigurationException(
                            "A `LightningSystem` supports' only a single nn.Module expects `torchmetrics.Metric`."
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
    def _has_module(self) -> bool:
        modules = self.__dict__.get("_modules")
        return sum(not isinstance(m, Metric) for m in modules) > 0

    @property
    def _name_module(self) -> Optional[str]:
        if self._has_module:
            modules = self.__dict__.get("_modules")
            return [not isinstance(m, Metric) for m in modules][0]

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict()
        if self._has_module:
            return state_dict
        return state_dict
