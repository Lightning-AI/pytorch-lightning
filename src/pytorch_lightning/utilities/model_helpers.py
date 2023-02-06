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
from typing import Any, Optional, Type

from lightning_utilities.core.imports import RequirementCache
from torch import nn

import pytorch_lightning as pl


def is_overridden(method_name: str, instance: Optional[object] = None, parent: Optional[Type[object]] = None) -> bool:
    if instance is None:
        # if `self.lightning_module` was passed as instance, it can be `None`
        return False
    if parent is None:
        if isinstance(instance, pl.LightningModule):
            parent = pl.LightningModule
        elif isinstance(instance, pl.LightningDataModule):
            parent = pl.LightningDataModule
        elif isinstance(instance, pl.Callback):
            parent = pl.Callback
        if parent is None:
            raise ValueError("Expected a parent")
    from lightning_utilities.core.overrides import is_overridden

    return is_overridden(method_name, instance, parent)


def get_torchvision_model(model_name: str, **kwargs: Any) -> nn.Module:
    from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

    if not _TORCHVISION_AVAILABLE:
        raise ModuleNotFoundError(str(_TORCHVISION_AVAILABLE))

    from torchvision import models

    torchvision_greater_equal_0_14 = RequirementCache("torchvision>=0.14.0")
    # TODO: deprecate this function when 0.14 is the minimum supported torchvision
    if torchvision_greater_equal_0_14:
        return models.get_model(model_name, **kwargs)
    return getattr(models, model_name)(**kwargs)
