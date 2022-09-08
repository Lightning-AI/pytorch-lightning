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
import operator
from functools import partial
from typing import Any, Optional, Type
from unittest.mock import Mock

from torch import nn

import pytorch_lightning as pl
from lightning_utilities.core.imports import compare_version
from pytorch_lightning.utilities.exceptions import MisconfigurationException


_TORCHVISION_GREATER_EQUAL_0_14 = compare_version("torchvision", operator.ge, "0.14.0")


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

    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")

    return instance_attr.__code__ != parent_attr.__code__


def get_torchvision_model(model_name: str, **kwargs: Any) -> nn.Module:
    from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE

    if not _TORCHVISION_AVAILABLE:
        raise MisconfigurationException("You have asked for TorchVision but `torchvision` is not installed.")

    from torchvision import models

    if _TORCHVISION_GREATER_EQUAL_0_14:
        return models.get_model(model_name, **kwargs)
    else:
        return getattr(models, model_name)(**kwargs)
