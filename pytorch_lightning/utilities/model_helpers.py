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
from functools import partial
from typing import Optional, Type, Union
from unittest.mock import Mock

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_deprecation


def is_overridden(
    method_name: str,
    instance: Optional[object] = None,
    parent: Optional[Type[object]] = None,
    model: Optional[Union["pl.LightningModule", "pl.LightningDataModule"]] = None,
) -> bool:
    if model is not None and instance is None:
        rank_zero_deprecation(
            "`is_overriden(model=...)` has been deprecated and will be removed in v1.6."
            "Please use `is_overriden(instance=...)`"
        )
        instance = model

    if instance is None:
        # if `self.lightning_module` was passed as instance, it can be `None`
        return False

    if parent is None:
        if isinstance(instance, pl.LightningModule):
            parent = pl.LightningModule
        elif isinstance(instance, pl.LightningDataModule):
            parent = pl.LightningDataModule
        if parent is None:
            raise ValueError("Expected a parent")

    instance_attr = getattr(instance, method_name, None)
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

    # cannot pickle `__code__` so cannot verify if `PatchDataloader`
    # exists which shows dataloader methods have been overwritten.
    # so, we hack it by using the string representation
    instance_code = getattr(instance_attr, "patch_loader_code", None) or instance_attr.__code__
    parent_code = parent_attr.__code__

    return instance_code != parent_code
