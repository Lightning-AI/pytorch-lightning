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
import pytest
import torch
from torch.nn import DataParallel

from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
    unwrap_lightning_module,
)
from tests.helpers import BoringModel


@pytest.mark.parametrize("wrapper_class", [
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
])
def test_wrapper_device_dtype(wrapper_class):
    model = BoringModel()
    wrapped_model = wrapper_class(model)

    wrapped_model.to(dtype=torch.float16)
    assert model.dtype == torch.float16


def test_unwrap_lightning_module():
    model = BoringModel()
    wrapped_model = _LightningPrecisionModuleWrapperBase(model)
    wrapped_model = _LightningModuleWrapperBase(wrapped_model)
    wrapped_model = DataParallel(wrapped_model)

    assert unwrap_lightning_module(wrapped_model) == model
