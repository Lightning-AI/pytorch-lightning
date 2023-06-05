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
import torch

from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.overrides.base import _LightningPrecisionModuleWrapperBase


def test_wrapper_device_dtype():
    model = BoringModel()
    wrapped_model = _LightningPrecisionModuleWrapperBase(model)

    wrapped_model.to(dtype=torch.float16)
    assert model.dtype == torch.float16
