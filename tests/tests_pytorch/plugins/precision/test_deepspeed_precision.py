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
from unittest import mock

import pytest

from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_invalid_precision_with_deepspeed_precision():
    with pytest.raises(ValueError, match="is not supported. `precision` must be one of"):
        DeepSpeedPrecisionPlugin(precision=64, amp_type="native")


@mock.patch("pytorch_lightning.plugins.precision.deepspeed._DEEPSPEED_GREATER_EQUAL_0_6", False)
def test_incompatible_bfloat16_raises_error_with_deepspeed_version():
    with pytest.raises(MisconfigurationException, match="is not supported with `deepspeed < v0.6`"):
        DeepSpeedPrecisionPlugin(precision="bf16", amp_type="native")
