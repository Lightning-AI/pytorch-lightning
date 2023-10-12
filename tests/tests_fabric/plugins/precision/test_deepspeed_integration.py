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
from unittest import mock

import pytest
from lightning.fabric.connector import _Connector
from lightning.fabric.plugins import DeepSpeedPrecision
from lightning.fabric.strategies import DeepSpeedStrategy

from tests_fabric.helpers.runif import RunIf


@RunIf(deepspeed=True)
@pytest.mark.parametrize("precision", ["bf16-mixed", "16-mixed", "32-true"])
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_deepspeed_precision_choice(_, precision):
    """Test to ensure precision plugin is correctly chosen.

    DeepSpeed handles precision via custom DeepSpeedPrecision.

    """
    connector = _Connector(
        accelerator="auto",
        strategy="deepspeed",
        precision=precision,
    )

    assert isinstance(connector.strategy, DeepSpeedStrategy)
    assert isinstance(connector.strategy.precision, DeepSpeedPrecision)
    assert connector.strategy.precision.precision == str(precision)
