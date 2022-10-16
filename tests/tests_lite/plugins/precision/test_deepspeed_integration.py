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
from tests_lite.helpers.runif import RunIf

from lightning_lite.connector import _Connector
from lightning_lite.plugins import DeepSpeedPrecision
from lightning_lite.strategies import DeepSpeedStrategy


@RunIf(deepspeed=True)
@pytest.mark.parametrize("precision", ["bf16", 16, 32])
def test_deepspeed_precision_choice(precision, tmpdir):
    """Test to ensure precision plugin is correctly chosen.

    DeepSpeed handles precision via custom DeepSpeedPrecision.
    """
    connector = _Connector(
        accelerator="gpu",
        strategy="deepspeed",
        precision=precision,
    )

    assert isinstance(connector.strategy, DeepSpeedStrategy)
    assert isinstance(connector.strategy.precision, DeepSpeedPrecision)
    assert connector.strategy.precision.precision == precision
