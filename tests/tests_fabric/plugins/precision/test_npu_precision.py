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
import pytest

from lightning.fabric.accelerators.npu import NPUAccelerator
from lightning.fabric.connector import _Connector
from lightning.fabric.plugins.precision.bitsandbytes import BitsandbytesPrecision
from lightning.fabric.utilities.exceptions import MisconfigurationException
from tests_fabric.helpers.runif import RunIf


@RunIf(min_npu_gpus=1)
def test_bitsandbytes_precision_not_supported_on_npu(monkeypatch):
    # bypass the bitsandbytes import/class resolution in the plugin constructor: we only exercise the
    # connector's NPU guard, which runs before the plugin is ever used
    monkeypatch.setattr(BitsandbytesPrecision, "__init__", lambda self, *args, **kwargs: None)
    precision = BitsandbytesPrecision()
    with pytest.raises(MisconfigurationException, match="bitsandbytes.*not enabled on NPU"):
        _Connector(plugins=precision, accelerator=NPUAccelerator())


@RunIf(min_npu_gpus=1)
@pytest.mark.parametrize("precision", ["transformer-engine", "transformer-engine-float16"])
def test_transformer_engine_precision_not_supported_on_npu(precision):
    with pytest.raises(MisconfigurationException, match="transformer-engine.*not enabled on NPU"):
        _Connector(precision=precision, accelerator=NPUAccelerator())