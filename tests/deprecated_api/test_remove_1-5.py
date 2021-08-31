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
"""Test deprecated functionality which will be removed in v1.5.0"""
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
from tests.deprecated_api import no_deprecated_call
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.runif import RunIf


def test_v1_5_0_datamodule_setter():
    model = BoringModel()
    datamodule = BoringDataModule()
    with no_deprecated_call(match="The `LightningModule.datamodule`"):
        model.datamodule = datamodule
    from pytorch_lightning.core.lightning import warning_cache

    warning_cache.clear()
    _ = model.datamodule
    assert any("The `LightningModule.datamodule`" in w for w in warning_cache)


@RunIf(deepspeed=True)
@pytest.mark.parametrize(
    "params", [dict(cpu_offload=True), dict(cpu_offload_params=True), dict(cpu_offload_use_pin_memory=True)]
)
def test_v1_5_0_deepspeed_cpu_offload(tmpdir, params):

    with pytest.deprecated_call(match="is deprecated since v1.4 and will be removed in v1.5"):
        DeepSpeedPlugin(**params)


def test_v1_5_0_distributed_backend_trainer_flag():
    with pytest.deprecated_call(match="has been deprecated and will be removed in v1.5."):
        Trainer(distributed_backend="ddp_cpu")
