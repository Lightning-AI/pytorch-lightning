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

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator, IPUAccelerator, TPUAccelerator
from tests.helpers.boring_model import BoringModel


@mock.patch("torch.cuda.device_count", return_value=2)
def test_auto_device_count(device_count_mock):
    assert CPUAccelerator.auto_device_count() == 1
    assert GPUAccelerator.auto_device_count() == 2
    assert TPUAccelerator.auto_device_count() == 8
    assert IPUAccelerator.auto_device_count() == 4


def test_pluggable_accelerator(tmpdir):
    class TestAccelerator(Accelerator):
        @staticmethod
        def parse_devices(devices) -> int:
            """Accelerator Parsing logic."""
            return devices

        @staticmethod
        def get_parallel_devices(devices):
            return [torch.device("cpu")] * devices

        @staticmethod
        def auto_device_count() -> int:
            """Get the devices when set to auto."""
            return 1

        @staticmethod
        def is_available() -> bool:
            return True

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        accelerator=TestAccelerator(),
        devices=2,
        strategy="ddp",
    )
    trainer.fit(model)

    assert isinstance(trainer.accelerator, TestAccelerator)
    assert trainer._accelerator_connector.parallel_devices == [torch.device("cpu")] * 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        strategy=DDPStrategy(TestAccelerator()),
        devices=2,
    )
    trainer.fit(model)

    assert isinstance(trainer.accelerator, TestAccelerator)
    assert trainer._accelerator_connector.parallel_devices == [torch.device("cpu")] * 2
