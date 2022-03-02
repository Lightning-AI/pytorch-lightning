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
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry


def test_accelerator_registry_with_new_accelerator():

    accelerator_name = "custom_accelerator"

    class TestAccelerator(Accelerator):
        @staticmethod
        def parse_devices(devices):
            return devices

        @staticmethod
        def get_parallel_devices(devices):
            return ["foo"] * devices

        @staticmethod
        def auto_device_count():
            return 3

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def name():
            return accelerator_name

    AcceleratorRegistry.register(TestAccelerator)

    assert accelerator_name in AcceleratorRegistry
    assert isinstance(AcceleratorRegistry.get(accelerator_name), TestAccelerator)

    trainer = Trainer(accelerator=TestAccelerator(), devices="auto")
    assert isinstance(trainer.accelerator, TestAccelerator)
    assert trainer._accelerator_connector.parallel_devices == ["foo"] * 3

    AcceleratorRegistry.remove(accelerator_name)
    assert accelerator_name not in AcceleratorRegistry


def test_available_accelerators_in_registry():
    assert AcceleratorRegistry.available_accelerators() == ["cpu", "gpu", "ipu", "tpu"]
