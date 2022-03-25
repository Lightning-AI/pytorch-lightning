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

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.accelerators.registry import ACCELERATOR_REGISTRY
from pytorch_lightning.trainer.connectors.accelerator_connector import _populate_registries


@pytest.fixture(autouse=True)
def clear_registries():
    # since the registries are global, it's good to clear them after each test to avoid unwanted interactions
    yield
    ACCELERATOR_REGISTRY.clear()


def test_accelerator_registry_with_new_accelerator():
    name = "custom"
    description = "My custom Accelerator"

    class CustomAccelerator(Accelerator):
        def __init__(self, param1=None, param2=None):
            self.param1 = param1
            self.param2 = param2
            super().__init__()

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
            return "custom"

    ACCELERATOR_REGISTRY.register(CustomAccelerator, name=name, description=description, param1="abc")

    assert name in ACCELERATOR_REGISTRY
    assert ACCELERATOR_REGISTRY[name] == {
        "accelerator": CustomAccelerator,
        "description": description,
        "kwargs": {"param1": "abc"},
    }
    instance = ACCELERATOR_REGISTRY.get(name)
    assert isinstance(instance, CustomAccelerator)
    assert instance.param1 == "abc"

    assert ACCELERATOR_REGISTRY.get("foo", 123) == 123

    ACCELERATOR_REGISTRY.clear()

    trainer = Trainer(accelerator=name, devices="auto")
    assert isinstance(trainer.accelerator, CustomAccelerator)
    assert trainer.strategy.parallel_devices == ["foo"] * 3

    @ACCELERATOR_REGISTRY
    class NewAccelerator(CustomAccelerator):
        @staticmethod
        def name():
            return "new"

    assert "new" in ACCELERATOR_REGISTRY


def test_available_accelerators_in_registry():
    _populate_registries()
    assert ACCELERATOR_REGISTRY.names == ["cpu", "gpu", "hpu", "ipu", "tpu"]
