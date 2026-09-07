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
from typing import Any

import torch

from lightning.fabric.accelerators import ACCELERATOR_REGISTRY, Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry


class TestAccelerator(Accelerator):
    """Helper accelerator class for testing."""

    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2
        super().__init__()

    def setup_device(self, device: torch.device) -> None:
        pass

    def teardown(self) -> None:
        pass

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
        return "test_accelerator"


def test_accelerator_registry_with_new_accelerator():
    accelerator_name = "custom_accelerator"
    accelerator_description = "Custom Accelerator"

    class CustomAccelerator(Accelerator):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
            super().__init__()

        def setup_device(self, device: torch.device) -> None:
            pass

        def get_device_stats(self, device: torch.device) -> dict[str, Any]:
            pass

        def teardown(self) -> None:
            pass

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
            return "custom_accelerator"

    ACCELERATOR_REGISTRY.register(
        accelerator_name, CustomAccelerator, description=accelerator_description, param1="abc", param2=123
    )

    assert accelerator_name in ACCELERATOR_REGISTRY

    assert ACCELERATOR_REGISTRY[accelerator_name]["description"] == accelerator_description
    assert ACCELERATOR_REGISTRY[accelerator_name]["init_params"] == {"param1": "abc", "param2": 123}
    assert ACCELERATOR_REGISTRY[accelerator_name]["accelerator_name"] == accelerator_name

    assert isinstance(ACCELERATOR_REGISTRY.get(accelerator_name), CustomAccelerator)

    ACCELERATOR_REGISTRY.remove(accelerator_name)
    assert accelerator_name not in ACCELERATOR_REGISTRY


def test_available_accelerators_in_registry():
    assert ACCELERATOR_REGISTRY.available_accelerators() == {"cpu", "cuda", "mps", "tpu"}


def test_registry_as_decorator():
    """Test that the registry can be used as a decorator."""
    test_registry = _AcceleratorRegistry()

    # Test decorator usage
    @test_registry.register("test_decorator", description="Test decorator accelerator", param1="value1", param2=42)
    class DecoratorAccelerator(TestAccelerator):
        pass

    # Verify registration worked
    assert "test_decorator" in test_registry
    assert test_registry["test_decorator"]["description"] == "Test decorator accelerator"
    assert test_registry["test_decorator"]["init_params"] == {"param1": "value1", "param2": 42}
    assert test_registry["test_decorator"]["accelerator"] == DecoratorAccelerator
    assert test_registry["test_decorator"]["accelerator_name"] == "test_decorator"

    # Test that we can instantiate the accelerator
    instance = test_registry.get("test_decorator")
    assert isinstance(instance, DecoratorAccelerator)
    assert instance.param1 == "value1"
    assert instance.param2 == 42


def test_registry_as_static_method():
    """Test that the registry can be used as a static method call."""
    test_registry = _AcceleratorRegistry()

    class StaticMethodAccelerator(TestAccelerator):
        pass

    # Test static method usage
    result = test_registry.register(
        "test_static",
        StaticMethodAccelerator,
        description="Test static method accelerator",
        param1="static_value",
        param2=100,
    )

    # Verify registration worked
    assert "test_static" in test_registry
    assert test_registry["test_static"]["description"] == "Test static method accelerator"
    assert test_registry["test_static"]["init_params"] == {"param1": "static_value", "param2": 100}
    assert test_registry["test_static"]["accelerator"] == StaticMethodAccelerator
    assert test_registry["test_static"]["accelerator_name"] == "test_static"
    assert result == StaticMethodAccelerator  # Should return the accelerator class

    # Test that we can instantiate the accelerator
    instance = test_registry.get("test_static")
    assert isinstance(instance, StaticMethodAccelerator)
    assert instance.param1 == "static_value"
    assert instance.param2 == 100


def test_registry_without_parameters():
    """Test registration without init parameters."""
    test_registry = _AcceleratorRegistry()

    class SimpleAccelerator(TestAccelerator):
        def __init__(self):
            super().__init__()

    test_registry.register("simple", SimpleAccelerator, description="Simple accelerator")

    assert "simple" in test_registry
    assert test_registry["simple"]["description"] == "Simple accelerator"
    assert test_registry["simple"]["init_params"] == {}

    instance = test_registry.get("simple")
    assert isinstance(instance, SimpleAccelerator)
