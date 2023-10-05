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
from typing import Any, Dict

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.strategies import DDPStrategy


def test_pluggable_accelerator(mps_count_0, cuda_count_2):
    class TestAccelerator(Accelerator):
        def setup_device(self, device: torch.device) -> None:
            pass

        def get_device_stats(self, device: torch.device) -> Dict[str, Any]:
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
            return "custom_acc_name"

    trainer = Trainer(accelerator=TestAccelerator(), devices=2, strategy="ddp")
    assert isinstance(trainer.accelerator, TestAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.parallel_devices == ["foo"] * 2

    trainer = Trainer(strategy=DDPStrategy(TestAccelerator()), devices="auto")
    assert isinstance(trainer.accelerator, TestAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.parallel_devices == ["foo"] * 3
