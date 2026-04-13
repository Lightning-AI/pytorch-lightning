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
import sys

from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.utilities.registry import _register_classes
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from lightning.pytorch.strategies.single_xla import SingleDeviceXLAStrategy  # noqa: F401
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.strategies.xla import XLAStrategy  # noqa: F401

StrategyRegistry = _StrategyRegistry()
_register_classes(StrategyRegistry, "register_strategies", sys.modules[__name__], Strategy)

__all__ = [
    "DDPStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    "ModelParallelStrategy",
    "ParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy",
]
