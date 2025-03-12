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

from lightning.fabric.strategies.ddp import DDPStrategy  # noqa: F401
from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from lightning.fabric.strategies.dp import DataParallelStrategy  # noqa: F401
from lightning.fabric.strategies.fsdp import FSDPStrategy  # noqa: F401
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy  # noqa: F401
from lightning.fabric.strategies.parallel import ParallelStrategy  # noqa: F401
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from lightning.fabric.strategies.single_xla import SingleDeviceXLAStrategy  # noqa: F401
from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.xla import XLAStrategy  # noqa: F401
from lightning.fabric.strategies.xla_fsdp import XLAFSDPStrategy  # noqa: F401
from lightning.fabric.utilities.registry import _register_classes

STRATEGY_REGISTRY = _StrategyRegistry()
_register_classes(STRATEGY_REGISTRY, "register_strategies", sys.modules[__name__], Strategy)
