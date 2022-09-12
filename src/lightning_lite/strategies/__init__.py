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
from lightning_lite.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from lightning_lite.strategies.fairscale import DDPSpawnShardedStrategy  # noqa: F401
from lightning_lite.strategies.parallel import ParallelStrategy  # noqa: F401
from lightning_lite.strategies.registry import _StrategyRegistry, call_register_strategies
from lightning_lite.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from lightning_lite.strategies.strategy import Strategy  # noqa: F401
from lightning_lite.strategies.tpu_spawn import TPUSpawnStrategy  # noqa: F401

_STRATEGIES_BASE_MODULE = "lightning_lite.strategies"
STRATEGY_REGISTRY = _StrategyRegistry()
call_register_strategies(STRATEGY_REGISTRY, _STRATEGIES_BASE_MODULE)
