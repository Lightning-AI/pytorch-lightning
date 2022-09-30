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
from lightning_lite.strategies.ddp import DDPStrategy  # noqa: F401
from lightning_lite.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from lightning_lite.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from lightning_lite.strategies.dp import DataParallelStrategy  # noqa: F401
from lightning_lite.strategies.fairscale import DDPShardedStrategy  # noqa: F401
from lightning_lite.strategies.fairscale import DDPSpawnShardedStrategy  # noqa: F401
from lightning_lite.strategies.parallel import ParallelStrategy  # noqa: F401
from lightning_lite.strategies.registry import _call_register_strategies, _StrategyRegistry
from lightning_lite.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from lightning_lite.strategies.single_tpu import SingleTPUStrategy  # noqa: F401
from lightning_lite.strategies.strategy import Strategy  # noqa: F401
from lightning_lite.strategies.xla import XLAStrategy  # noqa: F401

STRATEGY_REGISTRY = _StrategyRegistry()
_STRATEGIES_BASE_MODULE = "lightning_lite.strategies"
_call_register_strategies(STRATEGY_REGISTRY, _STRATEGIES_BASE_MODULE)
