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

from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.strategies.single_xla import SingleDeviceXLAStrategy
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation


class SingleTPUStrategy(SingleDeviceXLAStrategy):
    """Legacy class.

    Use `SingleDeviceXLAStrategy` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        rank_zero_deprecation("The 'single_tpu' strategy is deprecated. Use 'single_xla' instead.")
        super().__init__(*args, **kwargs)

    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("single_tpu", cls, description="Legacy class. Use `single_xla` instead.")
