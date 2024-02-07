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

from lightning.fabric.strategies import STRATEGY_REGISTRY


def test_strategy_registry_with_new_strategy():
    class TestStrategy:
        strategy_name = "test_strategy"

        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

    strategy_name = "test_strategy"
    strategy_description = "Test Strategy"

    # TODO(fabric): Registering classes that do not inherit from Strategy should not be allowed
    STRATEGY_REGISTRY.register(strategy_name, TestStrategy, description=strategy_description, param1="abc", param2=123)

    assert strategy_name in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY[strategy_name]["description"] == strategy_description
    assert STRATEGY_REGISTRY[strategy_name]["init_params"] == {"param1": "abc", "param2": 123}
    assert STRATEGY_REGISTRY[strategy_name]["strategy_name"] == "test_strategy"
    assert isinstance(STRATEGY_REGISTRY.get(strategy_name), TestStrategy)

    STRATEGY_REGISTRY.remove(strategy_name)
    assert strategy_name not in STRATEGY_REGISTRY


def test_available_strategies_in_registry():
    expected = {
        "ddp",
        "deepspeed",
        "deepspeed_stage_1",
        "deepspeed_stage_1_offload",
        "deepspeed_stage_2",
        "deepspeed_stage_2_offload",
        "deepspeed_stage_3",
        "deepspeed_stage_3_offload",
        "deepspeed_stage_3_offload_nvme",
        "ddp_spawn",
        "ddp_fork",
        "ddp_notebook",
        "single_tpu",  # legacy
        "single_xla",
        "xla",
        "xla_fsdp",
        "dp",
        "fsdp",
        "fsdp_cpu_offload",
    }
    assert set(STRATEGY_REGISTRY.available_strategies()) == expected
