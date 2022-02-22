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
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.strategies import (
    DDPFullyShardedStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    StrategyRegistry,
    TPUSpawnStrategy,
)
from tests.helpers.runif import RunIf


def test_strategy_registry_with_new_strategy():
    class TestStrategy:

        strategy_name = "test_strategy"

        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

    strategy_name = "test_strategy"
    strategy_description = "Test Strategy"

    StrategyRegistry.register(strategy_name, TestStrategy, description=strategy_description, param1="abc", param2=123)

    assert strategy_name in StrategyRegistry
    assert StrategyRegistry[strategy_name]["description"] == strategy_description
    assert StrategyRegistry[strategy_name]["init_params"] == {"param1": "abc", "param2": 123}
    assert StrategyRegistry[strategy_name]["strategy_name"] == "test_strategy"
    assert isinstance(StrategyRegistry.get(strategy_name), TestStrategy)

    StrategyRegistry.remove(strategy_name)
    assert strategy_name not in StrategyRegistry


@pytest.mark.parametrize(
    "strategy_name, init_params",
    [
        ("deepspeed", {}),
        ("deepspeed_stage_1", {"stage": 1}),
        ("deepspeed_stage_2", {"stage": 2}),
        ("deepspeed_stage_2_offload", {"stage": 2, "offload_optimizer": True}),
        ("deepspeed_stage_3", {"stage": 3}),
        ("deepspeed_stage_3_offload", {"stage": 3, "offload_parameters": True, "offload_optimizer": True}),
    ],
)
def test_strategy_registry_with_deepspeed_strategies(strategy_name, init_params):

    assert strategy_name in StrategyRegistry
    assert StrategyRegistry[strategy_name]["init_params"] == init_params
    assert StrategyRegistry[strategy_name]["strategy"] == DeepSpeedStrategy


@RunIf(deepspeed=True)
@pytest.mark.parametrize("strategy", ["deepspeed", "deepspeed_stage_2_offload", "deepspeed_stage_3"])
def test_deepspeed_strategy_registry_with_trainer(tmpdir, strategy):

    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy, precision=16)

    assert isinstance(trainer.strategy, DeepSpeedStrategy)


def test_tpu_spawn_debug_strategy_registry(tmpdir):

    strategy = "tpu_spawn_debug"

    assert strategy in StrategyRegistry
    assert StrategyRegistry[strategy]["init_params"] == {"debug": True}
    assert StrategyRegistry[strategy]["strategy"] == TPUSpawnStrategy

    trainer = Trainer(strategy=strategy)

    assert isinstance(trainer.strategy, TPUSpawnStrategy)


def test_fsdp_strategy_registry(tmpdir):

    strategy = "fsdp"

    assert strategy in StrategyRegistry
    assert StrategyRegistry[strategy]["strategy"] == DDPFullyShardedStrategy

    trainer = Trainer(strategy=strategy)

    assert isinstance(trainer.strategy, DDPFullyShardedStrategy)


@pytest.mark.parametrize(
    "strategy_name, strategy",
    [
        ("ddp_find_unused_parameters_false", DDPStrategy),
        ("ddp_spawn_find_unused_parameters_false", DDPSpawnStrategy),
        ("ddp_sharded_spawn_find_unused_parameters_false", DDPSpawnShardedStrategy),
        ("ddp_sharded_find_unused_parameters_false", DDPShardedStrategy),
    ],
)
def test_ddp_find_unused_parameters_strategy_registry(tmpdir, strategy_name, strategy):

    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy_name)

    assert isinstance(trainer.strategy, strategy)

    assert strategy_name in StrategyRegistry
    assert StrategyRegistry[strategy_name]["init_params"] == {"find_unused_parameters": False}
    assert StrategyRegistry[strategy_name]["strategy"] == strategy


def test_custom_registered_strategy_to_strategy_flag():
    class CustomCheckpointIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path):
            pass

        def load_checkpoint(self, path):
            pass

        def remove_checkpoint(self, path):
            pass

    custom_checkpoint_io = CustomCheckpointIO()

    # Register the DDP Strategy with your custom CheckpointIO plugin
    StrategyRegistry.register(
        "ddp_custom_checkpoint_io",
        DDPStrategy,
        description="DDP Strategy with custom checkpoint io plugin",
        checkpoint_io=custom_checkpoint_io,
    )
    trainer = Trainer(strategy="ddp_custom_checkpoint_io", accelerator="cpu", devices=2)

    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.checkpoint_io == custom_checkpoint_io
