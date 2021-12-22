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
from pytorch_lightning.plugins import (
    CheckpointIO,
    DDPFullyShardedStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    TPUSpawnStrategy,
    TrainingTypePluginsRegistry,
)
from tests.helpers.runif import RunIf


def test_training_type_plugins_registry_with_new_plugin():
    class TestPlugin:

        distributed_backend = "test_plugin"

        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

    plugin_name = "test_plugin"
    plugin_description = "Test Plugin"

    TrainingTypePluginsRegistry.register(
        plugin_name, TestPlugin, description=plugin_description, param1="abc", param2=123
    )

    assert plugin_name in TrainingTypePluginsRegistry
    assert TrainingTypePluginsRegistry[plugin_name]["description"] == plugin_description
    assert TrainingTypePluginsRegistry[plugin_name]["init_params"] == {"param1": "abc", "param2": 123}
    assert TrainingTypePluginsRegistry[plugin_name]["distributed_backend"] == "test_plugin"
    assert isinstance(TrainingTypePluginsRegistry.get(plugin_name), TestPlugin)

    TrainingTypePluginsRegistry.remove(plugin_name)
    assert plugin_name not in TrainingTypePluginsRegistry


@pytest.mark.parametrize(
    "plugin_name, init_params",
    [
        ("deepspeed", {}),
        ("deepspeed_stage_1", {"stage": 1}),
        ("deepspeed_stage_2", {"stage": 2}),
        ("deepspeed_stage_2_offload", {"stage": 2, "offload_optimizer": True}),
        ("deepspeed_stage_3", {"stage": 3}),
        ("deepspeed_stage_3_offload", {"stage": 3, "offload_parameters": True, "offload_optimizer": True}),
    ],
)
def test_training_type_plugins_registry_with_deepspeed_plugins(plugin_name, init_params):

    assert plugin_name in TrainingTypePluginsRegistry
    assert TrainingTypePluginsRegistry[plugin_name]["init_params"] == init_params
    assert TrainingTypePluginsRegistry[plugin_name]["plugin"] == DeepSpeedStrategy


@RunIf(deepspeed=True)
@pytest.mark.parametrize("plugin", ["deepspeed", "deepspeed_stage_2_offload", "deepspeed_stage_3"])
def test_deepspeed_training_type_plugins_registry_with_trainer(tmpdir, plugin):

    trainer = Trainer(default_root_dir=tmpdir, strategy=plugin, precision=16)

    assert isinstance(trainer.strategy, DeepSpeedStrategy)


def test_tpu_spawn_debug_plugins_registry(tmpdir):

    plugin = "tpu_spawn_debug"

    assert plugin in TrainingTypePluginsRegistry
    assert TrainingTypePluginsRegistry[plugin]["init_params"] == {"debug": True}
    assert TrainingTypePluginsRegistry[plugin]["plugin"] == TPUSpawnStrategy

    trainer = Trainer(strategy=plugin)

    assert isinstance(trainer.strategy, TPUSpawnStrategy)


def test_fsdp_strategys_registry(tmpdir):

    plugin = "fsdp"

    assert plugin in TrainingTypePluginsRegistry
    assert TrainingTypePluginsRegistry[plugin]["plugin"] == DDPFullyShardedStrategy

    trainer = Trainer(strategy=plugin)

    assert isinstance(trainer.strategy, DDPFullyShardedStrategy)


@pytest.mark.parametrize(
    "plugin_name, plugin",
    [
        ("ddp_find_unused_parameters_false", DDPStrategy),
        ("ddp_spawn_find_unused_parameters_false", DDPSpawnStrategy),
        ("ddp_sharded_spawn_find_unused_parameters_false", DDPSpawnShardedStrategy),
        ("ddp_sharded_find_unused_parameters_false", DDPShardedStrategy),
    ],
)
def test_ddp_find_unused_parameters_training_type_plugins_registry(tmpdir, plugin_name, plugin):

    trainer = Trainer(default_root_dir=tmpdir, strategy=plugin_name)

    assert isinstance(trainer.strategy, plugin)

    assert plugin_name in TrainingTypePluginsRegistry
    assert TrainingTypePluginsRegistry[plugin_name]["init_params"] == {"find_unused_parameters": False}
    assert TrainingTypePluginsRegistry[plugin_name]["plugin"] == plugin


def test_custom_registered_training_plugin_to_strategy():
    class CustomCheckpointIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path):
            pass

        def load_checkpoint(self, path):
            pass

        def remove_checkpoint(self, path):
            pass

    custom_checkpoint_io = CustomCheckpointIO()

    # Register the DDP Strategy with your custom CheckpointIO plugin
    TrainingTypePluginsRegistry.register(
        "ddp_custom_checkpoint_io",
        DDPStrategy,
        description="DDP Strategy with custom checkpoint io plugin",
        checkpoint_io=custom_checkpoint_io,
    )
    trainer = Trainer(strategy="ddp_custom_checkpoint_io", accelerator="cpu", devices=2)

    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.checkpoint_io == custom_checkpoint_io
