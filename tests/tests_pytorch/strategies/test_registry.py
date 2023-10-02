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
from unittest import mock

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.plugins import CheckpointIO
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, StrategyRegistry, XLAStrategy

from tests_pytorch.helpers.runif import RunIf


@pytest.mark.parametrize(
    ("strategy_name", "init_params"),
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
    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy, precision="16-mixed")

    assert isinstance(trainer.strategy, DeepSpeedStrategy)


@RunIf(skip_windows=True)
@mock.patch("lightning.pytorch.strategies.xla.XLAStrategy.set_world_ranks")
def test_xla_debug_strategy_registry(_, tpu_available):
    strategy = "xla_debug"

    assert strategy in StrategyRegistry
    assert StrategyRegistry[strategy]["init_params"] == {"debug": True}
    assert StrategyRegistry[strategy]["strategy"] == XLAStrategy

    trainer = Trainer(strategy=strategy)
    assert isinstance(trainer.strategy, XLAStrategy)


def test_fsdp_strategy_registry(cuda_count_1):
    strategy = "fsdp"
    assert strategy in StrategyRegistry
    assert StrategyRegistry[strategy]["strategy"] == FSDPStrategy

    trainer = Trainer(accelerator="cuda", strategy=strategy)
    assert isinstance(trainer.strategy, FSDPStrategy)


@pytest.mark.parametrize(
    ("strategy_name", "strategy", "expected_init_params"),
    [
        (
            "ddp_find_unused_parameters_false",
            DDPStrategy,
            {"find_unused_parameters": False, "start_method": "popen"},
        ),
        (
            "ddp_find_unused_parameters_true",
            DDPStrategy,
            {"find_unused_parameters": True, "start_method": "popen"},
        ),
        (
            "ddp_spawn_find_unused_parameters_false",
            DDPStrategy,
            {"find_unused_parameters": False, "start_method": "spawn"},
        ),
        (
            "ddp_spawn_find_unused_parameters_true",
            DDPStrategy,
            {"find_unused_parameters": True, "start_method": "spawn"},
        ),
        pytest.param(
            "ddp_fork_find_unused_parameters_false",
            DDPStrategy,
            {"find_unused_parameters": False, "start_method": "fork"},
            marks=RunIf(skip_windows=True),
        ),
        pytest.param(
            "ddp_fork_find_unused_parameters_true",
            DDPStrategy,
            {"find_unused_parameters": True, "start_method": "fork"},
            marks=RunIf(skip_windows=True),
        ),
        pytest.param(
            "ddp_notebook_find_unused_parameters_false",
            DDPStrategy,
            {"find_unused_parameters": False, "start_method": "fork"},
            marks=RunIf(skip_windows=True),
        ),
        pytest.param(
            "ddp_notebook_find_unused_parameters_true",
            DDPStrategy,
            {"find_unused_parameters": True, "start_method": "fork"},
            marks=RunIf(skip_windows=True),
        ),
    ],
)
def test_ddp_find_unused_parameters_strategy_registry(
    tmpdir, strategy_name, strategy, expected_init_params, mps_count_0
):
    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy_name)
    assert isinstance(trainer.strategy, strategy)
    assert strategy_name in StrategyRegistry
    assert StrategyRegistry[strategy_name]["init_params"] == expected_init_params
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
