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
import math
from unittest.mock import Mock, patch

import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import GradientAccumulationScheduler, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.dataloaders import ResumableIterable


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2, 3])
def test_trainer_accumulate_grad_batches_zero_grad(tmp_path, accumulate_grad_batches):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmp_path,
            limit_train_batches=20,
            limit_val_batches=1,
            max_epochs=1,
            enable_model_summary=False,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        assert trainer.accumulate_grad_batches == accumulate_grad_batches
        trainer.fit(model)
        assert sgd_zero_grad.call_count == math.ceil(trainer.limit_train_batches / accumulate_grad_batches)


@pytest.mark.parametrize(
    ("accumulate_grad_batches", "expected_call_count"),
    [
        ({1: 2, 3: 4}, 10 + 5 + 5 + 3),
        ({0: 2, 2: 1}, 5 + 5 + 10 + 10),
    ],
)
def test_trainer_accumulate_grad_batches_with_callback(tmp_path, accumulate_grad_batches, expected_call_count):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmp_path,
            limit_train_batches=10,
            limit_val_batches=1,
            max_epochs=4,
            enable_model_summary=False,
            callbacks=GradientAccumulationScheduler(accumulate_grad_batches),
        )
        assert trainer.accumulate_grad_batches == 1  # initial value of Trainer argument
        trainer.fit(model)

        assert sum(isinstance(cb, GradientAccumulationScheduler) for cb in trainer.callbacks) == 1
        assert sgd_zero_grad.call_count == expected_call_count


@pytest.mark.parametrize(
    "scheduling",
    [
        {1: 2, -3: 4},
        {0: 2, "2": 1},
    ],
)
def test_invalid_keys_for_grad_accum_scheduler(scheduling):
    with pytest.raises(MisconfigurationException, match="Epoch should be an int"):
        _ = GradientAccumulationScheduler(scheduling=scheduling)


@pytest.mark.parametrize(
    "scheduling",
    [
        {1: 0, 3: 4},
        {0: 2, 2: "2"},
    ],
)
def test_invalid_values_for_grad_accum_scheduler(scheduling):
    with pytest.raises(MisconfigurationException, match="Accumulation factor should be an int"):
        _ = GradientAccumulationScheduler(scheduling=scheduling)


@pytest.mark.parametrize("strategy_class", [DeepSpeedStrategy])
def test_unsupported_strategies(strategy_class):
    """Test that an error is raised for strategies that require the gradient accumulation factor to be fixed."""
    scheduler = GradientAccumulationScheduler({1: 2})
    model = BoringModel()
    trainer = Trainer()
    trainer._accelerator_connector.strategy = Mock(spec=strategy_class)
    with pytest.raises(RuntimeError, match="does not support `accumulate_grad_batches` changing between epochs"):
        scheduler.on_train_start(trainer, model)


def test_unsupported_manual_optimization():
    """Test that an error is raised when attempting to use the callback with manual optimization."""
    scheduler = GradientAccumulationScheduler({1: 2})
    model = BoringModel()
    model.automatic_optimization = False
    trainer = Trainer()
    with pytest.raises(RuntimeError, match="Automatic gradient accumulation and the `GradientAccumulationScheduler`"):
        scheduler.on_train_start(trainer, model)


def test_warn_if_model_has_overridden_optimization_hooks():
    """Test that the callback warns if optimization hooks were overridden in the LightningModule."""

    class OverriddenOptimizerStepModel(BoringModel):
        def optimizer_step(self, *args, **kwargs):
            super().optimizer_step(*args, **kwargs)

    class OverriddenZeroGradModel(BoringModel):
        def optimizer_zero_grad(self, *args, **kwargs):
            super().optimizer_zero_grad(*args, **kwargs)

    scheduler = GradientAccumulationScheduler({1: 2})
    trainer = Trainer()

    model = OverriddenOptimizerStepModel()
    with pytest.warns(UserWarning, match="the hooks will not be called on every batch"):
        scheduler.on_train_start(trainer, model)

    model = OverriddenZeroGradModel()
    with pytest.warns(UserWarning, match="the hooks will not be called on every batch"):
        scheduler.on_train_start(trainer, model)


def test_raises_when_accumulate_grad_batches_with_callback(tmp_path):
    """Test that it is not allowed to set both the Trainer argument and also pass a callback."""
    trainer = Trainer(
        default_root_dir=tmp_path, accumulate_grad_batches=2, callbacks=[GradientAccumulationScheduler({0: 2})]
    )
    with pytest.raises(ValueError, match="`accumulate_grad_batches` and are using the `GradientAccumulationScheduler`"):
        trainer.fit(BoringModel())


def test_mid_epoch_resume_keeps_scheduled_accumulation(tmp_path):
    """Test that resuming from a mid-epoch checkpoint accumulates with the factor scheduled for that epoch."""

    class AccumulationModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.accumulation = {}

        def training_step(self, batch, batch_idx):
            self.accumulation[(self.current_epoch, batch_idx)] = self.trainer.accumulate_grad_batches
            return super().training_step(batch, batch_idx)

        def train_dataloader(self):
            return ResumableIterable()

    def fit(ckpt_path=None, callbacks=()):
        model = AccumulationModel()
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cpu",
            max_epochs=3,
            limit_val_batches=0,
            callbacks=[GradientAccumulationScheduler({1: 2}), *callbacks],
            enable_checkpointing=bool(callbacks),
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, ckpt_path=ckpt_path)
        return trainer, model

    # 4 steps in epoch 0, then 2 per epoch. Step 5 is saved after 2 of the 4 batches of epoch 1
    checkpoint = ModelCheckpoint(dirpath=tmp_path, filename="{step}", every_n_train_steps=5, save_top_k=-1)
    trainer, model = fit(callbacks=[checkpoint])
    assert trainer.global_step == 8

    resumed_trainer, resumed_model = fit(ckpt_path=tmp_path / "step=5.ckpt")
    expected = {key: value for key, value in model.accumulation.items() if key >= (1, 2)}
    assert resumed_model.accumulation == expected
    assert resumed_trainer.global_step == 8
    torch.testing.assert_close(resumed_model.state_dict(), model.state_dict())
