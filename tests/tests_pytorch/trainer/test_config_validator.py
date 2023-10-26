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
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.trainer.configuration_validator import (
    __verify_eval_loop_configuration,
    __verify_train_val_loop_configuration,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_wrong_train_setting(tmpdir):
    """Test that an error is raised when no `training_step()` is defined."""
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    with pytest.raises(MisconfigurationException, match=r"No `training_step\(\)` method defined."):
        model = BoringModel()
        model.training_step = None
        trainer.fit(model)


def test_wrong_configure_optimizers(tmpdir):
    """Test that an error is thrown when no `configure_optimizers()` is defined."""
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    with pytest.raises(MisconfigurationException, match=r"No `configure_optimizers\(\)` method defined."):
        model = BoringModel()
        model.configure_optimizers = None
        trainer.fit(model)


def test_fit_val_loop_config(tmpdir):
    """When either val loop or val data are missing raise warning."""
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    # no val data has val loop
    with pytest.warns(UserWarning, match=r"You passed in a `val_dataloader` but have no `validation_step`"):
        model = BoringModel()
        model.validation_step = None
        trainer.fit(model)

    # has val loop but no val data
    with pytest.warns(PossibleUserWarning, match=r"You defined a `validation_step` but have no `val_dataloader`"):
        model = BoringModel()
        model.val_dataloader = None
        trainer.fit(model)


def test_eval_loop_config(tmpdir):
    """When either eval step or eval data is missing."""
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    # has test data but no val step
    model = BoringModel()
    model.validation_step = None
    with pytest.raises(MisconfigurationException, match=r"No `validation_step\(\)` method defined"):
        trainer.validate(model)

    # has test data but no test step
    model = BoringModel()
    model.test_step = None
    with pytest.raises(MisconfigurationException, match=r"No `test_step\(\)` method defined"):
        trainer.test(model)

    # has predict data but no predict_step
    model = BoringModel()
    model.predict_step = None
    with pytest.raises(MisconfigurationException, match=r"`predict_step` cannot be None."):
        trainer.predict(model)

    # has predict data but no forward
    model = BoringModel()
    model.forward = None
    with pytest.raises(MisconfigurationException, match=r"requires `forward` method to run."):
        trainer.predict(model)


@pytest.mark.parametrize("datamodule", [False, True])
def test_trainer_predict_verify_config(tmpdir, datamodule):
    class TestModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            return self.layer(x)

    class TestLightningDataModule(LightningDataModule):
        def __init__(self, dataloaders):
            super().__init__()
            self._dataloaders = dataloaders

        def test_dataloader(self):
            return self._dataloaders

        def predict_dataloader(self):
            return self._dataloaders

    data = [torch.utils.data.DataLoader(RandomDataset(32, 2)), torch.utils.data.DataLoader(RandomDataset(32, 2))]
    if datamodule:
        data = TestLightningDataModule(data)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir)
    results = trainer.predict(model, data)

    assert len(results) == 2
    assert results[0][0].shape == torch.Size([1, 2])


def test_trainer_manual_optimization_config():
    """Test error message when requesting Trainer features unsupported with manual optimization."""
    model = BoringModel()
    model.automatic_optimization = False

    trainer = Trainer(gradient_clip_val=1.0)
    with pytest.raises(MisconfigurationException, match="Automatic gradient clipping is not supported"):
        trainer.fit(model)

    trainer = Trainer(accumulate_grad_batches=2)
    with pytest.raises(MisconfigurationException, match="Automatic gradient accumulation is not supported"):
        trainer.fit(model)


def test_legacy_epoch_end_hooks():
    class TrainingEpochEndModel(BoringModel):
        def training_epoch_end(self, outputs):
            pass

    class ValidationEpochEndModel(BoringModel):
        def validation_epoch_end(self, outputs):
            pass

    trainer = Mock()
    with pytest.raises(NotImplementedError, match="training_epoch_end` has been removed in v2.0"):
        __verify_train_val_loop_configuration(trainer, TrainingEpochEndModel())
    with pytest.raises(NotImplementedError, match="validation_epoch_end` has been removed in v2.0"):
        __verify_train_val_loop_configuration(trainer, ValidationEpochEndModel())

    class TestEpochEndModel(BoringModel):
        def test_epoch_end(self, outputs):
            pass

    with pytest.raises(NotImplementedError, match="validation_epoch_end` has been removed in v2.0"):
        __verify_eval_loop_configuration(ValidationEpochEndModel(), "val")
    with pytest.raises(NotImplementedError, match="test_epoch_end` has been removed in v2.0"):
        __verify_eval_loop_configuration(TestEpochEndModel(), "test")
