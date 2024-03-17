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
import os
from unittest import mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.trainer.states import RunningStage
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.datasets import SklearnDataset
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@pytest.mark.parametrize("overfit_batches", [1, 2, 0.1, 0.25, 1.0])
def test_overfit_basic(tmp_path, overfit_batches):
    """Tests that only training_step can be used when overfitting."""
    model = BoringModel()
    model.validation_step = None
    total_train_samples = len(BoringModel().train_dataloader())

    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, overfit_batches=overfit_batches, enable_model_summary=False
    )
    trainer.fit(model)

    assert trainer.num_val_batches == []
    assert trainer.num_training_batches == int(
        overfit_batches * (1 if isinstance(overfit_batches, int) else total_train_samples)
    )


def test_overfit_batches_raises_warning_in_case_of_sequential_sampler(tmp_path):
    class NonSequentialSampler(Sampler):
        def __init__(self, data_source):
            self.data_source = data_source

        def __iter__(self):
            return iter(range(len(self.data_source)))

        def __len__(self):
            return len(self.data_source)

    class TestModel(BoringModel):
        def train_dataloader(self):
            dataset = RandomDataset(32, 64)
            sampler = NonSequentialSampler(dataset)
            return torch.utils.data.DataLoader(dataset, sampler=sampler)

        def val_dataloader(self):
            dataset = RandomDataset(32, 64)
            sampler = NonSequentialSampler(dataset)
            return torch.utils.data.DataLoader(dataset, sampler=sampler)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, overfit_batches=2)

    with pytest.warns(UserWarning, match="requested to overfit but enabled train dataloader shuffling"):
        trainer.fit(model)

    assert isinstance(trainer.train_dataloader.sampler, SequentialSampler)
    assert isinstance(trainer.val_dataloaders.sampler, SequentialSampler)


@pytest.mark.parametrize(
    ("stage", "mode"),
    [(RunningStage.VALIDATING, "val"), (RunningStage.TESTING, "test"), (RunningStage.PREDICTING, "predict")],
)
@pytest.mark.parametrize("overfit_batches", [0.11, 4])
@RunIf(sklearn=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_overfit_batch_limits_eval(stage, mode, overfit_batches):
    model = ClassificationModel()
    dm = ClassifDataModule()
    eval_loader = getattr(dm, f"{mode}_dataloader")()
    trainer = Trainer(overfit_batches=overfit_batches)
    model.trainer = trainer
    trainer.strategy.connect(model)
    trainer._data_connector.attach_datamodule(model, datamodule=dm)

    if stage == RunningStage.VALIDATING:
        trainer.fit_loop.epoch_loop.val_loop.setup_data()
        assert (
            trainer.num_val_batches[0] == overfit_batches
            if isinstance(overfit_batches, int)
            else len(dm.val_dataloader()) * overfit_batches
        )
    elif stage == RunningStage.TESTING:
        trainer.test_loop.setup_data()
        assert trainer.num_test_batches[0] == len(eval_loader)
        assert isinstance(trainer.test_dataloaders.sampler, SequentialSampler)
    elif stage == RunningStage.PREDICTING:
        trainer.predict_loop.setup_data()
        assert trainer.num_predict_batches[0] == len(eval_loader)
        assert isinstance(trainer.predict_dataloaders.sampler, SequentialSampler)


@pytest.mark.parametrize("overfit_batches", [0.11, 4])
@RunIf(sklearn=True)
def test_overfit_batch_limits_train(overfit_batches):
    class CustomDataModule(ClassifDataModule):
        def train_dataloader(self):
            return DataLoader(
                SklearnDataset(self.x_train, self.y_train, self._x_type, self._y_type),
                batch_size=self.batch_size,
                shuffle=True,
            )

    model = ClassificationModel()
    dm = CustomDataModule()

    # original train loader which should be replaced in all methods
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader.sampler, RandomSampler)

    # Create a reference train dataloader without shuffling.
    train_loader = DataLoader(dm.train_dataloader().dataset, shuffle=False)
    (xa, ya) = next(iter(train_loader))
    train_loader = DataLoader(dm.train_dataloader().dataset, shuffle=True)
    full_train_samples = len(train_loader)

    # set the model loaders
    model.train_dataloader = lambda: train_loader

    # test train loader applies correct limits
    trainer = Trainer(overfit_batches=overfit_batches)
    model.trainer = trainer
    trainer.strategy.connect(model)
    trainer._data_connector.attach_dataloaders(model=model)
    trainer.fit_loop.setup_data()
    expected_batches = (
        int(overfit_batches * full_train_samples) if isinstance(overfit_batches, float) else overfit_batches
    )
    assert trainer.num_training_batches == expected_batches

    # make sure the loaders are the same
    (xb, yb) = next(iter(trainer.train_dataloader))
    assert torch.eq(xa, xb).all()
    assert torch.eq(ya, yb).all()


@RunIf(skip_windows=True)
def test_distributed_sampler_with_overfit_batches():
    model = BoringModel()
    trainer = Trainer(
        overfit_batches=1,
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
    )
    model.trainer = trainer
    trainer.strategy.connect(model)
    trainer._data_connector.attach_dataloaders(model)
    trainer.fit_loop.setup_data()
    train_sampler = trainer.train_dataloader.sampler
    assert isinstance(train_sampler, DistributedSampler)
    assert train_sampler.shuffle is False
