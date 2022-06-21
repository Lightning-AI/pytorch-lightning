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
import torch
from legacy.simple_classif_training import ClassifDataModule, ClassificationModel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset
from pytorch_lightning.trainer.states import RunningStage
from tests_pytorch.helpers.runif import RunIf


@pytest.mark.parametrize("overfit_batches", [1, 2, 0.1, 0.25, 1.0])
def test_overfit_basic(tmpdir, overfit_batches):
    """Tests that only training_step can be used when overfitting."""

    model = BoringModel()
    model.validation_step = None
    total_train_samples = len(BoringModel().train_dataloader())

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, overfit_batches=overfit_batches, enable_model_summary=False
    )
    trainer.fit(model)

    assert trainer.num_val_batches == []
    assert trainer.num_training_batches == int(
        overfit_batches * (1 if isinstance(overfit_batches, int) else total_train_samples)
    )


def test_overfit_batches_raises_warning_in_case_of_sequential_sampler(tmpdir):
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
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, overfit_batches=2)

    with pytest.warns(UserWarning, match="requested to overfit but enabled training dataloader shuffling"):
        trainer.fit(model)

    assert isinstance(trainer.train_dataloader.loaders.sampler, SequentialSampler)
    assert isinstance(trainer.val_dataloaders[0].sampler, SequentialSampler)


@pytest.mark.parametrize(
    "stage,mode",
    [(RunningStage.VALIDATING, "val"), (RunningStage.TESTING, "test"), (RunningStage.PREDICTING, "predict")],
)
@pytest.mark.parametrize("overfit_batches", [0.11, 4])
def test_overfit_batch_limits_eval(stage, mode, overfit_batches):
    model = ClassificationModel()
    dm = ClassifDataModule()
    eval_loader = getattr(dm, f"{mode}_dataloader")()
    trainer = Trainer(overfit_batches=overfit_batches)
    model.trainer = trainer
    trainer._data_connector.attach_datamodule(model, datamodule=dm)

    loader_num_batches, dataloaders = trainer._data_connector._reset_eval_dataloader(stage, model=model)
    if stage == RunningStage.VALIDATING:
        assert (
            loader_num_batches[0] == overfit_batches
            if isinstance(overfit_batches, int)
            else len(dm.val_dataloader()) * overfit_batches
        )
    else:
        assert loader_num_batches[0] == len(eval_loader)
        assert isinstance(dataloaders[0].sampler, SequentialSampler)


@pytest.mark.parametrize("overfit_batches", [0.11, 4])
def test_overfit_batch_limits_train(overfit_batches):
    model = ClassificationModel()
    dm = ClassifDataModule()

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
    trainer._data_connector.attach_dataloaders(model=model)
    trainer.reset_train_dataloader(model)
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
    trainer.model = model
    trainer._data_connector.attach_dataloaders(model)
    trainer.reset_train_dataloader()
    train_sampler = trainer.train_dataloader.loaders.sampler
    assert isinstance(train_sampler, DistributedSampler)
    assert train_sampler.shuffle is False
