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
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from legacy.simple_classif_training import ClassifDataModule, ClassificationModel
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage
from tests.helpers.boring_model import BoringModel, RandomDataset


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

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, overfit_batches=2)

    with pytest.warns(UserWarning, match="requested to overfit but enabled training dataloader shuffling"):
        trainer.fit(model)

    assert isinstance(trainer.train_dataloader.loaders.sampler, SequentialSampler)


def test_overfit_batch_limits(tmpdir):
    # ------------------------------------------------------
    # Make sure shuffle is correct across loaders initially
    # ------------------------------------------------------
    model = ClassificationModel()
    dm = ClassifDataModule()

    # original train loader which should be replaced in all methods
    train_loader = dm.train_dataloader()

    # make sure the val and tests are not shuffled
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
    assert isinstance(dm.test_dataloader().sampler, SequentialSampler)

    # ------------------------------------------------------
    # get the training loader and batch
    # ------------------------------------------------------
    # Create a reference train dataloader without shuffling.
    train_loader = DataLoader(dm.train_dataloader().dataset, shuffle=False)
    (xa, ya) = next(iter(train_loader))
    train_loader = DataLoader(dm.train_dataloader().dataset, shuffle=True)
    full_train_samples = len(train_loader)
    num_train_samples = int(0.11 * full_train_samples)

    # ------------------------------------------------------
    # set VAL and Test loaders
    # ------------------------------------------------------
    val_loader = DataLoader(dm.val_dataloader().dataset, shuffle=False)
    test_loader = DataLoader(dm.test_dataloader().dataset, shuffle=False)

    # set the model loaders
    model.train_dataloader = lambda: train_loader
    model.val_dataloader = lambda: val_loader
    model.test_dataloader = lambda: test_loader

    # ------------------------------------------------------
    # test train loader applies correct limits
    # ------------------------------------------------------
    trainer = Trainer(overfit_batches=4)
    model.trainer = trainer
    trainer._data_connector.attach_dataloaders(model=model)
    trainer.reset_train_dataloader(model)
    assert trainer.num_training_batches == 4

    # make sure the loaders are the same
    (xb, yb) = next(iter(trainer.train_dataloader))
    assert torch.eq(xa, xb).all()
    assert torch.eq(ya, yb).all()

    trainer = Trainer(overfit_batches=0.11)
    model.trainer = trainer
    trainer._data_connector.attach_dataloaders(model=model)
    trainer.reset_train_dataloader(model)
    # The dataloader should have been overwritten with a Sequential sampler.
    assert trainer.train_dataloader is not train_loader
    assert trainer.num_training_batches == num_train_samples

    # make sure the loaders are the same
    (xb, yb) = next(iter(trainer.train_dataloader))
    assert torch.eq(xa, xb).all()
    assert torch.eq(ya, yb).all()

    # ------------------------------------------------------
    # run tests for both val and test
    # ------------------------------------------------------
    for split in (RunningStage.VALIDATING, RunningStage.TESTING):

        # ------------------------------------------------------
        # test overfit_batches as percent
        # ------------------------------------------------------
        trainer = Trainer(overfit_batches=0.11)
        trainer._data_connector.attach_dataloaders(model)
        loader_num_batches, _ = trainer._reset_eval_dataloader(split, model=model)
        if split == RunningStage.VALIDATING:
            assert loader_num_batches[0] == 0
        else:
            assert loader_num_batches[0] == len(test_loader)

        # ------------------------------------------------------
        # test overfit_batches as int
        # ------------------------------------------------------
        trainer = Trainer(overfit_batches=1)
        trainer._data_connector.attach_dataloaders(model)
        loader_num_batches, dataloaders = trainer._reset_eval_dataloader(split, model=model)
        if split == RunningStage.VALIDATING:
            assert loader_num_batches[0] == 0
        else:
            assert loader_num_batches[0] == len(test_loader)
            # make sure we turned off shuffle for the user
            assert isinstance(dataloaders[0].sampler, SequentialSampler)

        trainer = Trainer(overfit_batches=5)
        trainer._data_connector.attach_dataloaders(model)
        loader_num_batches, _ = trainer._reset_eval_dataloader(split, model=model)
        if split == RunningStage.VALIDATING:
            assert loader_num_batches[0] == 0
        else:
            assert loader_num_batches[0] == len(test_loader)
