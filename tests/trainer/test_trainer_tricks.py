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
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


def test_num_training_batches(tmpdir):
    """
    Tests that the correct number of batches are allocated
    """
    # when we have fewer batches in the dataloader we should use those instead of the limit
    model = EvalModelTemplate()
    trainer = Trainer(
        limit_val_batches=100,
        limit_train_batches=100,
        max_epochs=1,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)

    assert len(model.train_dataloader()) == 10
    assert len(model.val_dataloader()) == 10
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 10
    assert trainer.num_training_batches == 10

    # when we have more batches in the dataloader we should limit them
    model = EvalModelTemplate()
    trainer = Trainer(
        limit_val_batches=7,
        limit_train_batches=7,
        max_epochs=1,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)

    assert len(model.train_dataloader()) == 10
    assert len(model.val_dataloader()) == 10
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 7
    assert trainer.num_training_batches == 7


def test_overfit_batch_limits(tmpdir):
    # ------------------------------------------------------
    # Make sure shuffle is correct across loaders initially
    # ------------------------------------------------------
    model = EvalModelTemplate()
    model.train_dataloader()

    # original train loader which should be replaced in all methods
    train_loader = model.train_dataloader()

    # make sure the val and tests are not shuffled
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(model.val_dataloader().sampler, SequentialSampler)
    assert isinstance(model.test_dataloader().sampler, SequentialSampler)

    # ------------------------------------------------------
    # get the training loader and batch
    # ------------------------------------------------------
    # Create a reference train dataloader without shuffling.
    train_loader = DataLoader(model.train_dataloader().dataset, shuffle=False)
    (xa, ya) = next(iter(train_loader))
    train_loader = DataLoader(model.train_dataloader().dataset, shuffle=True)
    full_train_samples = len(train_loader)
    num_train_samples = int(0.11 * full_train_samples)

    # ------------------------------------------------------
    # set VAL and Test loaders
    # ------------------------------------------------------
    val_loader = DataLoader(model.val_dataloader().dataset, shuffle=False)
    test_loader = DataLoader(model.test_dataloader().dataset, shuffle=False)

    # set the model loaders
    model.train_dataloader = lambda: train_loader
    model.val_dataloader = lambda: val_loader
    model.test_dataloader = lambda: test_loader

    # ------------------------------------------------------
    # test train loader applies correct limits
    # ------------------------------------------------------
    trainer = Trainer(overfit_batches=4)
    trainer.reset_train_dataloader(model)
    assert trainer.num_training_batches == 4

    # make sure the loaders are the same
    (xb, yb) = next(iter(trainer.train_dataloader))
    assert torch.eq(xa, xb).all()
    assert torch.eq(ya, yb).all()

    trainer = Trainer(overfit_batches=0.11)
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
    for split in ['val', 'test']:

        # ------------------------------------------------------
        # test overfit_batches as percent
        # ------------------------------------------------------
        loader_num_batches, dataloaders = Trainer(overfit_batches=0.11)._reset_eval_dataloader(model, split)
        assert loader_num_batches[0] == num_train_samples

        # make sure we turned off shuffle for the user
        assert isinstance(dataloaders[0].sampler, SequentialSampler)

        # make sure the loaders are the same
        (xb, yb) = next(iter(dataloaders[0]))
        assert torch.eq(xa, xb).all()
        assert torch.eq(ya, yb).all()

        # ------------------------------------------------------
        # test overfit_batches as int
        # ------------------------------------------------------
        loader_num_batches, dataloaders = Trainer(overfit_batches=1)._reset_eval_dataloader(model, split)
        assert loader_num_batches[0] == 1
        loader_num_batches, dataloaders = Trainer(overfit_batches=5)._reset_eval_dataloader(model, split)
        assert loader_num_batches[0] == 5

        # ------------------------------------------------------
        # test limit_xxx_batches as percent AND int
        # ------------------------------------------------------
        if split == 'val':
            loader_num_batches, dataloaders = Trainer(limit_val_batches=0.1)._reset_eval_dataloader(model, split)
            assert loader_num_batches[0] == int(0.1 * len(val_loader))

            loader_num_batches, dataloaders = Trainer(limit_val_batches=10)._reset_eval_dataloader(model, split)
            assert loader_num_batches[0] == 10
        else:
            loader_num_batches, dataloaders = Trainer(limit_test_batches=0.1)._reset_eval_dataloader(model, split)
            assert loader_num_batches[0] == int(0.1 * len(test_loader))

            loader_num_batches, dataloaders = Trainer(limit_test_batches=10)._reset_eval_dataloader(model, split)
            assert loader_num_batches[0] == 10
