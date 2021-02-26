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
import os
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _NATIVE_AMP_AVAILABLE, AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.helpers.datamodules import MNISTDataModule


def test_num_training_batches(tmpdir):
    """
    Tests that the correct number of batches are allocated
    """
    # when we have fewer batches in the dataloader we should use those instead of the limit
    model = EvalModelTemplate()
    trainer = Trainer(limit_val_batches=100, limit_train_batches=100, max_epochs=1)
    trainer.fit(model)

    assert len(model.train_dataloader()) == 10
    assert len(model.val_dataloader()) == 10
    assert isinstance(trainer.num_val_batches, list)
    assert trainer.num_val_batches[0] == 10
    assert trainer.num_training_batches == 10

    # when we have more batches in the dataloader we should limit them
    model = EvalModelTemplate()
    trainer = Trainer(limit_val_batches=7, limit_train_batches=7, max_epochs=1)
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


def test_model_reset_correctly(tmpdir):
    """ Check that model weights are correctly reset after scaling batch size. """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    before_state_dict = deepcopy(model.state_dict())

    trainer.tuner.scale_batch_size(model, max_trials=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after scaling batch size'


def test_trainer_reset_correctly(tmpdir):
    """ Check that all trainer parameters are reset correctly after scaling batch size. """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    changed_attributes = [
        'max_steps',
        'weights_summary',
        'logger',
        'callbacks',
        'checkpoint_callback',
        'limit_train_batches',
        'current_epoch',
    ]

    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)

    trainer.tuner.scale_batch_size(model, max_trials=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize('scale_arg', ['power', 'binsearch', True])
def test_auto_scale_batch_size_trainer_arg(tmpdir, scale_arg):
    """ Test possible values for 'batch size auto scaling' Trainer argument. """
    tutils.reset_seed()
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    before_batch_size = hparams.get('batch_size')
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        auto_scale_batch_size=scale_arg,
        gpus=1,
    )
    trainer.tune(model)
    after_batch_size = model.batch_size
    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'

    assert not os.path.exists(tmpdir / 'scale_batch_size_temp_model.ckpt')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize('use_hparams', [True, False])
def test_auto_scale_batch_size_set_model_attribute(tmpdir, use_hparams):
    """ Test that new batch size gets written to the correct hyperparameter attribute. """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()
    before_batch_size = hparams.get('batch_size')

    class HparamsEvalModelTemplate(EvalModelTemplate):

        def dataloader(self, *args, **kwargs):
            # artificially set batch_size so we can get a dataloader
            # remove it immediately after, because we want only self.hparams.batch_size
            setattr(self, "batch_size", before_batch_size)
            dataloader = super().dataloader(*args, **kwargs)
            del self.batch_size
            return dataloader

    datamodule_model = MNISTDataModule(data_dir=tmpdir, batch_size=111)  # this datamodule should get ignored!
    datamodule_fit = MNISTDataModule(data_dir=tmpdir, batch_size=before_batch_size)

    model_class = HparamsEvalModelTemplate if use_hparams else EvalModelTemplate
    model = model_class(**hparams)
    model.datamodule = datamodule_model  # unused when another module gets passed to .tune() / .fit()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        auto_scale_batch_size=True,
        gpus=1,
    )
    trainer.tune(model, datamodule_fit)
    after_batch_size = model.hparams.batch_size if use_hparams else model.batch_size
    assert trainer.datamodule == datamodule_fit
    assert before_batch_size != after_batch_size
    assert after_batch_size <= len(trainer.train_dataloader.dataset)
    assert datamodule_fit.batch_size == after_batch_size
    # should be left unchanged, since it was not passed to .tune()
    assert datamodule_model.batch_size == 111


def test_auto_scale_batch_size_duplicate_attribute_warning(tmpdir):
    """ Test for a warning when model.batch_size and model.hparams.batch_size both present. """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.hparams = hparams
    # now we have model.batch_size and model.hparams.batch_size
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, max_epochs=1000, auto_scale_batch_size=True)
    expected_message = "Field `model.batch_size` and `model.hparams.batch_size` are mutually exclusive!"
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)


@pytest.mark.parametrize('scale_method', ['power', 'binsearch'])
def test_call_to_trainer_method(tmpdir, scale_method):
    """ Test that calling the trainer method itself works. """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    before_batch_size = hparams.get('batch_size')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    after_batch_size = trainer.tuner.scale_batch_size(model, mode=scale_method, max_trials=5)
    model.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


def test_error_on_dataloader_passed_to_fit(tmpdir):
    """Verify that when the auto scale batch size feature raises an error
       if a train dataloader is passed to fit """

    # only train passed to fit
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        auto_scale_batch_size='power',
    )
    fit_options = dict(train_dataloader=model.dataloader(train=True))

    with pytest.raises(MisconfigurationException):
        trainer.tune(model, **fit_options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(not _NATIVE_AMP_AVAILABLE, reason="test requires native AMP.")
def test_auto_scale_batch_size_with_amp(tmpdir):
    model = EvalModelTemplate()
    batch_size_before = model.batch_size
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        auto_scale_batch_size=True,
        gpus=1,
        precision=16,
    )
    trainer.tune(model)
    batch_size_after = model.batch_size
    assert trainer.amp_backend == AMPType.NATIVE
    assert trainer.scaler is not None
    assert batch_size_after != batch_size_before
