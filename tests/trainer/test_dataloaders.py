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
import platform
from distutils.version import LooseVersion
from unittest import mock
from unittest.mock import patch

import pytest
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

import tests.base.develop_pipelines as tpipes
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.base.boring_model import BoringModel, RandomDataset


def test_fit_train_loader_only(tmpdir):

    model = EvalModelTemplate()
    train_dataloader = model.train_dataloader()

    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    model.validation_step = None
    model.validation_epoch_end = None

    model.test_step = None
    model.test_epoch_end = None

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloader=train_dataloader)


def test_fit_val_loader_only(tmpdir):

    model = EvalModelTemplate()
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()

    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    model.test_step = None
    model.test_epoch_end = None

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


@pytest.mark.parametrize("dataloader_options", [
    dict(val_check_interval=10000),
])
def test_dataloader_config_errors_runtime(tmpdir, dataloader_options):
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        **dataloader_options,
    )
    with pytest.raises(ValueError):
        # fit model
        trainer.fit(model)


@pytest.mark.parametrize("dataloader_options", [
    dict(limit_train_batches=-0.1),
    dict(limit_train_batches=1.2),
    dict(limit_val_batches=-0.1),
    dict(limit_val_batches=1.2),
    dict(limit_test_batches=-0.1),
    dict(limit_test_batches=1.2),
    dict(val_check_interval=-0.1),
    dict(val_check_interval=1.2),
    dict(overfit_batches=-0.1),
    dict(overfit_batches=1.2),
])
def test_dataloader_config_errors_init(tmpdir, dataloader_options):
    with pytest.raises(MisconfigurationException, match='passed invalid value'):
        Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            **dataloader_options,
        )


def test_multiple_val_dataloader(tmpdir):
    """Verify multiple val_dataloader."""

    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__multiple
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=1.0,
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    # verify there are 2 val loaders
    assert len(trainer.val_dataloaders) == 2, \
        'Multiple val_dataloaders not initiated properly'

    # make sure predictions are good for each val set
    for dataloader in trainer.val_dataloaders:
        tpipes.run_prediction(dataloader, trainer.model)


@pytest.mark.parametrize('ckpt_path', [None, 'best', 'specific'])
def test_multiple_test_dataloader(tmpdir, ckpt_path):
    """Verify multiple test_dataloader."""

    model_template = EvalModelTemplate()

    class MultipleTestDataloaderModel(EvalModelTemplate):
        def test_dataloader(self):
            return model_template.test_dataloader__multiple()

        def test_step(self, batch, batch_idx, *args, **kwargs):
            return model_template.test_step__multiple_dataloaders(batch, batch_idx, *args, **kwargs)

    model = MultipleTestDataloaderModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    trainer.fit(model)
    if ckpt_path == 'specific':
        ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.test(ckpt_path=ckpt_path)

    # verify there are 2 test loaders
    assert len(trainer.test_dataloaders) == 2, \
        'Multiple test_dataloaders not initiated properly'

    # make sure predictions are good for each test set
    for dataloader in trainer.test_dataloaders:
        tpipes.run_prediction(dataloader, trainer.model)

    # run the test method
    trainer.test(ckpt_path=ckpt_path)


def test_train_dataloader_passed_to_fit(tmpdir):
    """Verify that train dataloader can be passed to fit """

    # only train passed to fit
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    fit_options = dict(train_dataloader=model.dataloader(train=True))
    result = trainer.fit(model, **fit_options)

    assert result == 1


def test_train_val_dataloaders_passed_to_fit(tmpdir):
    """ Verify that train & val dataloader can be passed to fit """

    # train, val passed to fit
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    fit_options = dict(train_dataloader=model.dataloader(train=True),
                       val_dataloaders=model.dataloader(train=False))

    result = trainer.fit(model, **fit_options)
    assert result == 1
    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'


@pytest.mark.parametrize('ckpt_path', [None, 'best', 'specific'])
def test_all_dataloaders_passed_to_fit(tmpdir, ckpt_path):
    """Verify train, val & test dataloader(s) can be passed to fit and test method"""

    model = EvalModelTemplate()

    # train, val and test passed to fit
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    fit_options = dict(train_dataloader=model.dataloader(train=True),
                       val_dataloaders=model.dataloader(train=False))
    result = trainer.fit(model, **fit_options)

    if ckpt_path == 'specific':
        ckpt_path = trainer.checkpoint_callback.best_model_path
    test_options = dict(test_dataloaders=model.dataloader(train=False),
                        ckpt_path=ckpt_path)
    trainer.test(**test_options)

    assert result == 1
    assert len(trainer.val_dataloaders) == 1, \
        f'val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 1, \
        f'test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


@pytest.mark.parametrize('ckpt_path', [None, 'best', 'specific'])
def test_multiple_dataloaders_passed_to_fit(tmpdir, ckpt_path):
    """Verify that multiple val & test dataloaders can be passed to fit."""

    model = EvalModelTemplate()
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    model.test_step = model.test_step__multiple_dataloaders

    # train, multiple val and multiple test passed to fit
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    fit_options = dict(train_dataloader=model.dataloader(train=True),
                       val_dataloaders=[model.dataloader(train=False),
                                        model.dataloader(train=False)])
    trainer.fit(model, **fit_options)
    if ckpt_path == 'specific':
        ckpt_path = trainer.checkpoint_callback.best_model_path
    test_options = dict(test_dataloaders=[model.dataloader(train=False),
                                          model.dataloader(train=False)],
                        ckpt_path=ckpt_path)
    trainer.test(**test_options)

    assert len(trainer.val_dataloaders) == 2, \
        f'Multiple `val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 2, \
        f'Multiple `test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


@pytest.mark.parametrize(['limit_train_batches', 'limit_val_batches', 'limit_test_batches'], [
    pytest.param(0.0, 0.0, 0.0),
    pytest.param(1.0, 1.0, 1.0),
])
def test_inf_dataloaders_with_limit_percent_batches(tmpdir, limit_train_batches, limit_val_batches, limit_test_batches):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit in percent"""
    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__infinite
    model.val_dataloader = model.val_dataloader__infinite
    model.test_dataloader = model.test_dataloader__infinite

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    results = trainer.fit(model)
    assert results == 1
    assert trainer.num_training_batches == (0 if limit_train_batches == 0.0 else float('inf'))
    assert trainer.num_val_batches[0] == (0 if limit_val_batches == 0.0 else float('inf'))

    trainer.test(ckpt_path=None)
    assert trainer.num_test_batches[0] == (0 if limit_test_batches == 0.0 else float('inf'))


@pytest.mark.parametrize(['limit_train_batches', 'limit_val_batches', 'limit_test_batches'], [
    pytest.param(0, 0, 0),
    pytest.param(10, 10, 10),
])
def test_inf_dataloaders_with_limit_num_batches(tmpdir, limit_train_batches, limit_val_batches, limit_test_batches):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number"""
    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__infinite
    model.val_dataloader = model.val_dataloader__infinite
    model.test_dataloader = model.test_dataloader__infinite

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    results = trainer.fit(model)
    assert results
    assert trainer.num_training_batches == limit_train_batches
    assert trainer.num_val_batches[0] == limit_val_batches

    trainer.test(ckpt_path=None)
    assert trainer.num_test_batches[0] == limit_test_batches


@pytest.mark.parametrize(
    ['limit_train_batches', 'limit_val_batches', 'limit_test_batches'],
    [
        pytest.param(0.0, 0.0, 0.0),
        pytest.param(0, 0, 0.5),
        pytest.param(1.0, 1.0, 1.0),
        pytest.param(0.2, 0.4, 0.4),
    ]
)
def test_dataloaders_with_limit_percent_batches(tmpdir, limit_train_batches, limit_val_batches, limit_test_batches):
    """Verify num_batches for train, val & test dataloaders passed with batch limit in percent"""
    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__multiple_mixed_length
    model.test_dataloader = model.test_dataloader__multiple_mixed_length
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    model.test_step = model.test_step__multiple_dataloaders
    model.test_epoch_end = model.test_epoch_end__multiple_dataloaders

    # train, multiple val and multiple test passed with percent_check
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )
    trainer.fit(model)
    expected_train_batches = int(len(trainer.train_dataloader) * limit_train_batches)
    expected_val_batches = [
        int(len(dataloader) * limit_val_batches) for dataloader in trainer.val_dataloaders
    ]
    assert trainer.num_training_batches == expected_train_batches
    assert trainer.num_val_batches == expected_val_batches

    trainer.test(ckpt_path=None)
    expected_test_batches = [
        int(len(dataloader) * limit_test_batches) for dataloader in trainer.test_dataloaders
    ]
    assert trainer.num_test_batches == expected_test_batches


@pytest.mark.parametrize(
    ['limit_train_batches', 'limit_val_batches', 'limit_test_batches'],
    [
        pytest.param(0, 0, 0),
        pytest.param(1, 2, 3),
        pytest.param(1, 2, 1e50),
    ]
)
@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_with_limit_num_batches(tmpdir, limit_train_batches, limit_val_batches, limit_test_batches):
    """Verify num_batches for train, val & test dataloaders passed with batch limit as number"""

    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__multiple_mixed_length
    model.test_dataloader = model.test_dataloader__multiple_mixed_length
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    model.test_step = model.test_step__multiple_dataloaders
    model.test_epoch_end = model.test_epoch_end__multiple_dataloaders

    # train, multiple val and multiple test passed with percent_check
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )
    trainer.fit(model)

    # -------------------------------------------
    # MAKE SURE THE TRAINER SET THE CORRECT VALUES
    # -------------------------------------------
    assert trainer.num_training_batches == limit_train_batches
    assert trainer.num_val_batches == [limit_val_batches] * len(trainer.val_dataloaders)
    trainer.test(ckpt_path=None)

    # when the limit is greater than the number of test batches it should be the num in loaders
    test_dataloader_lengths = [len(x) for x in model.test_dataloader()]
    if limit_test_batches > 1e10:
        assert trainer.num_test_batches == test_dataloader_lengths
    else:
        assert trainer.num_test_batches == [limit_test_batches] * len(trainer.test_dataloaders)

    # -------------------------------------------
    # make sure we actually saw the expected num of batches
    # -------------------------------------------
    num_val_dataloaders = len(model.val_dataloader())
    num_test_dataloaders = len(model.test_dataloader())
    if limit_train_batches > 0:

        # make sure val batches are as expected
        assert len(trainer.dev_debugger.num_seen_val_check_batches) == num_val_dataloaders
        for dataloader_idx, num_batches in trainer.dev_debugger.num_seen_val_check_batches.items():
            assert num_batches == limit_val_batches

        # make sure test batches are as expected
        assert len(trainer.dev_debugger.num_seen_test_check_batches) == num_test_dataloaders
        for dataloader_idx, num_batches in trainer.dev_debugger.num_seen_test_check_batches.items():
            if limit_test_batches > 1e10:
                assert num_batches == test_dataloader_lengths[dataloader_idx]
            else:
                assert num_batches == limit_test_batches


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.parametrize('fast_dev_run', [True, 1, 3, -1, 'temp'])
def test_dataloaders_with_fast_dev_run(tmpdir, fast_dev_run):
    """
    Verify num_batches for train, val & test dataloaders passed with fast_dev_run
    """
    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__multiple_mixed_length
    model.test_dataloader = model.test_dataloader__multiple_mixed_length
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    model.test_step = model.test_step__multiple_dataloaders
    model.test_epoch_end = model.test_epoch_end__multiple_dataloaders

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=2,
        fast_dev_run=fast_dev_run,
    )

    if fast_dev_run == 'temp':
        with pytest.raises(MisconfigurationException, match='either a bool or an int'):
            trainer = Trainer(**trainer_options)
    elif fast_dev_run == -1:
        with pytest.raises(MisconfigurationException, match='should be >= 0'):
            trainer = Trainer(**trainer_options)
    else:
        trainer = Trainer(**trainer_options)

        # fast_dev_run is set to True when it is 1
        if fast_dev_run == 1:
            fast_dev_run = True

        assert trainer.fast_dev_run is fast_dev_run

        if fast_dev_run is True:
            fast_dev_run = 1

        assert trainer.limit_train_batches == fast_dev_run
        assert trainer.limit_val_batches == fast_dev_run
        assert trainer.limit_test_batches == fast_dev_run
        assert trainer.num_sanity_val_steps == 0
        assert trainer.max_epochs == 1

        trainer.fit(model)
        assert not trainer.disable_validation
        assert trainer.num_training_batches == fast_dev_run
        assert trainer.num_val_batches == [fast_dev_run] * len(trainer.val_dataloaders)

        trainer.test(ckpt_path=None)
        assert trainer.num_test_batches == [fast_dev_run] * len(trainer.test_dataloaders)

        # verify sanity check batches match as expected
        num_val_dataloaders = len(model.val_dataloader())
        assert trainer.dev_debugger.num_seen_sanity_check_batches == trainer.num_sanity_val_steps * num_val_dataloaders


@pytest.mark.parametrize('ckpt_path', [None, 'best', 'specific'])
def test_mixing_of_dataloader_options(tmpdir, ckpt_path):
    """Verify that dataloaders can be passed to fit"""

    model = EvalModelTemplate()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    results = trainer.fit(model, val_dataloaders=model.dataloader(train=False))
    assert results

    # fit model
    trainer = Trainer(**trainer_options)
    results = trainer.fit(model, val_dataloaders=model.dataloader(train=False))
    assert results
    if ckpt_path == 'specific':
        ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer.test(test_dataloaders=model.dataloader(train=False), ckpt_path=ckpt_path)

    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 1, \
        f'`test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


def test_train_inf_dataloader_error(tmpdir):
    """Test inf train data loader (e.g. IterableDataset)"""
    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__infinite

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, val_check_interval=0.5)

    with pytest.raises(MisconfigurationException, match='using an IterableDataset'):
        trainer.fit(model)


def test_val_inf_dataloader_error(tmpdir):
    """Test inf train data loader (e.g. IterableDataset)"""
    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__infinite

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_val_batches=0.5)

    with pytest.raises(MisconfigurationException, match='using an IterableDataset'):
        trainer.fit(model)


def test_test_inf_dataloader_error(tmpdir):
    """Test inf train data loader (e.g. IterableDataset)"""
    model = EvalModelTemplate()
    model.test_dataloader = model.test_dataloader__infinite

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_test_batches=0.5)

    with pytest.raises(MisconfigurationException, match='using an IterableDataset'):
        trainer.test(model)


@pytest.mark.parametrize('check_interval', [50, 1.0])
def test_inf_train_dataloader(tmpdir, check_interval):
    """Test inf train data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__infinite

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=check_interval,
    )
    result = trainer.fit(model)
    # verify training completed
    assert result == 1


@pytest.mark.parametrize('check_interval', [1.0])
def test_inf_val_dataloader(tmpdir, check_interval):
    """Test inf val data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__infinite

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=check_interval,
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1


def test_error_on_zero_len_dataloader(tmpdir):
    """ Test that error is raised if a zero-length dataloader is defined """

    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__zero_length

    # fit model
    with pytest.raises(ValueError):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            limit_train_batches=0.1,
            limit_val_batches=0.1,
            limit_test_batches=0.1,
        )
        trainer.fit(model)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Does not apply to Windows platform.')
@pytest.mark.parametrize('ckpt_path', [None, 'best', 'specific'])
@patch('pytorch_lightning.trainer.data_loading.multiprocessing.cpu_count', return_value=4)
def test_warning_with_few_workers(mock, tmpdir, ckpt_path):
    """ Test that error is raised if dataloader with only a few workers is used """

    model = EvalModelTemplate()

    # logger file to get meta
    train_dl = model.dataloader(train=True)
    train_dl.num_workers = 0

    val_dl = model.dataloader(train=False)
    val_dl.num_workers = 0

    train_dl = model.dataloader(train=False)
    train_dl.num_workers = 0

    fit_options = dict(train_dataloader=train_dl,
                       val_dataloaders=val_dl)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )

    # fit model
    with pytest.warns(
        UserWarning, match='The dataloader, train dataloader, does not have many workers which may be a bottleneck.'
    ):
        trainer.fit(model, **fit_options)

    with pytest.warns(
        UserWarning, match='The dataloader, val dataloader 0, does not have many workers which may be a bottleneck.'
    ):
        trainer.fit(model, **fit_options)

    if ckpt_path == 'specific':
        ckpt_path = trainer.checkpoint_callback.best_model_path
    test_options = dict(test_dataloaders=train_dl, ckpt_path=ckpt_path)
    with pytest.warns(
        UserWarning, match='The dataloader, test dataloader 0, does not have many workers which may be a bottleneck.'
    ):
        trainer.test(**test_options)


@pytest.mark.xfail(
    LooseVersion(torch.__version__) < LooseVersion("1.4.0"),
    reason="IterableDataset with __len__ before 1.4 raises",
)
def test_warning_with_iterable_dataset_and_len(tmpdir):
    """ Tests that a warning message is shown when an IterableDataset defines `__len__`. """
    model = EvalModelTemplate()
    original_dataset = model.train_dataloader().dataset

    class IterableWithLen(IterableDataset):

        def __iter__(self):
            return iter(original_dataset)

        def __len__(self):
            return len(original_dataset)

    dataloader = DataLoader(IterableWithLen(), batch_size=16)
    assert has_len(dataloader)
    assert has_iterable_dataset(dataloader)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=3,
    )
    with pytest.warns(UserWarning, match='Your `IterableDataset` has `__len__` defined.'):
        trainer.fit(model, train_dataloader=dataloader, val_dataloaders=[dataloader])
    with pytest.warns(UserWarning, match='Your `IterableDataset` has `__len__` defined.'):
        trainer.test(model, test_dataloaders=[dataloader])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Test requires multiple GPUs')
def test_dataloader_reinit_for_subclass(tmpdir):

    class CustomDataLoader(torch.utils.data.DataLoader):
        def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                     batch_sampler=None, num_workers=0, collate_fn=None,
                     pin_memory=False, drop_last=False, timeout=0,
                     worker_init_fn=None, dummy_kwarg=None, **kwargs):
            super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                             num_workers, collate_fn, pin_memory, drop_last, timeout,
                             worker_init_fn)

            self.dummy_kwarg = dummy_kwarg

    trainer = Trainer(
        gpus=[0, 1],
        num_nodes=1,
        accelerator='ddp_spawn',
        default_root_dir=tmpdir,
    )

    class CustomDummyObj:
        sampler = None

    result = trainer.auto_add_sampler(CustomDummyObj(), shuffle=True)
    assert isinstance(result, CustomDummyObj), "Wrongly reinstantiated data loader"

    dataset = list(range(1000))
    result = trainer.auto_add_sampler(CustomDataLoader(dataset), shuffle=True)
    assert isinstance(result, torch.utils.data.DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert hasattr(result, 'dummy_kwarg')

    # Shuffled DataLoader should also work
    result = trainer.auto_add_sampler(CustomDataLoader(list(range(1000)), shuffle=True), shuffle=True)
    assert isinstance(result, torch.utils.data.DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert hasattr(result, 'dummy_kwarg')

    class CustomSampler(torch.utils.data.Sampler):
        pass

    # Should raise an error if existing sampler is being replaced
    with pytest.raises(MisconfigurationException, match='DistributedSampler'):
        trainer.auto_add_sampler(
            CustomDataLoader(list(range(1000)), sampler=CustomSampler(list(range(1000)))), shuffle=True)


class DistribSamplerCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        train_sampler = trainer.train_dataloader.sampler
        assert isinstance(train_sampler, DistributedSampler)
        assert train_sampler.shuffle

    def on_validation_start(self, trainer, pl_module):
        val_sampler = trainer.val_dataloaders[0].sampler
        assert isinstance(val_sampler, DistributedSampler)
        assert not val_sampler.shuffle

    def on_test_start(self, trainer, pl_module):
        test_sampler = trainer.test_dataloaders[0].sampler
        assert isinstance(test_sampler, DistributedSampler)
        assert not test_sampler.shuffle


@pytest.mark.skipif(platform.system() == 'Windows', reason='Does not apply to Windows platform.')
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Test requires multiple GPUs')
def test_dataloader_distributed_sampler(tmpdir):
    """ Test DistributedSampler and it's arguments for DDP backend """

    model = EvalModelTemplate()
    trainer = Trainer(
        gpus=[0, 1],
        num_nodes=1,
        accelerator='ddp_spawn',
        default_root_dir=tmpdir,
        max_steps=1,
        callbacks=[DistribSamplerCallback()],
    )
    trainer.fit(model)
    trainer.test(ckpt_path=None)


class ModelWithDataLoaderDistributedSampler(EvalModelTemplate):

    def train_dataloader(self):
        dataloader = super().train_dataloader()
        dist_sampler = DistributedSampler(dataloader.dataset, shuffle=True)
        return DataLoader(
            dataloader.dataset,
            batch_size=self.batch_size,
            drop_last=False,
            sampler=dist_sampler,
            shuffle=False
        )


@pytest.mark.skipif(platform.system() == 'Windows', reason='Does not apply to Windows platform.')
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Test requires multiple GPUs')
def test_dataloader_distributed_sampler_already_attached(tmpdir):
    """ Test DistributedSampler and it's arguments for DDP backend when DistSampler already included on dataloader """

    model = ModelWithDataLoaderDistributedSampler()
    trainer = Trainer(
        gpus=[0, 1],
        num_nodes=1,
        accelerator='ddp_spawn',
        default_root_dir=tmpdir,
        max_steps=100,
        callbacks=[DistribSamplerCallback()],
        replace_sampler_ddp=True,
    )
    result = trainer.fit(model)
    assert result == 1, "DDP Training failed"


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason='Test requires multiple GPUs')
def test_batch_size_smaller_than_num_gpus(tmpdir):
    # we need at least 3 gpus for this test
    num_gpus = 3
    batch_size = 3

    class CurrentTestModel(EvalModelTemplate):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # batch norm doesn't work with batch size 1, we replace it
            self.c_d1_bn = torch.nn.ReLU()

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            loss = output['loss']
            # we make sure to add some metrics to the output dict,
            # this is essential for this test
            output['progress_bar'] = {'train_loss': loss}
            return output

        def train_dataloader(self):
            dataloader = super().train_dataloader()
            # construct a dataset with a size that is not divisible by num_gpus
            # therefore the last batch will have a size < num_gpus
            size = num_gpus * batch_size + (num_gpus - 1)
            dataset = Subset(dataloader.dataset, range(size))
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                drop_last=False,
            )
            return dataloader

    hparams = EvalModelTemplate.get_default_hparams()
    hparams['batch_size'] = batch_size
    model = CurrentTestModel(**hparams)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0,
        gpus=num_gpus,
    )

    # we expect the reduction for the metrics also to happen on the last batch
    # where we will get fewer metrics than gpus
    result = trainer.fit(model)
    assert 1 == result


@pytest.mark.parametrize('check_interval', [1.0])
def test_val_dataloader_not_implemented_error(tmpdir, check_interval):
    """Test not_implemented_error data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__not_implemented_error

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=5,
        max_epochs=1,
        val_check_interval=check_interval,
    )
    result = trainer.fit(model)
    # verify training completed
    assert result == 1


@pytest.mark.parametrize('check_interval', [50, 1.0])
def test_train_dataloader_not_implemented_error(tmpdir, check_interval):
    """Test not_implemented_error train data loader (e.g. IterableDataset)"""

    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__not_implemented_error
    model.val_dataloader = model.val_dataloader__not_implemented_error

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=5,
        max_epochs=1,
        val_check_interval=check_interval
    )
    result = trainer.fit(model)
    # verify training completed
    assert result == 1


def test_train_dataloader_not_implemented_error_failed(tmpdir):
    """Test not_implemented_error train data loader (e.g. IterableDataset)"""
    model = EvalModelTemplate()
    model.train_dataloader = model.train_dataloader__not_implemented_error

    trainer = Trainer(default_root_dir=tmpdir, max_steps=5, max_epochs=1, val_check_interval=0.5)

    with pytest.raises(MisconfigurationException, match='using an IterableDataset'):
        trainer.fit(model)


def test_val_dataloader_not_implemented_error_failed(tmpdir):
    """Test not_implemented_error train data loader (e.g. IterableDataset)"""
    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__not_implemented_error

    trainer = Trainer(default_root_dir=tmpdir, max_steps=5, max_epochs=1, limit_val_batches=0.5)

    with pytest.raises(MisconfigurationException, match='using an IterableDataset'):
        trainer.fit(model)


def test_test_dataloader_not_implemented_error_failed(tmpdir):
    """Test not_implemented_error train data loader (e.g. IterableDataset)"""
    model = EvalModelTemplate()
    model.test_dataloader = model.test_dataloader__not_implemented_error

    trainer = Trainer(default_root_dir=tmpdir, max_steps=5, max_epochs=1, limit_test_batches=0.5)

    with pytest.raises(MisconfigurationException, match='using an IterableDataset'):
        trainer.test(model)


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_load_only_once(tmpdir):

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        max_epochs=3,
    )
    result = trainer.fit(model)

    assert len(trainer.dev_debugger.val_dataloader_calls) == 1
    assert len(trainer.dev_debugger.test_dataloader_calls) == 0
    assert len(trainer.dev_debugger.train_dataloader_calls) == 1

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = [
        'val_dataloader',
        'train_dataloader',
    ]
    for call, expected in zip(calls, expected_sequence):
        assert call['name'] == expected


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_load_only_once_val_interval(tmpdir):

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_val_batches=10,
        val_check_interval=0.3,
        reload_dataloaders_every_epoch=True,
        max_epochs=3,
    )
    result = trainer.fit(model)

    trainer.test()

    assert len(trainer.dev_debugger.val_dataloader_calls) == 10
    assert len(trainer.dev_debugger.test_dataloader_calls) == 1
    assert len(trainer.dev_debugger.train_dataloader_calls) == 3

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = [
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'val_dataloader',
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'val_dataloader',
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'val_dataloader',
        'val_dataloader',
        'test_dataloader'
    ]
    for call, expected in zip(calls, expected_sequence):
        assert call['name'] == expected


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_load_only_once_no_sanity_check(tmpdir):

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        num_sanity_val_steps=0,
        max_epochs=3,
    )
    result = trainer.fit(model)

    assert len(trainer.dev_debugger.val_dataloader_calls) == 1
    assert len(trainer.dev_debugger.test_dataloader_calls) == 0
    assert len(trainer.dev_debugger.train_dataloader_calls) == 1

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = [
        'train_dataloader',
        'val_dataloader',
    ]
    for call, expected in zip(calls, expected_sequence):
        assert call['name'] == expected


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_load_every_epoch(tmpdir):

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        reload_dataloaders_every_epoch=True,
        max_epochs=3,
    )
    result = trainer.fit(model)

    trainer.test()

    assert len(trainer.dev_debugger.val_dataloader_calls) == 4
    assert len(trainer.dev_debugger.train_dataloader_calls) == 3
    assert len(trainer.dev_debugger.test_dataloader_calls) == 1

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = [
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'test_dataloader'
    ]
    for call, expected in zip(calls, expected_sequence):
        assert call['name'] == expected


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_load_every_epoch_no_sanity_check(tmpdir):

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        num_sanity_val_steps=0,
        reload_dataloaders_every_epoch=True,
        max_epochs=3,
    )
    result = trainer.fit(model)

    trainer.test()

    assert len(trainer.dev_debugger.val_dataloader_calls) == 3
    assert len(trainer.dev_debugger.train_dataloader_calls) == 3
    assert len(trainer.dev_debugger.test_dataloader_calls) == 1

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = [
        'train_dataloader',
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'train_dataloader',
        'val_dataloader',
        'test_dataloader'
    ]
    for call, expected in zip(calls, expected_sequence):
        assert call['name'] == expected


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_dataloaders_load_only_once_passed_loaders(tmpdir):

    model = EvalModelTemplate()
    train_loader = model.train_dataloader()
    model.train_dataloader = None
    val_loader = model.val_dataloader()
    model.val_dataloader = None
    test_loader = model.test_dataloader()
    model.test_dataloader = None

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        max_epochs=3,
    )
    result = trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)

    assert len(trainer.dev_debugger.val_dataloader_calls) == 1
    assert len(trainer.dev_debugger.test_dataloader_calls) == 1
    assert len(trainer.dev_debugger.train_dataloader_calls) == 1

    # verify the sequence
    calls = trainer.dev_debugger.dataloader_sequence_calls
    expected_sequence = [
        'val_dataloader',
        'train_dataloader',
    ]
    for call, expected in zip(calls, expected_sequence):
        assert call['name'] == expected


def test_replace_sampler_with_multiprocessing_context(tmpdir):
    """
    This test verifies that replace_sampler conserves multiprocessing context
    """
    train = RandomDataset(32, 64)
    context = 'spawn'
    train = DataLoader(train, batch_size=32, num_workers=2, multiprocessing_context=context, shuffle=True)

    class ExtendedBoringModel(BoringModel):

        def train_dataloader(self):
            return train

    trainer = Trainer(
        max_epochs=1,
        progress_bar_refresh_rate=20,
        overfit_batches=5,
    )

    new_data_loader = trainer.replace_sampler(train, SequentialSampler(train.dataset))
    assert (new_data_loader.multiprocessing_context == train.multiprocessing_context)
