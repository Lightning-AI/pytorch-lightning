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
import pickle
from argparse import ArgumentParser
from typing import Any, Dict
from unittest import mock
from unittest.mock import call, PropertyMock

import pytest
import torch

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel
from tests.helpers.utils import reset_seed


@mock.patch("pytorch_lightning.trainer.trainer.Trainer.node_rank", new_callable=PropertyMock)
@mock.patch("pytorch_lightning.trainer.trainer.Trainer.local_rank", new_callable=PropertyMock)
def test_can_prepare_data(local_rank, node_rank):

    dm = BoringDataModule()
    trainer = Trainer()
    trainer.datamodule = dm

    # 1 no DM
    # prepare_data_per_node = True
    # local rank = 0   (True)
    trainer.prepare_data_per_node = True

    local_rank.return_value = 0
    assert trainer.local_rank == 0
    assert trainer.data_connector.can_prepare_data()

    # local rank = 1   (False)
    local_rank.return_value = 1
    assert trainer.local_rank == 1
    assert not trainer.data_connector.can_prepare_data()

    # prepare_data_per_node = False (prepare across all nodes)
    # global rank = 0   (True)
    trainer.prepare_data_per_node = False
    node_rank.return_value = 0
    local_rank.return_value = 0
    assert trainer.data_connector.can_prepare_data()

    # global rank = 1   (False)
    node_rank.return_value = 1
    local_rank.return_value = 0
    assert not trainer.data_connector.can_prepare_data()
    node_rank.return_value = 0
    local_rank.return_value = 1
    assert not trainer.data_connector.can_prepare_data()

    # 2 dm
    # prepar per node = True
    # local rank = 0 (True)
    trainer.prepare_data_per_node = True
    local_rank.return_value = 0

    # is_overridden prepare data = True
    # has been called
    # False
    dm._has_prepared_data = True
    assert not trainer.data_connector.can_prepare_data()

    # has not been called
    # True
    dm._has_prepared_data = False
    assert trainer.data_connector.can_prepare_data()

    # is_overridden prepare data = False
    # True
    dm.prepare_data = None
    assert trainer.data_connector.can_prepare_data()


def test_hooks_no_recursion_error():
    # hooks were appended in cascade every tine a new data module was instantiated leading to a recursion error.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/3652
    class DummyDM(LightningDataModule):

        def setup(self, *args, **kwargs):
            pass

        def prepare_data(self, *args, **kwargs):
            pass

    for i in range(1005):
        dm = DummyDM()
        dm.setup()
        dm.prepare_data()


def test_helper_boringdatamodule():
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup()


def test_helper_boringdatamodule_with_verbose_setup():
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')


def test_data_hooks_called():
    dm = BoringDataModule()
    assert not dm.has_prepared_data
    assert not dm.has_setup_fit
    assert not dm.has_setup_test
    assert not dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.prepare_data()
    assert dm.has_prepared_data
    assert not dm.has_setup_fit
    assert not dm.has_setup_test
    assert not dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.setup()
    assert dm.has_prepared_data
    assert dm.has_setup_fit
    assert dm.has_setup_test
    assert dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.teardown()
    assert dm.has_prepared_data
    assert dm.has_setup_fit
    assert dm.has_setup_test
    assert dm.has_setup_validate
    assert not dm.has_setup_predict
    assert dm.has_teardown_fit
    assert dm.has_teardown_test
    assert dm.has_teardown_validate
    assert not dm.has_teardown_predict


@pytest.mark.parametrize("use_kwarg", (False, True))
def test_data_hooks_called_verbose(use_kwarg):
    dm = BoringDataModule()
    dm.prepare_data()
    assert not dm.has_setup_fit
    assert not dm.has_setup_test
    assert not dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.setup(stage='fit') if use_kwarg else dm.setup('fit')
    assert dm.has_setup_fit
    assert not dm.has_setup_validate
    assert not dm.has_setup_test
    assert not dm.has_setup_predict

    dm.setup(stage='validate') if use_kwarg else dm.setup('validate')
    assert dm.has_setup_fit
    assert dm.has_setup_validate
    assert not dm.has_setup_test
    assert not dm.has_setup_predict

    dm.setup(stage='test') if use_kwarg else dm.setup('test')
    assert dm.has_setup_fit
    assert dm.has_setup_validate
    assert dm.has_setup_test
    assert not dm.has_setup_predict

    dm.setup(stage='predict') if use_kwarg else dm.setup('predict')
    assert dm.has_setup_fit
    assert dm.has_setup_validate
    assert dm.has_setup_test
    assert dm.has_setup_predict

    dm.teardown(stage='fit') if use_kwarg else dm.teardown('fit')
    assert dm.has_teardown_fit
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_test
    assert not dm.has_teardown_predict

    dm.teardown(stage='validate') if use_kwarg else dm.teardown('validate')
    assert dm.has_teardown_fit
    assert dm.has_teardown_validate
    assert not dm.has_teardown_test
    assert not dm.has_teardown_predict

    dm.teardown(stage='test') if use_kwarg else dm.teardown('test')
    assert dm.has_teardown_fit
    assert dm.has_teardown_validate
    assert dm.has_teardown_test
    assert not dm.has_teardown_predict

    dm.teardown(stage='predict') if use_kwarg else dm.teardown('predict')
    assert dm.has_teardown_fit
    assert dm.has_teardown_validate
    assert dm.has_teardown_test
    assert dm.has_teardown_predict


def test_dm_add_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', str(tmpdir)])
    assert args.data_dir == str(tmpdir)


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', str(tmpdir)])
    dm = BoringDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    assert dm.data_dir == args.data_dir == str(tmpdir)


def test_dm_pickle_after_init():
    dm = BoringDataModule()
    pickle.dumps(dm)


def test_train_loop_only(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.callback_metrics['train_loss'] < 1.0


def test_train_val_loop_only(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.callback_metrics['train_loss'] < 1.0


def test_dm_checkpoint_save(tmpdir):

    class CustomBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log('early_stop_on', out['x'])
            return out

    class CustomBoringDataModule(BoringDataModule):

        def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            checkpoint[self.__class__.__name__] = self.__class__.__name__

        def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            self.checkpoint_state = checkpoint.get(self.__class__.__name__)

    reset_seed()
    dm = CustomBoringDataModule()
    model = CustomBoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        weights_summary=None,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on')],
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__name__ in checkpoint
    assert checkpoint[dm.__class__.__name__] == dm.__class__.__name__


def test_full_loop(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]['val_acc'] > 0.7

    # test
    result = trainer.test(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]['test_acc'] > 0.6


@RunIf(min_gpus=1)
@mock.patch("pytorch_lightning.accelerators.accelerator.Accelerator.lightning_module", new_callable=PropertyMock)
def test_dm_apply_batch_transfer_handler(get_module_mock):
    expected_device = torch.device('cuda', 0)

    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestDM(LightningDataModule):
        rank = 0
        transfer_batch_to_device_hook_rank = None
        on_before_batch_transfer_hook_rank = None
        on_after_batch_transfer_hook_rank = None

        def on_before_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx is None
            self.on_before_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.samples += 1
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx is None
            assert batch.samples.device == batch.targets.device == expected_device
            self.on_after_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.targets *= 2
            return batch

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            assert dataloader_idx is None
            self.transfer_batch_to_device_hook_rank = self.rank
            self.rank += 1
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
            return batch

    dm = CurrentTestDM()
    model = BoringModel()

    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    get_module_mock.return_value = model
    if is_overridden('transfer_batch_to_device', dm):
        model.transfer_batch_to_device = dm.transfer_batch_to_device

    model.on_before_batch_transfer = dm.on_before_batch_transfer
    model.transfer_batch_to_device = dm.transfer_batch_to_device
    model.on_after_batch_transfer = dm.on_after_batch_transfer

    batch_gpu = trainer.accelerator.batch_to_device(batch, expected_device)

    assert dm.on_before_batch_transfer_hook_rank == 0
    assert dm.transfer_batch_to_device_hook_rank == 1
    assert dm.on_after_batch_transfer_hook_rank == 2
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected_device
    assert torch.allclose(batch_gpu.samples.cpu(), torch.ones(5, 32))
    assert torch.allclose(batch_gpu.targets.cpu(), torch.ones(5, 1, dtype=torch.long) * 2)


def test_dm_reload_dataloaders_every_epoch(tmpdir):
    """Test datamodule, where trainer argument
    reload_dataloaders_every_epoch is set to True/False"""

    class CustomBoringDataModule(BoringDataModule):

        def __init__(self):
            super().__init__()
            self._epochs_called_for = []

        def train_dataloader(self):
            assert self.trainer.current_epoch not in self._epochs_called_for
            self._epochs_called_for.append(self.trainer.current_epoch)
            return super().train_dataloader()

    dm = CustomBoringDataModule()
    model = BoringModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=0.01,
        reload_dataloaders_every_epoch=True,
    )
    trainer.fit(model, dm)


class DummyDS(torch.utils.data.Dataset):

    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 100


class DummyIDS(torch.utils.data.IterableDataset):

    def __iter__(self):
        yield 1


@pytest.mark.parametrize("iterable", (False, True))
def test_dm_init_from_datasets_dataloaders(iterable):
    ds = DummyIDS if iterable else DummyDS

    train_ds = ds()
    dm = LightningDataModule.from_datasets(train_ds, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_called_once_with(train_ds, batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True)
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None

    train_ds_sequence = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds_sequence, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_has_calls([
            call(train_ds_sequence[0], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True),
            call(train_ds_sequence[1], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True)
        ])
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None

    valid_ds = ds()
    test_ds = ds()
    dm = LightningDataModule.from_datasets(val_dataset=valid_ds, test_dataset=test_ds, batch_size=2, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.val_dataloader()
        dl_mock.assert_called_with(valid_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        dm.test_dataloader()
        dl_mock.assert_called_with(test_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    assert dm.train_dataloader() is None

    valid_dss = [ds(), ds()]
    test_dss = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds, valid_dss, test_dss, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.val_dataloader()
        dm.test_dataloader()
        dl_mock.assert_has_calls([
            call(valid_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
            call(valid_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
            call(test_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
            call(test_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
        ])


def test_datamodule_hooks_calls(tmpdir):
    """Test that repeated calls to DataHooks' hooks have no effect"""

    class TestDataModule(BoringDataModule):
        setup_calls = []
        teardown_calls = []
        prepare_data_calls = 0

        def setup(self, stage=None):
            super().setup(stage=stage)
            self.setup_calls.append(stage)

        def teardown(self, stage=None):
            super().teardown(stage=stage)
            self.teardown_calls.append(stage)

        def prepare_data(self):
            super().prepare_data()
            self.prepare_data_calls += 1

    dm = TestDataModule()
    dm.prepare_data()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('fit')
    dm.setup()
    dm.setup()
    dm.teardown('validate')
    dm.teardown('validate')

    assert dm.prepare_data_calls == 1
    assert dm.setup_calls == ['fit', None]
    assert dm.teardown_calls == ['validate']

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    trainer.test(BoringModel(), datamodule=dm)

    # same number of calls
    assert dm.prepare_data_calls == 1
    assert dm.setup_calls == ['fit', None]
    assert dm.teardown_calls == ['validate', 'test']
