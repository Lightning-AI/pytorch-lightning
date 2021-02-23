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
from unittest.mock import PropertyMock

import pytest
import torch
import torch.nn.functional as F

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringDataModule, BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.simple_models import ClassificationModel
from tests.helpers.utils import reset_seed, set_random_master_port


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


def test_hooks_no_recursion_error(tmpdir):
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


def test_base_datamodule(tmpdir):
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup()


def test_base_datamodule_with_verbose_setup(tmpdir):
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')


def test_data_hooks_called(tmpdir):
    dm = BoringDataModule()
    assert dm.has_prepared_data is False
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.prepare_data()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.setup()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_data_hooks_called_verbose(tmpdir):
    dm = BoringDataModule()
    assert dm.has_prepared_data is False
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.prepare_data()
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is False
    assert dm.has_setup_test is False

    dm.setup('fit')
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is False

    dm.setup('test')
    assert dm.has_prepared_data is True
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


def test_data_hooks_called_with_stage_kwarg(tmpdir):
    dm = BoringDataModule()
    dm.prepare_data()
    assert dm.has_prepared_data is True

    dm.setup(stage='fit')
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is False

    dm.setup(stage='test')
    assert dm.has_setup_fit is True
    assert dm.has_setup_test is True


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


def test_dm_pickle_after_init(tmpdir):
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
    result = trainer.fit(model, datamodule=dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result
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
    result = trainer.fit(model, datamodule=dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result
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
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__name__ in checkpoint
    assert checkpoint[dm.__class__.__name__] == dm.__class__.__name__


def test_test_loop_only(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.test(model, datamodule=dm)


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
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result

    # test
    result = trainer.test(datamodule=dm)
    assert result[0]['test_acc'] > 0.6


def test_trainer_attached_to_dm(tmpdir):
    reset_seed()

    dm = BoringDataModule()
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert dm.trainer is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_full_loop_single_gpu(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        gpus=1,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result

    # test
    result = trainer.test(datamodule=dm)
    assert result[0]['test_acc'] > 0.6


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_loop_dp(tmpdir):
    set_random_master_port()

    class CustomClassificationModelDP(ClassificationModel):

        def _step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            return {'logits': logits, 'y': y}

        def training_step(self, batch, batch_idx):
            _, y = batch
            out = self._step(batch, batch_idx)
            loss = F.cross_entropy(out['logits'], y)
            return loss

        def validation_step(self, batch, batch_idx):
            return self._step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            return self._step(batch, batch_idx)

        def test_step_end(self, outputs):
            self.log('test_acc', self.test_acc(outputs['logits'], outputs['y']))

    dm = ClassifDataModule()
    model = CustomClassificationModelDP()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        accelerator='dp',
        gpus=2,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, datamodule=dm)
    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"
    assert result

    # test
    result = trainer.test(datamodule=dm)
    assert result[0]['test_acc'] > 0.6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
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
            self.on_before_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.samples += 1
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            assert batch.samples.device == batch.targets.device == expected_device
            self.on_after_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.targets *= 2
            return batch

        def transfer_batch_to_device(self, batch, device):
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


def test_dm_init_from_datasets(tmpdir):

    train_ds = DummyDS()
    valid_ds = DummyDS()
    test_ds = DummyDS()

    valid_dss = [DummyDS(), DummyDS()]
    test_dss = [DummyDS(), DummyDS()]

    dm = LightningDataModule.from_datasets(train_ds, batch_size=4, num_workers=0)
    assert torch.all(next(iter(dm.train_dataloader())) == torch.ones(4))
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None

    dm = LightningDataModule.from_datasets(train_ds, valid_ds, test_ds, batch_size=4, num_workers=0)
    assert torch.all(next(iter(dm.val_dataloader())) == torch.ones(4))
    assert torch.all(next(iter(dm.test_dataloader())) == torch.ones(4))

    dm = LightningDataModule.from_datasets(train_ds, valid_dss, test_dss, batch_size=4, num_workers=0)
    assert torch.all(next(iter(dm.val_dataloader()[0])) == torch.ones(4))
    assert torch.all(next(iter(dm.val_dataloader()[1])) == torch.ones(4))
    assert torch.all(next(iter(dm.test_dataloader()[0])) == torch.ones(4))
    assert torch.all(next(iter(dm.test_dataloader()[1])) == torch.ones(4))
