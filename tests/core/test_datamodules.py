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
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.accelerators.gpu_accelerator import GPUAccelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_utils import is_overridden
from tests.base import EvalModelTemplate
from tests.base.datamodules import TrialMNISTDataModule
from tests.base.datasets import TrialMNIST
from tests.base.develop_utils import reset_seed


def test_can_prepare_data(tmpdir):

    dm = TrialMNISTDataModule()
    trainer = Trainer()
    trainer.datamodule = dm

    # 1 no DM
    # prepare_data_per_node = True
    # local rank = 0   (True)
    trainer.prepare_data_per_node = True
    trainer.local_rank = 0
    assert trainer.data_connector.can_prepare_data()

    # local rank = 1   (False)
    trainer.local_rank = 1
    assert not trainer.data_connector.can_prepare_data()

    # prepare_data_per_node = False (prepare across all nodes)
    # global rank = 0   (True)
    trainer.prepare_data_per_node = False
    trainer.node_rank = 0
    trainer.local_rank = 0
    assert trainer.data_connector.can_prepare_data()

    # global rank = 1   (False)
    trainer.node_rank = 1
    trainer.local_rank = 0
    assert not trainer.data_connector.can_prepare_data()
    trainer.node_rank = 0
    trainer.local_rank = 1
    assert not trainer.data_connector.can_prepare_data()

    # 2 dm
    # prepar per node = True
    # local rank = 0 (True)
    trainer.prepare_data_per_node = True
    trainer.local_rank = 0

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
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup()


def test_base_datamodule_with_verbose_setup(tmpdir):
    dm = TrialMNISTDataModule()
    dm.prepare_data()
    dm.setup('fit')
    dm.setup('test')


def test_data_hooks_called(tmpdir):
    dm = TrialMNISTDataModule()
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
    dm = TrialMNISTDataModule()
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
    dm = TrialMNISTDataModule()
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
    parser = TrialMNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', str(tmpdir)])
    assert args.data_dir == str(tmpdir)


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = TrialMNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args(['--data_dir', str(tmpdir)])
    dm = TrialMNISTDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()


def test_dm_pickle_after_init(tmpdir):
    dm = TrialMNISTDataModule()
    pickle.dumps(dm)


def test_train_loop_only(tmpdir):
    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()
    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1
    assert trainer.logger_connector.callback_metrics['loss'] < 0.6


def test_train_val_loop_only(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()
    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1
    assert trainer.logger_connector.callback_metrics['loss'] < 0.6


def test_dm_checkpoint_save(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on')],
    )

    # fit model
    result = trainer.fit(model, dm)
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__name__ in checkpoint
    assert checkpoint[dm.__class__.__name__] == dm.__class__.__name__


def test_test_loop_only(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.test(model, datamodule=dm)


def test_full_loop(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


def test_trainer_attached_to_dm(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1
    assert dm.trainer is not None

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert dm.trainer is not None


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires multi-GPU machine")
def test_full_loop_single_gpu(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        gpus=1,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_loop_dp(tmpdir):
    reset_seed()

    dm = TrialMNISTDataModule(tmpdir)

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
        accelerator='dp',
        gpus=2,
        deterministic=True,
    )

    # fit model
    result = trainer.fit(model, dm)
    assert result == 1

    # test
    result = trainer.test(datamodule=dm)
    result = result[0]
    assert result['test_acc'] > 0.8


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="test requires multi-GPU machine")
def test_dm_transfer_batch_to_device(tmpdir):
    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestDM(LightningDataModule):

        hook_called = False

        def transfer_batch_to_device(self, data, device):
            self.hook_called = True
            if isinstance(data, CustomBatch):
                data.samples = data.samples.to(device)
                data.targets = data.targets.to(device)
            else:
                data = super().transfer_batch_to_device(data, device)
            return data

    model = EvalModelTemplate()
    dm = CurrentTestDM()
    batch = CustomBatch((torch.zeros(5, 28), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    trainer.get_model = MagicMock(return_value=model)
    if is_overridden('transfer_batch_to_device', dm):
        model.transfer_batch_to_device = dm.transfer_batch_to_device

    trainer.accelerator_backend = GPUAccelerator(trainer)
    batch_gpu = trainer.accelerator_backend.batch_to_device(batch, torch.device('cuda:0'))
    expected = torch.device('cuda', 0)
    assert dm.hook_called
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected


class CustomMNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self._epochs_called_for = []

    def prepare_data(self):
        TrialMNIST(self.data_dir, train=True, download=True)

    def setup(self, stage: Optional[str] = None):

        mnist_full = TrialMNIST(
            root=self.data_dir, train=True, num_samples=64, download=True
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [128, 64])
        self.dims = self.mnist_train[0][0].shape

    def train_dataloader(self):
        assert self.trainer.current_epoch not in self._epochs_called_for
        self._epochs_called_for.append(self.trainer.current_epoch)

        return DataLoader(self.mnist_train, batch_size=4)


def test_dm_reload_dataloaders_every_epoch(tmpdir):
    """Test datamodule, where trainer argument
    reload_dataloaders_every_epoch is set to True/False"""

    dm = CustomMNISTDataModule(tmpdir)

    model = EvalModelTemplate()
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
