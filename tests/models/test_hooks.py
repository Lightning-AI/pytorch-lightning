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
from unittest import mock
from unittest.mock import PropertyMock

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from tests.helpers import BoringDataModule, BoringModel, RandomDataset
from tests.helpers.runif import RunIf


@pytest.mark.parametrize('max_steps', [1, 2, 3])
def test_on_before_zero_grad_called(tmpdir, max_steps):

    class CurrentTestModel(BoringModel):
        on_before_zero_grad_called = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_called += 1

    model = CurrentTestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=max_steps,
        max_epochs=2,
    )
    assert 0 == model.on_before_zero_grad_called
    trainer.fit(model)
    assert max_steps == model.on_before_zero_grad_called

    model.on_before_zero_grad_called = 0
    trainer.test(model)
    assert 0 == model.on_before_zero_grad_called


def test_training_epoch_end_metrics_collection(tmpdir):
    """ Test that progress bar metrics also get collected at the end of an epoch. """
    num_epochs = 3

    class CurrentModel(BoringModel):

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            self.log_dict({'step_metric': torch.tensor(-1), 'shared_metric': 100}, logger=False, prog_bar=True)
            return output

        def training_epoch_end(self, outputs):
            epoch = self.current_epoch
            # both scalar tensors and Python numbers are accepted
            self.log_dict(
                {
                    f'epoch_metric_{epoch}': torch.tensor(epoch),
                    'shared_metric': 111
                },
                logger=False,
                prog_bar=True,
            )

    model = CurrentModel()
    trainer = Trainer(
        max_epochs=num_epochs,
        default_root_dir=tmpdir,
        overfit_batches=2,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    metrics = trainer.progress_bar_dict

    # metrics added in training step should be unchanged by epoch end method
    assert metrics['step_metric'] == -1
    # a metric shared in both methods gets overwritten by epoch_end
    assert metrics['shared_metric'] == 111
    # metrics are kept after each epoch
    for i in range(num_epochs):
        assert metrics[f'epoch_metric_{i}'] == i


def test_training_epoch_end_metrics_collection_on_override(tmpdir):
    """ Test that batch end metrics are collected when training_epoch_end is overridden at the end of an epoch. """

    class OverriddenModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.len_outputs = 0

        def on_train_epoch_start(self):
            self.num_train_batches = 0

        def training_epoch_end(self, outputs):
            self.len_outputs = len(outputs)

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.num_train_batches += 1

    class NotOverriddenModel(BoringModel):

        def on_train_epoch_start(self):
            self.num_train_batches = 0

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.num_train_batches += 1

    overridden_model = OverriddenModel()
    not_overridden_model = NotOverriddenModel()
    not_overridden_model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        overfit_batches=2,
    )

    trainer.fit(overridden_model)
    assert overridden_model.len_outputs == overridden_model.num_train_batches


@RunIf(min_gpus=1)
@mock.patch("pytorch_lightning.accelerators.accelerator.Accelerator.lightning_module", new_callable=PropertyMock)
def test_apply_batch_transfer_handler(model_getter_mock):
    expected_device = torch.device('cuda', 0)

    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestModel(BoringModel):
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

    model = CurrentTestModel()
    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead

    model_getter_mock.return_value = model
    batch_gpu = trainer.accelerator.batch_to_device(batch, expected_device)

    assert model.on_before_batch_transfer_hook_rank == 0
    assert model.transfer_batch_to_device_hook_rank == 1
    assert model.on_after_batch_transfer_hook_rank == 2
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected_device
    assert torch.allclose(batch_gpu.samples.cpu(), torch.ones(5, 32))
    assert torch.allclose(batch_gpu.targets.cpu(), torch.ones(5, 1, dtype=torch.long) * 2)


@RunIf(min_gpus=2, special=True)
def test_transfer_batch_hook_ddp(tmpdir):
    """
    Test custom data are properly moved to the right device using ddp
    """

    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]

        def to(self, device, **kwargs):
            self.samples = self.samples.to(device, **kwargs)
            return self

    def collate_fn(batch):
        return CustomBatch(batch)

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            assert batch.samples.device == self.device
            assert isinstance(batch_idx, int)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(RandomDataset(32, 64), collate_fn=collate_fn)

    model = TestModel()
    model.validation_step = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        weights_summary=None,
        accelerator="ddp",
        gpus=2,
    )
    trainer.fit(model)


class HookedModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.called = []
        self.train_batch = [
            'on_train_batch_start',
            'on_before_batch_transfer',
            'transfer_batch_to_device',
            'on_after_batch_transfer',
            'training_step',
            'on_before_zero_grad',
            'optimizer_zero_grad',
            'backward',
            'on_after_backward',
            'optimizer_step',
            'on_train_batch_end',
        ]
        self.val_batch = [
            'on_validation_batch_start',
            'on_before_batch_transfer',
            'transfer_batch_to_device',
            'on_after_batch_transfer',
            'on_validation_batch_end',
        ]

    def prepare_data(self):
        self.called.append("prepare_data")
        return super().prepare_data()

    def configure_callbacks(self):
        self.called.append("configure_callbacks")
        return super().configure_callbacks()

    def configure_optimizers(self):
        self.called.append("configure_optimizers")
        return super().configure_optimizers()

    def training_step(self, *args, **kwargs):
        self.called.append("training_step")
        return super().training_step(*args, **kwargs)

    def optimizer_zero_grad(self, *args, **kwargs):
        self.called.append("optimizer_zero_grad")
        super().optimizer_zero_grad(*args, **kwargs)

    def training_epoch_end(self, *args, **kwargs):
        self.called.append("training_epoch_end")
        super().training_epoch_end(*args, **kwargs)

    def backward(self, *args, **kwargs):
        self.called.append("backward")
        super().backward(*args, **kwargs)

    def on_after_backward(self):
        self.called.append("on_after_backward")
        super().on_after_backward()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.called.append("optimizer_step")  # append after as closure calls other methods

    def validation_epoch_end(self, *args, **kwargs):
        self.called.append("validation_epoch_end")
        super().validation_epoch_end(*args, **kwargs)

    def on_before_zero_grad(self, *args, **kwargs):
        self.called.append("on_before_zero_grad")
        super().on_before_zero_grad(*args, **kwargs)

    def on_epoch_start(self):
        self.called.append("on_epoch_start")
        super().on_epoch_start()

    def on_epoch_end(self):
        self.called.append("on_epoch_end")
        super().on_epoch_end()

    def on_fit_start(self):
        self.called.append("on_fit_start")
        super().on_fit_start()

    def on_fit_end(self):
        self.called.append("on_fit_end")
        super().on_fit_end()

    def on_hpc_load(self, *args, **kwargs):
        self.called.append("on_hpc_load")
        super().on_hpc_load(*args, **kwargs)

    def on_hpc_save(self, *args, **kwargs):
        self.called.append("on_hpc_save")
        super().on_hpc_save(*args, **kwargs)

    def on_load_checkpoint(self, *args, **kwargs):
        self.called.append("on_load_checkpoint")
        super().on_load_checkpoint(*args, **kwargs)

    def on_save_checkpoint(self, *args, **kwargs):
        self.called.append("on_save_checkpoint")
        super().on_save_checkpoint(*args, **kwargs)

    def on_pretrain_routine_start(self):
        self.called.append("on_pretrain_routine_start")
        super().on_pretrain_routine_start()

    def on_pretrain_routine_end(self):
        self.called.append("on_pretrain_routine_end")
        super().on_pretrain_routine_end()

    def on_train_start(self):
        self.called.append("on_train_start")
        super().on_train_start()

    def on_train_end(self):
        self.called.append("on_train_end")
        super().on_train_end()

    def on_before_batch_transfer(self, *args, **kwargs):
        self.called.append("on_before_batch_transfer")
        return super().on_before_batch_transfer(*args, **kwargs)

    def transfer_batch_to_device(self, *args, **kwargs):
        self.called.append("transfer_batch_to_device")
        return super().transfer_batch_to_device(*args, **kwargs)

    def on_after_batch_transfer(self, *args, **kwargs):
        self.called.append("on_after_batch_transfer")
        return super().on_after_batch_transfer(*args, **kwargs)

    def on_train_batch_start(self, *args, **kwargs):
        self.called.append("on_train_batch_start")
        super().on_train_batch_start(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        self.called.append("on_train_batch_end")
        super().on_train_batch_end(*args, **kwargs)

    def on_train_epoch_start(self):
        self.called.append("on_train_epoch_start")
        super().on_train_epoch_start()

    def on_train_epoch_end(self):
        self.called.append("on_train_epoch_end")
        super().on_train_epoch_end()

    def on_validation_start(self):
        self.called.append("on_validation_start")
        super().on_validation_start()

    def on_validation_end(self):
        self.called.append("on_validation_end")
        super().on_validation_end()

    def on_validation_batch_start(self, *args, **kwargs):
        self.called.append("on_validation_batch_start")
        super().on_validation_batch_start(*args, **kwargs)

    def on_validation_batch_end(self, *args, **kwargs):
        self.called.append("on_validation_batch_end")
        super().on_validation_batch_end(*args, **kwargs)

    def on_validation_epoch_start(self):
        self.called.append("on_validation_epoch_start")
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.called.append("on_validation_epoch_end")
        super().on_validation_epoch_end(*args, **kwargs)

    def on_test_start(self):
        self.called.append("on_test_start")
        super().on_test_start()

    def on_test_batch_start(self, *args, **kwargs):
        self.called.append("on_test_batch_start")
        super().on_test_batch_start(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self.called.append("on_test_batch_end")
        super().on_test_batch_end(*args, **kwargs)

    def on_test_epoch_start(self):
        self.called.append("on_test_epoch_start")
        super().on_test_epoch_start()

    def on_test_epoch_end(self, *args, **kwargs):
        self.called.append("on_test_epoch_end")
        super().on_test_epoch_end(*args, **kwargs)

    def on_validation_model_eval(self):
        self.called.append("on_validation_model_eval")
        super().on_validation_model_eval()

    def on_validation_model_train(self):
        self.called.append("on_validation_model_train")
        super().on_validation_model_train()

    def on_test_model_eval(self):
        self.called.append("on_test_model_eval")
        super().on_test_model_eval()

    def on_test_model_train(self):
        self.called.append("on_test_model_train")
        super().on_test_model_train()

    def on_test_end(self):
        self.called.append("on_test_end")
        super().on_test_end()

    def setup(self, stage=None):
        self.called.append(f"setup_{stage}")
        super().setup(stage=stage)

    def teardown(self, stage=None):
        self.called.append(f"teardown_{stage}")
        super().teardown(stage)


def test_trainer_model_hook_system_fit(tmpdir):
    model = HookedModel()
    train_batches = 2
    val_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    assert model.called == []
    trainer.fit(model)
    expected = [
        'prepare_data',
        'configure_callbacks',
        'setup_fit',
        'configure_optimizers',
        'on_fit_start',
        'on_pretrain_routine_start',
        'on_pretrain_routine_end',
        'on_validation_model_eval',
        'on_validation_start',
        'on_epoch_start',
        'on_validation_epoch_start',
        *(model.val_batch * val_batches),
        'validation_epoch_end',
        'on_validation_epoch_end',
        'on_epoch_end',
        'on_validation_end',
        'on_validation_model_train',
        'on_train_start',
        'on_epoch_start',
        'on_train_epoch_start',
        *(model.train_batch * train_batches),
        'on_validation_model_eval',
        'on_validation_start',
        'on_epoch_start',
        'on_validation_epoch_start',
        *(model.val_batch * val_batches),
        'validation_epoch_end',
        'on_validation_epoch_end',
        'on_epoch_end',
        'on_save_checkpoint',
        'on_validation_end',
        'on_validation_model_train',
        'training_epoch_end',
        'on_train_epoch_end',
        'on_epoch_end',
        'on_train_end',
        'on_fit_end',
        'teardown_fit',
    ]
    assert model.called == expected


def test_trainer_model_hook_system_fit_no_val(tmpdir):
    model = HookedModel()
    train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=train_batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    assert model.called == []
    trainer.fit(model)
    expected = [
        'prepare_data',
        'configure_callbacks',
        'setup_fit',
        'configure_optimizers',
        'on_fit_start',
        'on_pretrain_routine_start',
        'on_pretrain_routine_end',
        'on_train_start',
        'on_epoch_start',
        'on_train_epoch_start',
        *(model.train_batch * train_batches),
        'training_epoch_end',
        'on_train_epoch_end',
        'on_epoch_end',
        'on_save_checkpoint',  # from train epoch end
        'on_train_end',
        'on_fit_end',
        'teardown_fit',
    ]
    assert model.called == expected


def test_trainer_model_hook_system_validate(tmpdir):
    model = HookedModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    assert model.called == []
    trainer.validate(model, verbose=False)
    expected = [
        'prepare_data',
        'configure_callbacks',
        'setup_validate',
        'on_validation_model_eval',
        'on_validation_start',
        'on_epoch_start',
        'on_validation_epoch_start',
        'on_validation_batch_start',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'on_validation_batch_end',
        'validation_epoch_end',
        'on_validation_epoch_end',
        'on_epoch_end',
        'on_validation_end',
        'on_validation_model_train',
        'teardown_validate',
    ]
    assert model.called == expected


def test_trainer_model_hook_system_test(tmpdir):
    model = HookedModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_test_batches=1,
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    assert model.called == []
    trainer.test(model, verbose=False)
    expected = [
        'prepare_data',
        'configure_callbacks',
        'setup_test',
        'on_test_model_eval',
        'on_test_start',
        'on_epoch_start',
        'on_test_epoch_start',
        'on_test_batch_start',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'on_test_batch_end',
        'on_test_epoch_end',
        'on_epoch_end',
        'on_test_end',
        'on_test_model_train',
        'teardown_test',
    ]
    assert model.called == expected


def test_hooks_with_different_argument_names(tmpdir):
    """
    Test that argument names can be anything in the hooks
    """

    class CustomBoringModel(BoringModel):

        def assert_args(self, x, batch_nb):
            assert isinstance(x, torch.Tensor)
            assert x.size() == (1, 32)
            assert isinstance(batch_nb, int)

        def training_step(self, x1, batch_nb1):
            self.assert_args(x1, batch_nb1)
            return super().training_step(x1, batch_nb1)

        def validation_step(self, x2, batch_nb2):
            self.assert_args(x2, batch_nb2)
            return super().validation_step(x2, batch_nb2)

        def test_step(self, x3, batch_nb3, dl_idx3):
            self.assert_args(x3, batch_nb3)
            assert isinstance(dl_idx3, int)
            return super().test_step(x3, batch_nb3)

        def predict(self, x4, batch_nb4, dl_idx4):
            self.assert_args(x4, batch_nb4)
            assert isinstance(dl_idx4, int)
            return super().predict(x4, batch_nb4, dl_idx4)

        def test_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

        def predict_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    model = CustomBoringModel()
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=5,
    )

    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    trainer.test(ckpt_path=None)

    preds = trainer.predict(model)
    assert len(preds) == 2
    assert all(len(x) == 5 for x in preds)


def test_trainer_datamodule_hook_system(tmpdir):
    """Test the LightningDataModule hook system."""

    class HookedDataModule(BoringDataModule):

        def __init__(self):
            super().__init__()
            self.called = []

        def prepare_data(self):
            self.called.append("prepare_data")
            super().prepare_data()

        def setup(self, stage=None):
            self.called.append(f"setup_{stage}")
            super().setup(stage=stage)

        def teardown(self, stage=None):
            self.called.append(f"teardown_{stage}")
            super().teardown(stage=stage)

        def train_dataloader(self):
            self.called.append("train_dataloader")
            return super().train_dataloader()

        def test_dataloader(self):
            self.called.append("test_dataloader")
            return super().test_dataloader()

        def val_dataloader(self):
            self.called.append("val_dataloader")
            return super().val_dataloader()

        def predict_dataloader(self):
            self.called.append("predict_dataloader")

        def transfer_batch_to_device(self, *args, **kwargs):
            self.called.append("transfer_batch_to_device")
            return super().transfer_batch_to_device(*args, **kwargs)

        def on_before_batch_transfer(self, *args, **kwargs):
            self.called.append("on_before_batch_transfer")
            return super().on_before_batch_transfer(*args, **kwargs)

        def on_after_batch_transfer(self, *args, **kwargs):
            self.called.append("on_after_batch_transfer")
            return super().on_after_batch_transfer(*args, **kwargs)

    model = BoringModel()
    dm = HookedDataModule()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=2,
        limit_test_batches=1,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        reload_dataloaders_every_epoch=True,
    )
    trainer.fit(model, datamodule=dm)
    expected = [
        'prepare_data',
        'setup_fit',
        'val_dataloader',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'train_dataloader',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'val_dataloader',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'teardown_fit',
    ]
    assert dm.called == expected

    dm = HookedDataModule()
    trainer.validate(model, datamodule=dm, verbose=False)
    expected = [
        'prepare_data',
        'setup_validate',
        'val_dataloader',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'teardown_validate',
    ]
    assert dm.called == expected

    dm = HookedDataModule()
    trainer.test(model, datamodule=dm, verbose=False)
    expected = [
        'prepare_data',
        'setup_test',
        'test_dataloader',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'teardown_test',
    ]
    assert dm.called == expected
