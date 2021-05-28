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
import inspect
from functools import partial
from unittest import mock
from unittest.mock import PropertyMock

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Callback, LightningModule, Trainer
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


class HookedCallback(Callback):

    def __init__(self, called):

        def call(hook, *_, **kwargs):
            name = f'Callback.{hook}'
            if 'stage' in kwargs:
                name += f'_{kwargs["stage"]}'
            called.append(name)

        hooks = [h for h, _ in inspect.getmembers(Callback, predicate=inspect.isfunction)]
        for h in hooks:
            setattr(self, h, partial(call, h))


class HookedModel(BoringModel):

    def __init__(self, called):
        super().__init__()
        # yapf: disable
        self.train_batch = [
            'Callback.on_batch_start',
            'Callback.on_train_batch_start', 'on_train_batch_start',
            'on_before_batch_transfer',
            'transfer_batch_to_device',
            'on_after_batch_transfer',
            'training_step',
            'training_step_end',
            'Callback.on_before_zero_grad', 'on_before_zero_grad',
            'optimizer_zero_grad',
            'backward',
            'Callback.on_after_backward', 'on_after_backward',
            'optimizer_step',
            'Callback.on_train_batch_end', 'on_train_batch_end',
            'Callback.on_batch_end',
        ]
        self.val_batch = [
            'Callback.on_validation_batch_start', 'on_validation_batch_start',
            'on_before_batch_transfer',
            'transfer_batch_to_device',
            'on_after_batch_transfer',
            'validation_step',
            'validation_step_end',
            'Callback.on_validation_batch_end', 'on_validation_batch_end',
        ]
        # yapf: enable

        def get_members(cls):
            return {h for h, _ in inspect.getmembers(cls, predicate=inspect.isfunction) if not h.startswith('_')}

        pl_module_hooks = get_members(LightningModule)
        # remove `nn.Module` hooks
        module_hooks = get_members(torch.nn.Module)
        pl_module_hooks.difference_update(module_hooks)

        def call(hook, fn):

            def add(*args, **kwargs):
                out = fn(*args, **kwargs)
                name = hook
                if 'stage' in kwargs:
                    name += f'_{kwargs["stage"]}'
                called.append(name)
                return out

            return add

        for h in pl_module_hooks:
            attr = getattr(self, h)
            # can't use partial here because `is_overridden` fails with
            # AttributeError: 'functools.partial' object has no attribute '__code__'
            setattr(self, h, call(h, attr))

    def validation_epoch_end(self, *args, **kwargs):
        # `BoringModel` does not have a return for `validation_step_end` so this would fail
        pass

    def test_epoch_end(self, *args, **kwargs):
        # `BoringModel` does not have a return for `test_step_end` so this would fail
        pass


def test_trainer_model_hook_system_fit(tmpdir):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    train_batches = 2
    val_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        callbacks=[callback]
    )
    assert called == ['Callback.on_init_start', 'Callback.on_init_end']
    trainer.fit(model)
    # yapf: disable
    expected = [
        'Callback.on_init_start',
        'Callback.on_init_end',
        'prepare_data',
        'configure_callbacks',
        'Callback.on_before_accelerator_backend_setup',
        'Callback.setup_fit', 'setup_fit',
        'configure_sharded_model',
        'Callback.on_configure_sharded_model',
        'configure_optimizers',
        'Callback.on_fit_start', 'on_fit_start',
        'Callback.on_pretrain_routine_start', 'on_pretrain_routine_start',
        'Callback.on_pretrain_routine_end', 'on_pretrain_routine_end',
        'Callback.on_sanity_check_start',
        'on_val_dataloader',
        'val_dataloader',
        'on_validation_model_eval',
        'Callback.on_validation_start', 'on_validation_start',
        'Callback.on_epoch_start', 'on_epoch_start',
        'Callback.on_validation_epoch_start', 'on_validation_epoch_start',
        *(model.val_batch * val_batches),
        'validation_epoch_end',
        'Callback.on_validation_epoch_end', 'on_validation_epoch_end',
        'Callback.on_epoch_end', 'on_epoch_end',
        'Callback.on_validation_end', 'on_validation_end',
        'on_validation_model_train',
        'Callback.on_sanity_check_end',
        'on_train_dataloader',
        'train_dataloader',
        'Callback.on_train_start', 'on_train_start',
        'Callback.on_epoch_start', 'on_epoch_start',
        'Callback.on_train_epoch_start', 'on_train_epoch_start',
        *(model.train_batch * train_batches),
        'on_validation_model_eval',
        'Callback.on_validation_start', 'on_validation_start',
        'Callback.on_epoch_start', 'on_epoch_start',
        'Callback.on_validation_epoch_start', 'on_validation_epoch_start',
        *(model.val_batch * val_batches),
        'validation_epoch_end',
        'Callback.on_validation_epoch_end', 'on_validation_epoch_end',
        'Callback.on_epoch_end', 'on_epoch_end',
        'Callback.on_validation_end',
        'Callback.on_save_checkpoint', 'on_save_checkpoint',
        'on_validation_end',
        'on_validation_model_train',
        'training_epoch_end',
        'Callback.on_train_epoch_end', 'on_train_epoch_end',
        'Callback.on_epoch_end', 'on_epoch_end',
        'Callback.on_train_end', 'on_train_end',
        'Callback.on_fit_end', 'on_fit_end',
        'Callback.teardown_fit', 'teardown_fit',
    ]
    # yapf: enable
    assert called == expected


def test_trainer_model_hook_system_fit_no_val(tmpdir):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    train_batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0,
        limit_train_batches=train_batches,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        callbacks=[callback],
    )
    assert called == ['Callback.on_init_start', 'Callback.on_init_end']
    trainer.fit(model)
    # yapf: disable
    expected = [
        'Callback.on_init_start',
        'Callback.on_init_end',
        'prepare_data',
        'configure_callbacks',
        'Callback.on_before_accelerator_backend_setup',
        'Callback.setup_fit', 'setup_fit',
        'configure_sharded_model',
        'Callback.on_configure_sharded_model',
        'configure_optimizers',
        'Callback.on_fit_start', 'on_fit_start',
        'Callback.on_pretrain_routine_start', 'on_pretrain_routine_start',
        'Callback.on_pretrain_routine_end', 'on_pretrain_routine_end',
        'on_train_dataloader',
        'train_dataloader',
        'on_val_dataloader',
        'val_dataloader',
        'Callback.on_train_start', 'on_train_start',
        'Callback.on_epoch_start', 'on_epoch_start',
        'Callback.on_train_epoch_start', 'on_train_epoch_start',
        *(model.train_batch * train_batches),
        'training_epoch_end',
        'Callback.on_train_epoch_end', 'on_train_epoch_end',
        'Callback.on_epoch_end', 'on_epoch_end',
        'Callback.on_save_checkpoint', 'on_save_checkpoint',  # from train epoch end
        'Callback.on_train_end', 'on_train_end',
        'Callback.on_fit_end', 'on_fit_end',
        'Callback.teardown_fit', 'teardown_fit',
    ]
    # yapf: enable
    assert called == expected


def test_trainer_model_hook_system_validate(tmpdir):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        callbacks=[callback],
    )
    assert called == ['Callback.on_init_start', 'Callback.on_init_end']
    trainer.validate(model, verbose=False)
    # yapf: disable
    expected = [
        'Callback.on_init_start',
        'Callback.on_init_end',
        'prepare_data',
        'configure_callbacks',
        'Callback.on_before_accelerator_backend_setup',
        'Callback.setup_validate', 'setup_validate',
        'configure_sharded_model',
        'Callback.on_configure_sharded_model',
        'on_val_dataloader',
        'val_dataloader',
        'on_validation_model_eval',
        'Callback.on_validation_start', 'on_validation_start',
        'Callback.on_epoch_start', 'on_epoch_start',
        'Callback.on_validation_epoch_start', 'on_validation_epoch_start',
        *model.val_batch,
        'validation_epoch_end',
        'Callback.on_validation_epoch_end', 'on_validation_epoch_end',
        'Callback.on_epoch_end', 'on_epoch_end',
        'Callback.on_validation_end', 'on_validation_end',
        'on_validation_model_train',
        'Callback.teardown_validate', 'teardown_validate',
    ]
    # yapf: enable
    assert called == expected


def test_trainer_model_hook_system_test(tmpdir):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_test_batches=1,
        progress_bar_refresh_rate=0,
        callbacks=[callback],
    )
    assert called == ['Callback.on_init_start', 'Callback.on_init_end']
    trainer.test(model, verbose=False)
    # yapf: disable
    expected = [
        'Callback.on_init_start',
        'Callback.on_init_end',
        'prepare_data',
        'configure_callbacks',
        'Callback.on_before_accelerator_backend_setup',
        'Callback.setup_test', 'setup_test',
        'configure_sharded_model',
        'Callback.on_configure_sharded_model',
        'on_test_dataloader',
        'test_dataloader',
        'on_test_model_eval',
        'Callback.on_test_start', 'on_test_start',
        'Callback.on_epoch_start', 'on_epoch_start',
        'Callback.on_test_epoch_start', 'on_test_epoch_start',
        'Callback.on_test_batch_start', 'on_test_batch_start',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'test_step',
        'test_step_end',
        'Callback.on_test_batch_end', 'on_test_batch_end',
        'test_epoch_end',
        'Callback.on_test_epoch_end', 'on_test_epoch_end',
        'Callback.on_epoch_end', 'on_epoch_end',
        'Callback.on_test_end', 'on_test_end',
        'on_test_model_train',
        'Callback.teardown_test', 'teardown_test',
    ]
    # yapf: enable
    assert called == expected


def test_trainer_model_hook_system_predict(tmpdir):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_predict_batches=1,
        progress_bar_refresh_rate=0,
        callbacks=[callback],
    )
    assert called == ['Callback.on_init_start', 'Callback.on_init_end']
    trainer.predict(model)
    # yapf: disable
    expected = [
        'Callback.on_init_start',
        'Callback.on_init_end',
        'prepare_data',
        'configure_callbacks',
        'Callback.on_before_accelerator_backend_setup',
        'Callback.setup_predict', 'setup_predict',
        'configure_sharded_model',
        'Callback.on_configure_sharded_model',
        'on_predict_dataloader',
        'predict_dataloader',
        'on_predict_model_eval',
        'Callback.on_predict_start', 'on_predict_start',
        # 'Callback.on_epoch_start', 'on_epoch_start',  TODO: missing
        'Callback.on_predict_epoch_start', 'on_predict_epoch_start',
        'Callback.on_predict_batch_start', 'on_predict_batch_start',
        'on_before_batch_transfer',
        'transfer_batch_to_device',
        'on_after_batch_transfer',
        'predict_step',
        'Callback.on_predict_batch_end', 'on_predict_batch_end',
        'Callback.on_predict_epoch_end', 'on_predict_epoch_end',
        # 'Callback.on_epoch_end', 'on_epoch_end',  TODO: missing
        'Callback.on_predict_end', 'on_predict_end',
        # 'on_predict_model_train', TODO: missing
        'Callback.teardown_predict', 'teardown_predict',
    ]
    # yapf: enable
    assert called == expected


# TODO: add test for tune


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
