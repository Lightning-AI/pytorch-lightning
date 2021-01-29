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
import os
from unittest.mock import MagicMock

import pytest
import torch


from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.gpu_accelerator import GPUAccelerator
import pytorch_lightning as pl
from tests.base import BoringModel, EvalModelTemplate, RandomDataset


@pytest.mark.parametrize('max_steps', [1, 2, 3])
def test_on_before_zero_grad_called(tmpdir, max_steps):

    class CurrentTestModel(EvalModelTemplate):
        on_before_zero_grad_called = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_called += 1

    model = CurrentTestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=max_steps,
        max_epochs=2,
        num_sanity_val_steps=5,
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

    class CurrentModel(EvalModelTemplate):

        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            output['progress_bar'].update({'step_metric': torch.tensor(-1)})
            output['progress_bar'].update({'shared_metric': 100})
            return output

        def training_epoch_end(self, outputs):
            epoch = self.current_epoch
            # both scalar tensors and Python numbers are accepted
            return {
                'progress_bar': {
                    f'epoch_metric_{epoch}': torch.tensor(epoch),  # add a new metric key every epoch
                    'shared_metric': 111,
                }
            }

    model = CurrentModel()
    trainer = Trainer(
        max_epochs=num_epochs,
        default_root_dir=tmpdir,
        overfit_batches=2,
    )
    result = trainer.fit(model)
    assert result == 1
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

    class LoggingCallback(pl.Callback):

        def on_train_epoch_start(self, trainer, pl_module):
            self.len_outputs = 0

        def on_train_epoch_end(self, trainer, pl_module, outputs):
            self.len_outputs = len(outputs[0])

    class OverriddenModel(EvalModelTemplate):

        def on_train_epoch_start(self):
            self.num_train_batches = 0

        def training_epoch_end(self, outputs):  # Overridden
            return

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.num_train_batches += 1

    class NotOverriddenModel(EvalModelTemplate):

        def on_train_epoch_start(self):
            self.num_train_batches = 0

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.num_train_batches += 1

    overridden_model = OverriddenModel()
    not_overridden_model = NotOverriddenModel()

    callback = LoggingCallback()
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        overfit_batches=2,
        callbacks=[callback],
    )

    trainer.fit(overridden_model)
    # outputs from on_train_batch_end should be accessible in on_train_epoch_end hook
    # if training_epoch_end is overridden
    assert callback.len_outputs == overridden_model.num_train_batches

    trainer.fit(not_overridden_model)
    # outputs from on_train_batch_end should be empty
    assert callback.len_outputs == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_transfer_batch_hook():

    class CustomBatch:

        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestModel(EvalModelTemplate):

        hook_called = False

        def transfer_batch_to_device(self, data, device):
            self.hook_called = True
            if isinstance(data, CustomBatch):
                data.samples = data.samples.to(device)
                data.targets = data.targets.to(device)
            else:
                data = super().transfer_batch_to_device(data, device)
            return data

    model = CurrentTestModel()
    batch = CustomBatch((torch.zeros(5, 28), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    trainer.accelerator_backend = GPUAccelerator(trainer)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    trainer.get_model = MagicMock(return_value=model)
    batch_gpu = trainer.accelerator_backend.batch_to_device(batch, torch.device('cuda:0'))
    expected = torch.device('cuda', 0)
    assert model.hook_called
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
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


@pytest.mark.parametrize(
    'max_epochs,batch_idx_',
    [(2, 5), (3, 8), (4, 12)]
)
def test_on_train_batch_start_hook(max_epochs, batch_idx_):
    class CurrentModel(EvalModelTemplate):
        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            if batch_idx == batch_idx_:
                return -1

    model = CurrentModel()
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model)
    if batch_idx_ > len(model.val_dataloader()) - 1:
        assert trainer.batch_idx == len(model.val_dataloader()) - 1
        assert trainer.global_step == len(model.val_dataloader()) * max_epochs
    else:
        assert trainer.batch_idx == batch_idx_
        assert trainer.global_step == (batch_idx_ + 1) * max_epochs


def test_trainer_model_hook_system(tmpdir):
    """Test the hooks system."""

    class HookedModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.called = []

        def on_after_backward(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_after_backward()

        def on_before_zero_grad(self, optimizer):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_before_zero_grad(optimizer)

        def on_epoch_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_epoch_start()

        def on_epoch_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_epoch_end()

        def on_fit_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_fit_start()

        def on_fit_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_fit_end()

        def on_hpc_load(self, checkpoint):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_hpc_load(checkpoint)

        def on_hpc_save(self, checkpoint):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_hpc_save(checkpoint)

        def on_load_checkpoint(self, checkpoint):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_load_checkpoint(checkpoint)

        def on_save_checkpoint(self, checkpoint):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_save_checkpoint(checkpoint)

        def on_pretrain_routine_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_pretrain_routine_start()

        def on_pretrain_routine_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_pretrain_routine_end()

        def on_train_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_train_start()

        def on_train_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_train_end()

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_train_batch_start(batch, batch_idx, dataloader_idx)

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

        def on_train_epoch_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_train_epoch_start()

        def on_train_epoch_end(self, outputs):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_train_epoch_end(outputs)

        def on_validation_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_start()

        def on_validation_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_end()

        def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

        def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

        def on_validation_epoch_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_epoch_start()

        def on_validation_epoch_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_epoch_end()

        def on_test_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_start()

        def on_test_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_end()

        def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_batch_start(batch, batch_idx, dataloader_idx)

        def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

        def on_test_epoch_start(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_epoch_start()

        def on_test_epoch_end(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_epoch_end()

        def on_validation_model_eval(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_model_eval()

        def on_validation_model_train(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_validation_model_train()

        def on_test_model_eval(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_model_eval()

        def on_test_model_train(self):
            self.called.append(inspect.currentframe().f_code.co_name)
            super().on_test_model_train()

    model = HookedModel()

    assert model.called == []

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=2,
        limit_test_batches=1,
        progress_bar_refresh_rate=0,
    )

    assert model.called == []

    trainer.fit(model)

    expected = [
        'on_fit_start',
        'on_pretrain_routine_start',
        'on_pretrain_routine_end',
        'on_validation_model_eval',
        'on_validation_epoch_start',
        'on_validation_batch_start',
        'on_validation_batch_end',
        'on_validation_epoch_end',
        'on_validation_model_train',
        'on_train_start',
        'on_epoch_start',
        'on_train_epoch_start',
        'on_train_batch_start',
        'on_after_backward',
        'on_before_zero_grad',
        'on_train_batch_end',
        'on_train_batch_start',
        'on_after_backward',
        'on_before_zero_grad',
        'on_train_batch_end',
        'on_validation_model_eval',
        'on_validation_epoch_start',
        'on_validation_batch_start',
        'on_validation_batch_end',
        'on_validation_epoch_end',
        'on_save_checkpoint',
        'on_validation_model_train',
        'on_epoch_end',
        'on_train_epoch_end',
        'on_train_end',
        'on_fit_end',
    ]

    assert model.called == expected

    model2 = HookedModel()
    trainer.test(model2)

    expected = [
        'on_fit_start',
        'on_test_model_eval',
        'on_test_epoch_start',
        'on_test_batch_start',
        'on_test_batch_end',
        'on_test_epoch_end',
        'on_test_model_train',
        'on_fit_end',
    ]

    assert model2.called == expected
