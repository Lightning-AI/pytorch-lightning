from unittest.mock import MagicMock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.gpu_backend import GPUBackend
from tests.base import EvalModelTemplate


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
    trainer.accelerator_backend = GPUBackend(trainer)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    trainer.get_model = MagicMock(return_value=model)
    batch_gpu = trainer.accelerator_backend.batch_to_device(batch, torch.device('cuda:0'))
    expected = torch.device('cuda', 0)
    assert model.hook_called
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected


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
        assert trainer.global_step == len(model.val_dataloader()) * max_epochs - 1
    else:
        assert trainer.batch_idx == batch_idx_
        assert trainer.global_step == (batch_idx_ + 1) * max_epochs - 1
