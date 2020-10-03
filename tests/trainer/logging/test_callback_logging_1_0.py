import torch
from tests.base import BoringModel
from pytorch_lightning import Trainer, Callback


def test_callback_logging_steps(tmpdir):

    class TestCallback(Callback):

        def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            assert pl_module._current_fx_name == 'training_step'
            pl_module.log('a1', torch.tensor(1))
            pl_module.log(f'b', torch.tensor(2), on_step=True, on_epoch=True)
            pl_module.log(f'b2', torch.tensor(2), on_step=False, on_epoch=True)

        def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            assert pl_module._current_fx_name == 'evaluation_step'
            pl_module.log('a2', torch.tensor(1))
            pl_module.log(f'c', torch.tensor(2), on_step=True, on_epoch=True)

        def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            assert pl_module._current_fx_name == 'evaluation_step'
            pl_module.log('a3', torch.tensor(1))
            pl_module.log(f'd', torch.tensor(2), on_step=True, on_epoch=True)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.train_epoch_calls = 0
            self.val_epoch_calls = 0

        def on_train_epoch_start(self) -> None:
            self.train_epoch_calls += 1

        def on_validation_epoch_start(self) -> None:
            if not self.trainer.running_sanity_check:
                self.val_epoch_calls += 1

    model = TestModel()
    trainer = Trainer(
        callbacks=[TestCallback()],
        row_log_interval=1,
        max_epochs=1,
        val_check_interval=1.0,
    )
    trainer.fit(model)

    expected_metrics = {
        'epoch',
        'a1',
        'b', 'step_b', 'epoch_b',
        'b2',
        'a2',
        'c', 'step_c/epoch_0', 'epoch_c',
    }
    generated_metrics = set(trainer.logged_metrics.keys())
    assert expected_metrics == generated_metrics

    trainer.test()

    expected_metrics.update({
        'step_d/epoch_o', 'epoch_d',
    })
    generated_metrics = set(trainer.logged_metrics.keys())
    assert expected_metrics == generated_metrics
