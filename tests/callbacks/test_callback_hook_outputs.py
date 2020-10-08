from pytorch_lightning import Trainer, Callback
from tests.base.boring_model import BoringModel


def test_train_step_no_return(tmpdir):
    """
    Tests that only training_step can be used
    """
    class CB(Callback):

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            d = outputs[0][0]
            assert 'minimize' in d

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            assert 'x' in outputs

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            assert 'x' in outputs

        def on_train_epoch_end(self, trainer, pl_module, outputs):
            d = outputs[0]
            assert len(d) == trainer.num_training_batches

    class TestModel(BoringModel):
        def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
            d = outputs[0][0]
            assert 'minimize' in d

        def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
            assert 'x' in outputs

        def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
            assert 'x' in outputs

        def on_train_epoch_end(self, outputs) -> None:
            d = outputs[0]
            assert len(d) == self.trainer.num_training_batches

    model = TestModel()

    trainer = Trainer(
        callbacks=[CB()],
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)
