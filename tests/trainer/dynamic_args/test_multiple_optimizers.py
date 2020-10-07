from pytorch_lightning import Trainer
from tests.base.boring_model import BoringModel
import torch
import os


os.environ['PL_DEV_DEBUG'] = '1'

def test_multiple_optimizers(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def on_train_epoch_start(self) -> None:
            self.opt_0_seen = False
            self.opt_1_seen = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            if optimizer_idx == 0:
                self.opt_0_seen = True
            elif optimizer_idx == 1:
                self.opt_1_seen = True
            else:
                raise Exception('should only have two optimizers')

            self.training_step_called = True
            loss = self.step(batch[0])
            return loss

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        row_log_interval=1,
        weights_summary=None,
    )

    trainer.fit(model)
    assert model.opt_0_seen
    assert model.opt_1_seen
