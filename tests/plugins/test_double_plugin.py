import torch

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


class DoublePrecisionBoringModel(BoringModel):

    def training_step(self, batch, batch_idx):
        output = self((batch, torch.ones_like(batch).long()))  # Add some non floating-point data
        loss = self.loss(batch, output)
        return {"loss": loss}

    def on_fit_start(self):
        assert self.layer.weight.dtype == torch.float64

    def forward(self, x):
        try:
            x, ones = x  # training
            assert ones.dtype == torch.long
        except ValueError:
            pass  # test / val
        assert x.dtype == torch.float64
        return super().forward(x)

    def on_after_backward(self):
        assert self.layer.weight.grad.dtype == torch.float64


def test_double_precision(tmpdir):
    model = DoublePrecisionBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        precision=64,
        log_every_n_steps=1,
    )
    trainer.fit(model)
