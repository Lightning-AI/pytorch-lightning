import contextlib

import pytest
import torch

from lightning.pytorch import LightningModule, Trainer
from lightning.fabric.utilities.spike import _TORCHMETRICS_GREATER_EQUAL_1_0_0, TrainingSpikeException
from lightning.pytorch.callbacks.spike import SpikeDetection


class IdentityModule(LightningModule):
    def __init__(self, spike_global_rank: int):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1, bias=False)
        self.spike_global_rank = spike_global_rank

    def training_step(self, batch, batch_idx: int):
        # initialize it all to weights one so that input = output but with gradients
        with torch.no_grad():
            self.layer.weight.data = torch.ones_like(self.layer.weight.data)

        curr_loss_val = 3 if batch_idx == 4 else 1 / (batch_idx + 1)

        loss = self.layer(torch.tensor(curr_loss_val, device=self.device, dtype=self.dtype).view(1, 1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class MyTrainerSpikeDetection(SpikeDetection):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        context = pytest.raises(TrainingSpikeException) if batch_idx == 4 else contextlib.nullcontext()

        with context:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


@pytest.mark.parametrize(
    ("global_rank_spike", "num_devices"),
    [pytest.param(0, 1), pytest.param(0, 2), pytest.param(0, 1)],
)
@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_trainer_spike_detection_integration(tmpdir, global_rank_spike, num_devices):
    trainer = Trainer(
        callbacks=[MyTrainerSpikeDetection()],
        accelerator="cpu",
        devices=num_devices,
        max_epochs=1,
        strategy="ddp_spawn",
    )
    trainer.fit(
        IdentityModule(global_rank_spike),
        torch.utils.data.DataLoader([1 for _ in range(10)]),
    )
