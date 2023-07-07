import contextlib
import sys

import pytest
import torch

from lightning.fabric.utilities.spike import _TORCHMETRICS_GREATER_EQUAL_1_0_0, TrainingSpikeException
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.spike import SpikeDetection


class IdentityModule(LightningModule):
    def __init__(self, spike_global_rank: int, spike_value):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1, bias=False)
        self.spike_global_rank = spike_global_rank
        self.spike_value = spike_value

    def training_step(self, batch, batch_idx: int):
        # initialize it all to weights one so that input = output but with gradients
        with torch.no_grad():
            self.layer.weight.data = torch.ones_like(self.layer.weight.data)

        if batch_idx == 4:
            curr_loss_val = 3 if self.spike_value is None else self.spike_value
        curr_loss_val = 3 if batch_idx == 4 else 1 / (batch_idx + 1)

        loss = self.layer(torch.tensor(curr_loss_val, device=self.device, dtype=self.dtype).view(1, 1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class MyTrainerSpikeDetection(SpikeDetection):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        context = (
            pytest.raises(TrainingSpikeException) if batch_idx == 4 and self.should_raise else contextlib.nullcontext()
        )

        with context:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


@pytest.mark.parametrize(
    ("global_rank_spike", "num_devices"),
    [
        pytest.param(0, 1),
        pytest.param(
            0,
            2,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="multiprocessing on other platforms takes forever"
            ),
        ),
    ],
)
@pytest.mark.parametrize("spike_value", [None, float("inf"), float("NaN"), -float("inf")])
@pytest.mark.parametrize("finite_only", [True, False])
@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_trainer_spike_detection_integration(tmp_path, global_rank_spike, num_devices, spike_value, finite_only):
    cb = MyTrainerSpikeDetection(exclude_batches_path=tmp_path, finite_only=finite_only)
    cb.should_raise = spike_value is None or (spike_value is not None and finite_only)

    trainer = Trainer(
        callbacks=[cb],
        accelerator="cpu",
        devices=num_devices,
        max_epochs=1,
        strategy="ddp_spawn",
    )
    trainer.fit(
        IdentityModule(global_rank_spike, spike_value),
        torch.utils.data.DataLoader([1 for _ in range(10)]),
    )
