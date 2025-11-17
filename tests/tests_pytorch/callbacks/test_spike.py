import contextlib

import pytest
import torch

from lightning.fabric.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from lightning.fabric.utilities.spike import TrainingSpikeException
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.spike import SpikeDetection
from tests_pytorch.helpers.runif import RunIf, _xfail_gloo_windows


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

        curr_loss_val = 1 / (batch_idx + 1)
        if self.trainer.global_rank == self.spike_global_rank and batch_idx == 4:
            curr_loss_val = self.spike_value

        if curr_loss_val is None:
            curr_loss_val = batch_idx

        return self.layer(torch.tensor(curr_loss_val, device=self.device, dtype=self.dtype).view(1, 1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class MyTrainerSpikeDetection(SpikeDetection):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        context = (
            pytest.raises(TrainingSpikeException) if batch_idx == 4 and self.should_raise else contextlib.nullcontext()
        )

        with context:
            if batch_idx == 4:
                print(outputs)
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    ("global_rank_spike", "num_devices", "spike_value", "finite_only"),
    # NOTE FOR ALL FOLLOWING TESTS:
    # adding run on linux only because multiprocessing on other platforms takes forever
    [
        pytest.param(0, 1, None, True, marks=_xfail_gloo_windows),
        pytest.param(0, 1, None, False, marks=_xfail_gloo_windows),
        pytest.param(0, 1, float("inf"), True, marks=_xfail_gloo_windows),
        pytest.param(0, 1, float("inf"), False, marks=_xfail_gloo_windows),
        pytest.param(0, 1, float("-inf"), True, marks=_xfail_gloo_windows),
        pytest.param(0, 1, float("-inf"), False, marks=_xfail_gloo_windows),
        pytest.param(0, 1, float("NaN"), True, marks=_xfail_gloo_windows),
        pytest.param(0, 1, float("NaN"), False, marks=_xfail_gloo_windows),
        pytest.param(0, 2, None, True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, None, False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, None, True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, None, False, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("-inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("-inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("-inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("-inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("NaN"), True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("NaN"), False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("NaN"), True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("NaN"), False, marks=RunIf(linux_only=True)),
    ],
)
@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_trainer_spike_detection_integration(tmp_path, global_rank_spike, num_devices, spike_value, finite_only):
    cb = MyTrainerSpikeDetection(exclude_batches_path=tmp_path, finite_only=finite_only)
    # spike_value == None -> typical spike detection
    # finite_only -> typical spike detection and raise with NaN +/- inf
    # if inf -> inf >> other values -> typical spike detection
    cb.should_raise = spike_value is None or finite_only or spike_value == float("inf")

    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=False,
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
