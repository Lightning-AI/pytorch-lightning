import contextlib
import sys
from functools import partial

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.utilities.spike import _TORCHMETRICS_GREATER_EQUAL_1_0_0, SpikeDetection, TrainingSpikeException


def spike_detection_test(fabric, global_rank_spike):
    loss_vals = [1 / i for i in range(1, 10)]
    if fabric.global_rank == global_rank_spike:
        loss_vals[4] = 3

    for i in range(len(loss_vals)):
        context = pytest.raises(TrainingSpikeException) if i == 4 else contextlib.nullcontext()

        with context:
            fabric.call(
                "on_train_batch_end",
                fabric,
                torch.tensor(loss_vals[i], device=fabric.device),
                None,
                i,
            )


@pytest.mark.parametrize(
    ("global_rank_spike", "num_devices"),
    [
        pytest.param(0, 1),
        pytest.param(
            0,
            2,
            marks=pytest.mark.skipif(sys.platform != "linux", reason="ddp-tests on OS but linux are slow and unstable"),
        ),
    ],
)
@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_fabric_spike_detection_integration(tmpdir, global_rank_spike, num_devices):
    fabric = Fabric(
        accelerator="cpu",
        devices=num_devices,
        callbacks=[SpikeDetection(exclude_batches_path=tmpdir)],
        strategy="ddp_spawn",
    )
    fabric.launch(partial(spike_detection_test, global_rank_spike=global_rank_spike))
