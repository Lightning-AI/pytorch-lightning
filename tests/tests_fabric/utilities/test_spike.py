import contextlib
import sys

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.utilities.spike import _TORCHMETRICS_GREATER_EQUAL_1_0_0, SpikeDetection, TrainingSpikeException


def spike_detection_test(fabric, global_rank_spike, spike_value, should_raise):
    loss_vals = [1 / i for i in range(1, 10)]
    if fabric.global_rank == global_rank_spike:
        if spike_value is None:
            loss_vals[4] = 3
        else:
            spike_value = spike_value

    for i in range(len(loss_vals)):
        context = pytest.raises(TrainingSpikeException) if i == 4 and should_raise else contextlib.nullcontext()

        with context:
            fabric.call(
                "on_train_batch_end",
                fabric=fabric,
                loss=torch.tensor(loss_vals[i], device=fabric.device),
                batch=None,
                batch_idx=i,
            )


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
@pytest.mark.paramtetrize("spike_value", [None, float("inf"), float("NaN"), -float("inf")])
@pytest.mark.parametrze("finite_only", [True, False])
@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_fabric_spike_detection_integration(tmpdir, global_rank_spike, num_devices, spike_value, finite_only):
    fabric = Fabric(
        accelerator="cpu",
        devices=num_devices,
        callbacks=[SpikeDetection(exclude_batches_path=tmpdir, finite_only=finite_only)],
        strategy="ddp_spawn",
    )

    should_raise = spike_value is None or (spike_value is not None and finite_only)
    fabric.launch(
        spike_detection_test,
        global_rank_spike=global_rank_spike,
        spike_value=spike_value,
        should_raise=should_raise,
    )
