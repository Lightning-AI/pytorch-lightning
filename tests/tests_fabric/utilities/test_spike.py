import contextlib

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from lightning.fabric.utilities.spike import SpikeDetection, TrainingSpikeException
from tests_fabric.helpers.runif import RunIf


def spike_detection_test(fabric, global_rank_spike, spike_value, should_raise):
    loss_vals = [1 / i for i in range(1, 10)]
    if fabric.global_rank == global_rank_spike:
        if spike_value is None:
            loss_vals[4] = 3
        else:
            loss_vals[4] = spike_value

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


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    ("global_rank_spike", "num_devices", "spike_value", "finite_only"),
    # NOTE FOR ALL FOLLOWING TESTS:
    # adding run on linux only because multiprocessing on other platforms takes forever
    [
        pytest.param(0, 1, None, True),
        pytest.param(0, 1, None, False),
        pytest.param(0, 1, float("inf"), True),
        pytest.param(0, 1, float("inf"), False),
        pytest.param(0, 1, float("-inf"), True),
        pytest.param(0, 1, float("-inf"), False),
        pytest.param(0, 1, float("NaN"), True),
        pytest.param(0, 1, float("NaN"), False),
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
def test_fabric_spike_detection_integration(tmp_path, global_rank_spike, num_devices, spike_value, finite_only):
    fabric = Fabric(
        accelerator="cpu",
        devices=num_devices,
        callbacks=[SpikeDetection(exclude_batches_path=tmp_path, finite_only=finite_only)],
        strategy="ddp_spawn",
    )

    # spike_value == None -> typical spike detection
    # finite_only -> typical spike detection and raise with NaN +/- inf
    # if inf -> inf >> other values -> typical spike detection
    should_raise = spike_value is None or finite_only or spike_value == float("inf")
    fabric.launch(
        spike_detection_test,
        global_rank_spike=global_rank_spike,
        spike_value=spike_value,
        should_raise=should_raise,
    )
