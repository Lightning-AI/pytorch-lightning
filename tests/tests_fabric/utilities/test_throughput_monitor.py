from unittest import mock
from unittest.mock import Mock, call

import torch
from lightning.fabric import Fabric
from lightning.fabric.utilities.throughput_monitor import ThroughputMonitor, measure_flops

from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel


@RunIf(min_torch="2.1")
def test_measure_flops():
    with torch.device("meta"):
        model = BoringModel()
        x = torch.randn(2, 32)
    model_fwd = lambda: model(x)
    model_loss = lambda y: y.sum()

    training_flops = measure_flops(model, model_fwd, model_loss)
    assert isinstance(training_flops, int)

    eval_flops = measure_flops(model.eval(), model_fwd, model_loss)
    assert isinstance(eval_flops, int)
    assert eval_flops < training_flops


def test_throughput_monitor():
    logger_mock = Mock()

    # simulate lit-gpt style script
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput_monitor._get_flops_available", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=3, time_unit="seconds")
    flops = 10
    total_lengths = 0
    total_t0 = 0.0  # fake times
    micro_batch_size = 3
    for iter_num in range(1, 6):
        # forward + backward + step + zero_grad ...
        t1 = iter_num + 0.5
        total_lengths += 2
        monitor.on_train_batch_end(
            iter_num * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            flops_per_batch=flops,
            lengths=total_lengths,
        )
        if iter_num % 2 == 0:
            # validation
            monitor.eval_end(0.2)

    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"time/train": 1.5, "time/val": 0.0, "time/total": 1.5, "samples": 3}, step=0),
        call(metrics={"time/train": 2.5, "time/val": 0.0, "time/total": 2.5, "samples": 6}, step=1),
        call(metrics={"time/train": 3.5, "time/val": 0.2, "time/total": 3.7, "samples": 9}, step=2),
        call(
            metrics={
                "batches_per_sec": 1.0,
                "samples_per_sec": 3.0,
                "device/batches_per_sec": 1.0,
                "device/samples_per_sec": 3.0,
                "items_per_sec": 6.0,
                "device/items_per_sec": 6.0,
                "flops_per_sec": 10.0,
                "device/flops_per_sec": 10.0,
                "device/mfu": 0.1,
                "time/train": 4.5,
                "time/val": 0.2,
                "time/total": 4.7,
                "samples": 12,
            },
            step=3,
        ),
        call(
            metrics={
                "batches_per_sec": 1.0,
                "samples_per_sec": 3.0,
                "device/batches_per_sec": 1.0,
                "device/samples_per_sec": 3.0,
                "items_per_sec": 6.0,
                "device/items_per_sec": 6.0,
                "flops_per_sec": 10.0,
                "device/flops_per_sec": 10.0,
                "device/mfu": 0.1,
                "time/train": 5.5,
                "time/val": 0.4,
                "time/total": 5.9,
                "samples": 15,
            },
            step=4,
        ),
    ]
