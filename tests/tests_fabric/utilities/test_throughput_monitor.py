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

    fwd_flops = measure_flops(model, model_fwd)
    assert isinstance(fwd_flops, int)

    fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)
    assert isinstance(fwd_and_bwd_flops, int)
    assert fwd_flops < fwd_and_bwd_flops


def mock_train_loop(monitor):
    # simulate lit-gpt style loop
    total_lengths = 0
    total_t0 = 0.0  # fake times
    micro_batch_size = 3
    for iter_num in range(1, 6):
        # forward + backward + step + zero_grad ...
        t1 = iter_num + 0.5
        total_lengths += 2
        monitor.compute(
            iter_num * micro_batch_size,
            t1 - total_t0,
            flops_per_batch=10,
            lengths=total_lengths,
        )


def test_throughput_monitor():
    logger_mock = Mock()
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput_monitor._get_flops_available", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=3, time_unit="seconds", separator="|")
    mock_train_loop(monitor)
    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"time": 1.5, "samples": 3}, step=0),
        call(metrics={"time": 2.5, "samples": 6}, step=1),
        call(metrics={"time": 3.5, "samples": 9}, step=2),
        call(
            metrics={
                "device|batches_per_sec": 1.0,
                "device|samples_per_sec": 3.0,
                "device|items_per_sec": 6.0,
                "device|flops_per_sec": 10.0,
                "device|mfu": 0.1,
                "time": 4.5,
                "samples": 12,
            },
            step=3,
        ),
        call(
            metrics={
                "device|batches_per_sec": 1.0,
                "device|samples_per_sec": 3.0,
                "device|items_per_sec": 6.0,
                "device|flops_per_sec": 10.0,
                "device|mfu": 0.1,
                "time": 5.5,
                "samples": 15,
            },
            step=4,
        ),
    ]


def test_throughput_monitor_world_size():
    logger_mock = Mock()
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput_monitor._get_flops_available", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=3, time_unit="seconds")
        # simulate that there are 2 devices
        monitor._monitor.world_size = 2
    mock_train_loop(monitor)
    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"time": 1.5, "samples": 3}, step=0),
        call(metrics={"time": 2.5, "samples": 6}, step=1),
        call(metrics={"time": 3.5, "samples": 9}, step=2),
        call(
            metrics={
                "time": 4.5,
                "samples": 12,
                "device/batches_per_sec": 1.0,
                "device/samples_per_sec": 3.0,
                "batches_per_sec": 2.0,
                "samples_per_sec": 6.0,
                "items_per_sec": 4.0,
                "device/items_per_sec": 6.0,
                "flops_per_sec": 20.0,
                "device/flops_per_sec": 10.0,
                "device/mfu": 0.1,
            },
            step=3,
        ),
        call(
            metrics={
                "time": 5.5,
                "samples": 15,
                "device/batches_per_sec": 1.0,
                "device/samples_per_sec": 3.0,
                "batches_per_sec": 2.0,
                "samples_per_sec": 6.0,
                "items_per_sec": 4.0,
                "device/items_per_sec": 6.0,
                "flops_per_sec": 20.0,
                "device/flops_per_sec": 10.0,
                "device/mfu": 0.1,
            },
            step=4,
        ),
    ]
