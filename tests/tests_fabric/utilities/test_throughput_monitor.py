from unittest import mock
from unittest.mock import Mock, call

import pytest
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins import Precision
from lightning.fabric.utilities.throughput_monitor import (
    Throughput,
    ThroughputMonitor,
    _get_flops_available,
    _MonotonicWindow,
    measure_flops,
)

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


def test_flops_available(xla_available):
    with mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 PCIe"):
        flops = _get_flops_available(torch.device("cuda"), torch.bfloat16)
    assert flops == 1.513e15 / 2

    with pytest.warns(match="not found for 'CocoNut"), mock.patch("torch.cuda.get_device_name", return_value="CocoNut"):
        assert _get_flops_available(torch.device("cuda"), torch.bfloat16) is None

    with pytest.warns(match="t4' does not support torch.bfloat"), mock.patch(
        "torch.cuda.get_device_name", return_value="t4"
    ):
        assert _get_flops_available(torch.device("cuda"), torch.bfloat16) is None

    from torch_xla.experimental import tpu

    assert isinstance(tpu, Mock)
    tpu.get_tpu_env.return_value = {"TYPE": "V4"}

    flops = _get_flops_available(torch.device("xla"), torch.bfloat16)
    assert flops == 275e12

    tpu.get_tpu_env.return_value = {"TYPE": "V1"}
    with pytest.warns(match="not found for TPU 'V1'"):
        assert _get_flops_available(torch.device("xla"), torch.bfloat16) is None

    tpu.reset_mock()


def test_throughput():
    # just samples
    tracker = Throughput()
    metrics = tracker.compute(time=2.0, samples=2)
    assert metrics == {"time": 2.0, "samples": 2}

    # different lengths and samples
    with pytest.raises(RuntimeError, match="same number of samples"):
        tracker.compute(time=2.1, samples=3, lengths=4)

    # lengths and samples
    tracker = Throughput(window_size=1)
    metrics = tracker.compute(time=2, samples=2, lengths=4)
    assert metrics == {"time": 2.0, "samples": 2}
    metrics = tracker.compute(time=2.5, samples=4, lengths=8)
    assert metrics == {
        "time": 2.5,
        "samples": 4,
        "device/batches_per_sec": 2.0,
        "device/samples_per_sec": 4.0,
        "device/items_per_sec": 16.0,
    }

    with pytest.raises(ValueError, match="Expected the value to increase"):
        tracker.compute(time=2.5, samples=2, lengths=4)

    # FIXME test flops


def mock_train_loop(monitor):
    # simulate lit-gpt style loop
    total_lengths = 0
    total_t0 = 0.0  # fake times
    micro_batch_size = 3
    for iter_num in range(1, 6):
        # forward + backward + step + zero_grad ...
        t1 = iter_num + 0.5
        total_lengths += 2
        monitor.compute_and_log(
            time=t1 - total_t0,
            samples=iter_num * micro_batch_size,
            flops_per_batch=10,
            lengths=total_lengths,
        )


def test_throughput_monitor():
    logger_mock = Mock()
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput_monitor._get_flops_available", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=3, separator="|")
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


def test_throughput_monitor_manual_step():
    fabric_mock = Mock()
    fabric_mock.world_size = 1
    fabric_mock.strategy.precision = Precision()
    monitor = ThroughputMonitor(fabric_mock)
    assert monitor.step == -1
    monitor.compute_and_log(time=0.5, samples=3)
    assert monitor.step == 0
    monitor.compute_and_log(time=1.5, samples=4, step=5)
    assert monitor.step == 5
    assert fabric_mock.log_dict.mock_calls == [
        call(metrics={"time": 0.5, "samples": 3}, step=0),
        call(metrics={"time": 1.5, "samples": 4}, step=5),
    ]


def test_throughput_monitor_world_size():
    logger_mock = Mock()
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput_monitor._get_flops_available", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=3)
        # simulate that there are 2 devices
        monitor._throughput.world_size = 2
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


def test_monotonic_window():
    w = _MonotonicWindow(maxlen=3)
    assert w == []
    assert len(w) == 0
    w.append(1)
    w.append(2)
    w.append(3)
    assert w == [1, 2, 3]
    assert len(w) == 3
    assert w[1] == 2
    assert w[-2:] == [2, 3]

    with pytest.raises(NotImplementedError):
        w[1] = 123
    with pytest.raises(NotImplementedError):
        w[1:2] = [1, 2]

    with pytest.raises(ValueError, match="Expected the value to increase"):
        w.append(2)
    w.clear()
    w.append(2)
