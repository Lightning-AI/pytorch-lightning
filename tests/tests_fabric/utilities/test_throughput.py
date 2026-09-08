import warnings
from unittest import mock
from unittest.mock import Mock, call

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.plugins import Precision
from lightning.fabric.utilities.throughput import (
    Throughput,
    ThroughputMonitor,
    _MonotonicWindow,
    get_available_flops,
    get_float32_matmul_precision_compat,
    measure_flops,
)
from tests_fabric.test_fabric import BoringModel


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


def test_get_available_flops(xla_available):
    with mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 PCIe"):
        flops = get_available_flops(torch.device("cuda"), torch.bfloat16)
    assert flops == 756e12

    with pytest.warns(match="not found for 'CocoNut"), mock.patch("torch.cuda.get_device_name", return_value="CocoNut"):
        assert get_available_flops(torch.device("cuda"), torch.bfloat16) is None

    with (
        pytest.warns(match="t4' does not support torch.bfloat"),
        mock.patch("torch.cuda.get_device_name", return_value="t4"),
    ):
        assert get_available_flops(torch.device("cuda"), torch.bfloat16) is None

    from torch_xla.experimental import tpu

    assert isinstance(tpu, Mock)

    tpu.get_tpu_env.return_value = {"TYPE": "V4"}
    flops = get_available_flops(torch.device("xla"), torch.bfloat16)
    assert flops == 275e12

    tpu.get_tpu_env.return_value = {"TYPE": "V1"}
    with pytest.warns(match="not found for TPU 'V1'"):
        assert get_available_flops(torch.device("xla"), torch.bfloat16) is None

    tpu.get_tpu_env.return_value = {"ACCELERATOR_TYPE": "v3-8"}
    flops = get_available_flops(torch.device("xla"), torch.bfloat16)
    assert flops == 123e12

    tpu.reset_mock()


@pytest.mark.parametrize(
    "device_name",
    [
        # Hopper
        "NVIDIA H200 SXM1",
        "NVIDIA H200 NVL1",
        "h100-nvl",  # TODO: switch with `torch.cuda.get_device_name()` result
        "h100-hbm3",  # TODO: switch with `torch.cuda.get_device_name()` result
        "NVIDIA H100 PCIe",
        "h100-hbm2e",  # TODO: switch with `torch.cuda.get_device_name()` result
        # Ada
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "Tesla L40",
        "NVIDIA L4",
        # Ampere
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 3080 Ti",
        "NVIDIA GeForce RTX 3070",
        pytest.param("NVIDIA GeForce RTX 3070 Ti", marks=pytest.mark.xfail(raises=AssertionError)),
        pytest.param("NVIDIA GeForce RTX 3060", marks=pytest.mark.xfail(raises=AssertionError)),
        pytest.param("NVIDIA GeForce RTX 3060 Ti", marks=pytest.mark.xfail(raises=AssertionError)),
        pytest.param("NVIDIA GeForce RTX 3050", marks=pytest.mark.xfail(raises=AssertionError)),
        pytest.param("NVIDIA GeForce RTX 3050 Ti", marks=pytest.mark.xfail(raises=AssertionError)),
        "NVIDIA A6000",
        "NVIDIA A40",
        "NVIDIA A10G",
        # Turing
        "NVIDIA GeForce RTX 2080 SUPER",
        "NVIDIA GeForce RTX 2080 Ti",
        "NVIDIA GeForce RTX 2080",
        "NVIDIA GeForce RTX 2070 Super",
        "Quadro RTX 5000 with Max-Q Design",
        "Tesla T4",
        "TITAN RTX",
        # Volta
        "Tesla V100-SXm2-32GB",
        "Tesla V100-PCIE-32GB",
        "Tesla V100S-PCIE-32GB",
    ],
)
@mock.patch("lightning.fabric.accelerators.cuda._is_ampere_or_later", return_value=False)
def test_get_available_flops_cuda_mapping_exists(_, device_name):
    """Tests `get_available_flops` against known device names."""
    with mock.patch("lightning.fabric.utilities.throughput.torch.cuda.get_device_name", return_value=device_name):
        assert get_available_flops(device=torch.device("cuda"), dtype=torch.float32) is not None


def test_throughput():
    # required args only
    throughput = Throughput()
    throughput.update(time=2.0, batches=1, samples=2)
    assert throughput.compute() == {"time": 2.0, "batches": 1, "samples": 2}

    # different lengths and samples
    with pytest.raises(RuntimeError, match="same number of samples"):
        throughput.update(time=2.1, batches=2, samples=3, lengths=4)

    # lengths and samples
    throughput = Throughput(window_size=2)
    throughput.update(time=2, batches=1, samples=2, lengths=4)
    throughput.update(time=2.5, batches=2, samples=4, lengths=8)
    assert throughput.compute() == {
        "time": 2.5,
        "batches": 2,
        "samples": 4,
        "lengths": 8,
        "device/batches_per_sec": 2.0,
        "device/samples_per_sec": 4.0,
        "device/items_per_sec": 8.0,
    }

    with pytest.raises(ValueError, match="Expected the value to increase"):
        throughput.update(time=2.5, batches=3, samples=2, lengths=4)

    # flops
    throughput = Throughput(available_flops=50, window_size=2)
    throughput.update(time=1, batches=1, samples=2, flops=10, lengths=10)
    throughput.update(time=2, batches=2, samples=4, flops=10, lengths=20)
    assert throughput.compute() == {
        "time": 2,
        "batches": 2,
        "samples": 4,
        "lengths": 20,
        "device/batches_per_sec": 1.0,
        "device/flops_per_sec": 10.0,
        "device/items_per_sec": 10.0,
        "device/mfu": 0.2,
        "device/samples_per_sec": 2.0,
    }

    # flops without available
    throughput.available_flops = None
    throughput.reset()
    throughput.update(time=1, batches=1, samples=2, flops=10, lengths=10)
    throughput.update(time=2, batches=2, samples=4, flops=10, lengths=20)
    assert throughput.compute() == {
        "time": 2,
        "batches": 2,
        "samples": 4,
        "lengths": 20,
        "device/batches_per_sec": 1.0,
        "device/flops_per_sec": 10.0,
        "device/items_per_sec": 10.0,
        "device/samples_per_sec": 2.0,
    }

    throughput = Throughput(window_size=2)
    with pytest.raises(ValueError, match=r"samples.*to be greater or equal than batches"):
        throughput.update(time=0, batches=2, samples=1)
    throughput = Throughput(window_size=2)
    with pytest.raises(ValueError, match=r"lengths.*to be greater or equal than samples"):
        throughput.update(time=0, batches=2, samples=2, lengths=1)


def test_throughput_sparse_model_scaling():
    """``using_sparse_model`` scales ``available_flops`` by the acceleration factor."""
    # explicit False — available_flops untouched (also avoids the unset-flag warning)
    assert Throughput(available_flops=100.0, using_sparse_model=False).available_flops == 100.0

    # sparse flag on with default 2.0 factor
    assert Throughput(available_flops=100.0, using_sparse_model=True).available_flops == 200.0

    # sparse flag on with custom factor
    throughput = Throughput(available_flops=100.0, using_sparse_model=True, sparse_cuda_acceleration_factor=4.0)
    assert throughput.available_flops == 400.0

    # sparse flag on with unknown peak — stays None, no error (MFU is skipped in compute())
    assert Throughput(available_flops=None, using_sparse_model=True).available_flops is None

    # factor below 1.0 is physically meaningless (sparsity can never lower the peak)
    with pytest.raises(AssertionError, match="sparse acceleration factor cannot reduce"):
        Throughput(available_flops=100.0, using_sparse_model=True, sparse_cuda_acceleration_factor=0.5)


def test_throughput_sparse_model_warning():
    """When ``using_sparse_model`` is unset and a peak is known, warn so MFU is not silently ambiguous."""
    # warning fires only when the peak is known and the user didn't specify intent
    with pytest.warns(UserWarning, match="MFU assumes dense model FLOPs"):
        throughput = Throughput(available_flops=100.0)
    assert throughput.available_flops == 100.0  # dense default, no scaling applied

    # explicit choice (True or False) silences the warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Throughput(available_flops=100.0, using_sparse_model=False)
        Throughput(available_flops=100.0, using_sparse_model=True)

    # no peak known → MFU is never computed, so no warning is needed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Throughput(available_flops=None)


def test_throughput_sparse_model_mfu():
    """MFU denominator reflects the sparse-scaled peak, so MFU halves when the peak doubles."""
    # baseline dense: mfu = flops_per_sec / available_flops = 10 / 50 = 0.2
    throughput = Throughput(available_flops=50, window_size=2, using_sparse_model=False)
    throughput.update(time=1, batches=1, samples=2, flops=10)
    throughput.update(time=2, batches=2, samples=4, flops=10)
    assert throughput.compute()["device/mfu"] == 0.2

    # sparse: peak doubles to 100, so mfu = 10 / 100 = 0.1
    throughput = Throughput(available_flops=50, window_size=2, using_sparse_model=True)
    throughput.update(time=1, batches=1, samples=2, flops=10)
    throughput.update(time=2, batches=2, samples=4, flops=10)
    assert throughput.compute()["device/mfu"] == 0.1


def mock_train_loop(monitor):
    # simulate lit-gpt style loop
    total_lengths = 0
    total_t0 = 0.0  # fake times
    micro_batch_size = 3
    for iter_num in range(1, 6):
        # forward + backward + step + zero_grad ...
        t1 = iter_num + 0.5
        total_lengths += 3 * 2
        monitor.update(
            time=t1 - total_t0,
            batches=iter_num,
            samples=iter_num * micro_batch_size,
            lengths=total_lengths,
            flops=10,
        )
        monitor.compute_and_log()


def test_throughput_monitor():
    logger_mock = Mock()
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput.get_available_flops", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=4, separator="|")
    mock_train_loop(monitor)
    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"time": 1.5, "batches": 1, "samples": 3, "lengths": 6}, step=0),
        call(metrics={"time": 2.5, "batches": 2, "samples": 6, "lengths": 12}, step=1),
        call(metrics={"time": 3.5, "batches": 3, "samples": 9, "lengths": 18}, step=2),
        call(
            metrics={
                "time": 4.5,
                "batches": 4,
                "samples": 12,
                "lengths": 24,
                "device|batches_per_sec": 1.0,
                "device|samples_per_sec": 3.0,
                "device|items_per_sec": 6.0,
                "device|flops_per_sec": 10.0,
                "device|mfu": 0.1,
            },
            step=3,
        ),
        call(
            metrics={
                "time": 5.5,
                "batches": 5,
                "samples": 15,
                "lengths": 30,
                "device|batches_per_sec": 1.0,
                "device|samples_per_sec": 3.0,
                "device|items_per_sec": 6.0,
                "device|flops_per_sec": 10.0,
                "device|mfu": 0.1,
            },
            step=4,
        ),
    ]


def test_throughput_monitor_step():
    fabric_mock = Mock()
    fabric_mock.world_size = 1
    fabric_mock.strategy.precision = Precision()
    monitor = ThroughputMonitor(fabric_mock)

    # automatic step increase
    assert monitor.step == -1
    monitor.update(time=0.5, batches=1, samples=3)
    metrics = monitor.compute_and_log()
    assert metrics == {"time": 0.5, "batches": 1, "samples": 3}
    assert monitor.step == 0

    # manual step
    monitor.update(time=1.5, batches=2, samples=4)
    metrics = monitor.compute_and_log(step=5)
    assert metrics == {"time": 1.5, "batches": 2, "samples": 4}
    assert monitor.step == 5
    assert fabric_mock.log_dict.mock_calls == [
        call(metrics={"time": 0.5, "batches": 1, "samples": 3}, step=0),
        call(metrics={"time": 1.5, "batches": 2, "samples": 4}, step=5),
    ]


def test_throughput_monitor_world_size():
    logger_mock = Mock()
    fabric = Fabric(devices=1, loggers=logger_mock)
    with mock.patch("lightning.fabric.utilities.throughput.get_available_flops", return_value=100):
        monitor = ThroughputMonitor(fabric, window_size=4)
        # simulate that there are 2 devices
        monitor.world_size = 2
    mock_train_loop(monitor)
    assert logger_mock.log_metrics.mock_calls == [
        call(metrics={"time": 1.5, "batches": 1, "samples": 3, "lengths": 6}, step=0),
        call(metrics={"time": 2.5, "batches": 2, "samples": 6, "lengths": 12}, step=1),
        call(metrics={"time": 3.5, "batches": 3, "samples": 9, "lengths": 18}, step=2),
        call(
            metrics={
                "time": 4.5,
                "batches": 4,
                "samples": 12,
                "lengths": 24,
                "device/batches_per_sec": 1.0,
                "device/samples_per_sec": 3.0,
                "batches_per_sec": 2.0,
                "samples_per_sec": 6.0,
                "items_per_sec": 12.0,
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
                "batches": 5,
                "samples": 15,
                "lengths": 30,
                "device/batches_per_sec": 1.0,
                "device/samples_per_sec": 3.0,
                "batches_per_sec": 2.0,
                "samples_per_sec": 6.0,
                "items_per_sec": 12.0,
                "device/items_per_sec": 6.0,
                "flops_per_sec": 20.0,
                "device/flops_per_sec": 10.0,
                "device/mfu": 0.1,
            },
            step=4,
        ),
    ]


def test_throughput_monitor_sparse_model():
    """``using_sparse_model`` and ``sparse_cuda_acceleration_factor`` propagate through ThroughputMonitor."""
    fabric_mock = Mock()
    fabric_mock.world_size = 1
    fabric_mock.strategy.precision = Precision()
    with mock.patch("lightning.fabric.utilities.throughput.get_available_flops", return_value=100):
        monitor = ThroughputMonitor(fabric_mock, using_sparse_model=True, sparse_cuda_acceleration_factor=2.0)
    # 100 (from hardware lookup) scaled by 2.0 (sparse factor) → 200
    assert monitor.available_flops == 200


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


def test_get_float32_matmul_precision_compat():
    """Test that the compatibility function works without warnings."""
    precision = get_float32_matmul_precision_compat()
    assert precision in ["highest", "high", "medium"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        precision = get_float32_matmul_precision_compat()

        deprecation_warnings = [
            warning
            for warning in w
            if "Please use the new API settings to control TF32 behavior" in str(warning.message)
        ]
        assert len(deprecation_warnings) == 0, (
            f"Compatibility function triggered {len(deprecation_warnings)} deprecation warnings"
        )
    assert precision in ["highest", "high", "medium"]
