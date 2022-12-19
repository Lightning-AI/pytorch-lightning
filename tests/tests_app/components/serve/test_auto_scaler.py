import time
from unittest import mock
from unittest.mock import patch

import pytest

from lightning_app import CloudCompute, LightningWork
from lightning_app.components import AutoScaler, Text


class EmptyWork(LightningWork):
    def run(self):
        pass


class AutoScaler1(AutoScaler):
    def scale(self, replicas: int, metrics) -> int:
        # only upscale
        return replicas + 1


class AutoScaler2(AutoScaler):
    def scale(self, replicas: int, metrics) -> int:
        # only downscale
        return replicas - 1


def test_num_replicas_after_init():
    """Test the number of works is the same as min_replicas after initialization."""
    min_replicas = 2
    auto_scaler = AutoScaler(EmptyWork, min_replicas=min_replicas)
    assert auto_scaler.num_replicas == min_replicas


@patch("uvicorn.run")
@patch("lightning_app.components.serve.auto_scaler._LoadBalancer.url")
@patch("lightning_app.components.serve.auto_scaler.AutoScaler.num_pending_requests")
def test_num_replicas_not_above_max_replicas(*_):
    """Test self.num_replicas doesn't exceed max_replicas."""
    max_replicas = 6
    auto_scaler = AutoScaler1(
        EmptyWork,
        min_replicas=1,
        max_replicas=max_replicas,
        scale_out_interval=0.001,
        scale_in_interval=0.001,
    )

    for _ in range(max_replicas + 1):
        time.sleep(0.002)
        auto_scaler.run()

    assert auto_scaler.num_replicas == max_replicas


@patch("uvicorn.run")
@patch("lightning_app.components.serve.auto_scaler._LoadBalancer.url")
@patch("lightning_app.components.serve.auto_scaler.AutoScaler.num_pending_requests")
def test_num_replicas_not_belo_min_replicas(*_):
    """Test self.num_replicas doesn't exceed max_replicas."""
    min_replicas = 1
    auto_scaler = AutoScaler2(
        EmptyWork,
        min_replicas=min_replicas,
        max_replicas=4,
        scale_out_interval=0.001,
        scale_in_interval=0.001,
    )

    for _ in range(3):
        time.sleep(0.002)
        auto_scaler.run()

    assert auto_scaler.num_replicas == min_replicas


@pytest.mark.parametrize(
    "replicas, metrics, expected_replicas",
    [
        pytest.param(1, {"pending_requests": 1, "pending_works": 0}, 2, id="increase if no pending work"),
        pytest.param(1, {"pending_requests": 1, "pending_works": 1}, 1, id="dont increase if pending works"),
        pytest.param(8, {"pending_requests": 1, "pending_works": 0}, 7, id="reduce if requests < 25% capacity"),
        pytest.param(8, {"pending_requests": 2, "pending_works": 0}, 8, id="dont reduce if requests >= 25% capacity"),
    ],
)
def test_scale(replicas, metrics, expected_replicas):
    """Test `scale()`, the default scaling strategy."""
    auto_scaler = AutoScaler(
        EmptyWork,
        min_replicas=1,
        max_replicas=8,
        max_batch_size=1,
    )

    assert auto_scaler.scale(replicas, metrics) == expected_replicas


def test_scale_from_zero_min_replica():
    auto_scaler = AutoScaler(
        EmptyWork,
        min_replicas=0,
        max_replicas=2,
        max_batch_size=10,
    )

    resp = auto_scaler.scale(0, {"pending_requests": 0, "pending_works": 0})
    assert resp == 0

    resp = auto_scaler.scale(0, {"pending_requests": 1, "pending_works": 0})
    assert resp == 1

    resp = auto_scaler.scale(0, {"pending_requests": 1, "pending_works": 1})
    assert resp <= 0


def test_create_work_cloud_compute_cloned():
    """Test CloudCompute is cloned to avoid creating multiple works in a single machine."""
    cloud_compute = CloudCompute("gpu")
    auto_scaler = AutoScaler(EmptyWork, cloud_compute=cloud_compute)
    _ = auto_scaler.create_work()
    assert auto_scaler._work_kwargs["cloud_compute"] is not cloud_compute


fastapi_mock = mock.MagicMock()
mocked_fastapi_creater = mock.MagicMock(return_value=fastapi_mock)


@patch("lightning_app.components.serve.auto_scaler._create_fastapi", mocked_fastapi_creater)
@patch("lightning_app.components.serve.auto_scaler.uvicorn.run", mock.MagicMock())
def test_API_ACCESS_ENDPOINT_creation():
    auto_scaler = AutoScaler(EmptyWork, input_type=Text, output_type=Text)
    assert auto_scaler.load_balancer._work_name == "EmptyWork"

    auto_scaler.load_balancer.run()
    fastapi_mock.mount.assert_called_once_with("/endpoint-info", mock.ANY, name="static")


def test_autoscaler_scale_up(monkeypatch):
    monkeypatch.setattr(AutoScaler, "num_pending_works", 0)
    monkeypatch.setattr(AutoScaler, "num_pending_requests", 100)
    monkeypatch.setattr(AutoScaler, "scale", mock.MagicMock(return_value=1))
    monkeypatch.setattr(AutoScaler, "create_work", mock.MagicMock())
    monkeypatch.setattr(AutoScaler, "add_work", mock.MagicMock())

    auto_scaler = AutoScaler(EmptyWork, min_replicas=0, max_replicas=4, scale_out_interval=0.001)

    # Mocking the attributes
    auto_scaler._last_autoscale = time.time() - 100000
    auto_scaler.num_replicas = 0

    # triggering scale up
    auto_scaler.autoscale()
    auto_scaler.scale.assert_called_once()
    auto_scaler.create_work.assert_called_once()
    auto_scaler.add_work.assert_called_once()


def test_autoscaler_scale_down(monkeypatch):
    monkeypatch.setattr(AutoScaler, "num_pending_works", 0)
    monkeypatch.setattr(AutoScaler, "num_pending_requests", 0)
    monkeypatch.setattr(AutoScaler, "scale", mock.MagicMock(return_value=0))
    monkeypatch.setattr(AutoScaler, "remove_work", mock.MagicMock())
    monkeypatch.setattr(AutoScaler, "workers", mock.MagicMock())

    auto_scaler = AutoScaler(EmptyWork, min_replicas=0, max_replicas=4, scale_in_interval=0.001)

    # Mocking the attributes
    auto_scaler._last_autoscale = time.time() - 100000
    auto_scaler.num_replicas = 1
    auto_scaler.__dict__["load_balancer"] = mock.MagicMock()

    # triggering scale up
    auto_scaler.autoscale()
    auto_scaler.scale.assert_called_once()
    auto_scaler.remove_work.assert_called_once()
