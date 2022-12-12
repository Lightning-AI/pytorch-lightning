import time
from unittest.mock import patch

import pytest

from lightning_app import CloudCompute, LightningWork
from lightning_app.components import AutoScaler


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
@patch("lightning_app.components.auto_scaler._LoadBalancer.url")
@patch("lightning_app.components.auto_scaler.AutoScaler.num_pending_requests")
def test_num_replicas_not_above_max_replicas(*_):
    """Test self.num_replicas doesn't exceed max_replicas."""
    max_replicas = 6
    auto_scaler = AutoScaler1(
        EmptyWork,
        min_replicas=1,
        max_replicas=max_replicas,
        autoscale_interval=0.001,
    )

    for _ in range(max_replicas + 1):
        time.sleep(0.002)
        auto_scaler.run()

    assert auto_scaler.num_replicas == max_replicas


@patch("uvicorn.run")
@patch("lightning_app.components.auto_scaler._LoadBalancer.url")
@patch("lightning_app.components.auto_scaler.AutoScaler.num_pending_requests")
def test_num_replicas_not_belo_min_replicas(*_):
    """Test self.num_replicas doesn't exceed max_replicas."""
    min_replicas = 1
    auto_scaler = AutoScaler2(
        EmptyWork,
        min_replicas=min_replicas,
        max_replicas=4,
        autoscale_interval=0.001,
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


def test_create_work_cloud_compute_cloned():
    """Test CloudCompute is cloned to avoid creating multiple works in a single machine."""
    cloud_compute = CloudCompute("gpu")
    auto_scaler = AutoScaler(EmptyWork, cloud_compute=cloud_compute)
    _ = auto_scaler.create_work()
    assert auto_scaler._work_kwargs["cloud_compute"] is not cloud_compute
