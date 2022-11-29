import time
from unittest.mock import patch

from lightning_app import LightningWork
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
