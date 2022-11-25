from lightning_app.components import AutoScaler


class SimpleAutoScaler(AutoScaler):
    def scale(self, replicas: int, metrics) -> int:
        return replicas + 1


def target_fn():
    auto_scaler = SimpleAutoScaler()
    auto_scaler.run()


def test_num_replicas_not_above_max_replicas():
    """Test self.num_replicas doesn't exceed max_replicas."""


def test_num_replicas_not_below_min_replicas():
    """Test num_replicas doesn't go below min_replicas."""
