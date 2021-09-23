from pytorch_lightning.plugins.training_type import TPUSpawnPlugin
from pytorch_lightning.accelerators import TPUAccelerator
from tests.helpers.runif import RunIf
from pytorch_lightning.plugins import SingleTPUPlugin


@RunIf(tpu=True)
def test_device_stats_tpu(tmpdir):
    """Test TPU get_device_stats."""
    plugin = SingleTPUPlugin(1)
    TPUAccel = TPUAccelerator(training_type_plugin=TPUSpawnPlugin(), precision_plugin=plugin)
    tpu_stats = TPUAccel.get_device_stats("1")
    fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]

    for f in fields:
        assert any(f in h for h in tpu_stats.keys())
