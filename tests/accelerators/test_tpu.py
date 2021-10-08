from pytorch_lightning.accelerators import TPUAccelerator
from pytorch_lightning.plugins import PrecisionPlugin, SingleTPUPlugin
from tests.helpers.runif import RunIf


@RunIf(tpu=True)
def test_device_stats_tpu(tmpdir):
    """Test TPU get_device_stats."""
    TPUAccel = TPUAccelerator(training_type_plugin=SingleTPUPlugin(1), precision_plugin=PrecisionPlugin())
    tpu_stats = TPUAccel.get_device_stats("xla:1")

    fields = ["avg. free memory (MB)", "avg. peak memory (MB)"]

    for f in fields:
        assert any(f in h for h in tpu_stats.keys())
