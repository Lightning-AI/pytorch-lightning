from pytorch_lightning.utilities.distributed import rank_zero_deprecation

rank_zero_deprecation(
    "Using ``import pytorch_lightning.profiler.profilers`` is depreceated in v1.4, and will be removed in v1.6. "
    "HINT: Use ``import pytorch_lightning.profiler`` directly."
)

from pytorch_lightning.profiler.advanced import AdvancedProfiler  # noqa E402
from pytorch_lightning.profiler.base import AbstractProfiler, BaseProfiler, PassThroughProfiler  # noqa E402
from pytorch_lightning.profiler.pytorch import PyTorchProfiler  # noqa E402
from pytorch_lightning.profiler.simple import SimpleProfiler  # noqa E402

__all__ = [
    'AbstractProfiler',
    'BaseProfiler',
    'AdvancedProfiler',
    'PassThroughProfiler',
    'PyTorchProfiler',
    'SimpleProfiler',
]
