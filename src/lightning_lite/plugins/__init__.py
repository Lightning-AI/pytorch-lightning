from typing import Union

from lightning_lite.plugins.environments import ClusterEnvironment
from lightning_lite.plugins.io.checkpoint_plugin import CheckpointIO
from lightning_lite.plugins.io.torch_plugin import TorchCheckpointIO
from lightning_lite.plugins.io.xla_plugin import XLACheckpointIO
from lightning_lite.plugins.layer_sync import LayerSync, NativeSyncBatchNorm
from lightning_lite.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning_lite.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from lightning_lite.plugins.precision.precision import PrecisionPlugin
from lightning_lite.plugins.precision.tpu import TPUPrecisionPlugin
from lightning_lite.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin

PLUGIN = Union[PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync]
PLUGIN_INPUT = Union[PLUGIN, str]

__all__ = [
    "CheckpointIO",
    "DeepSpeedPrecisionPlugin",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "NativeMixedPrecisionPlugin",
    "PrecisionPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
    "LayerSync",
    "NativeSyncBatchNorm",
]
