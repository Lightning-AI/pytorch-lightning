from typing import Union

from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO, XLACheckpointIO
from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
from lightning.pytorch.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin
from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin
from lightning.pytorch.plugins.precision.fsdp import FSDPMixedPrecisionPlugin, FSDPPrecisionPlugin
from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin
from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin

PLUGIN = Union[PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync]
PLUGIN_INPUT = Union[PLUGIN, str]

__all__ = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "BitsandbytesPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "HalfPrecisionPlugin",
    "MixedPrecisionPlugin",
    "PrecisionPlugin",
    "TransformerEnginePrecisionPlugin",
    "FSDPMixedPrecisionPlugin",
    "FSDPPrecisionPlugin",
    "XLAPrecisionPlugin",
    "LayerSync",
    "TorchSyncBatchNorm",
]
