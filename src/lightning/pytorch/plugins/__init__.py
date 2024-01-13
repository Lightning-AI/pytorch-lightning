from typing import Union

from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO, XLACheckpointIO
from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
from lightning.pytorch.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from lightning.pytorch.plugins.precision.amp import MixedPrecision
from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecision
from lightning.pytorch.plugins.precision.double import DoublePrecision
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
from lightning.pytorch.plugins.precision.half import HalfPrecision
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecision
from lightning.pytorch.plugins.precision.xla import XLAPrecision

_PLUGIN_INPUT = Union[Precision, ClusterEnvironment, CheckpointIO, LayerSync]

__all__ = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "BitsandbytesPrecision",
    "DeepSpeedPrecision",
    "DoublePrecision",
    "HalfPrecision",
    "MixedPrecision",
    "Precision",
    "TransformerEnginePrecision",
    "FSDPPrecision",
    "XLAPrecision",
    "LayerSync",
    "TorchSyncBatchNorm",
]
