from typing import Union

from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO, XLACheckpointIO
from pytorch_lightning.plugins.io.async_plugin import AsyncCheckpointIO
from pytorch_lightning.plugins.io.hpu_plugin import HPUCheckpointIO
from pytorch_lightning.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from pytorch_lightning.plugins.precision.amp import MixedPrecisionPlugin
from pytorch_lightning.plugins.precision.colossalai import ColossalAIPrecisionPlugin
from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin
from pytorch_lightning.plugins.precision.fsdp import FSDPMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.hpu import HPUPrecisionPlugin
from pytorch_lightning.plugins.precision.ipu import IPUPrecisionPlugin
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.plugins.precision.tpu import TPUPrecisionPlugin
from pytorch_lightning.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin

PLUGIN = Union[PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync]
PLUGIN_INPUT = Union[PLUGIN, str]

__all__ = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "HPUCheckpointIO",
    "ColossalAIPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "IPUPrecisionPlugin",
    "HPUPrecisionPlugin",
    "MixedPrecisionPlugin",
    "PrecisionPlugin",
    "FSDPMixedPrecisionPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
    "LayerSync",
    "TorchSyncBatchNorm",
]
