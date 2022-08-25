from typing import Union

from lightning_lite.lite.plugins.environments import ClusterEnvironment
from lightning_lite.lite.plugins.io.async_plugin import AsyncCheckpointIO
from lightning_lite.lite.plugins.io.checkpoint_plugin import CheckpointIO
from lightning_lite.lite.plugins.io.hpu_plugin import HPUCheckpointIO
from lightning_lite.lite.plugins.io.torch_plugin import TorchCheckpointIO
from lightning_lite.lite.plugins.io.xla_plugin import XLACheckpointIO
from lightning_lite.lite.plugins.layer_sync import LayerSync, NativeSyncBatchNorm
from lightning_lite.lite.plugins.precision.apex_amp import ApexMixedPrecisionPlugin
from lightning_lite.lite.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning_lite.lite.plugins.precision.double import DoublePrecisionPlugin
from lightning_lite.lite.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from lightning_lite.lite.plugins.precision.fully_sharded_native_amp import FullyShardedNativeMixedPrecisionPlugin
from lightning_lite.lite.plugins.precision.hpu import HPUPrecisionPlugin
from lightning_lite.lite.plugins.precision.ipu import IPUPrecisionPlugin
from lightning_lite.lite.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from lightning_lite.lite.plugins.precision.precision_plugin import PrecisionPlugin
from lightning_lite.lite.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from lightning_lite.lite.plugins.precision.tpu import TPUPrecisionPlugin
from lightning_lite.lite.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin
from lightning_lite.lite.strategies import Strategy

PLUGIN = Union[Strategy, PrecisionPlugin, ClusterEnvironment, CheckpointIO, LayerSync]
PLUGIN_INPUT = Union[PLUGIN, str]

__all__ = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "HPUCheckpointIO",
    "ApexMixedPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "IPUPrecisionPlugin",
    "HPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeNativeMixedPrecisionPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
    "LayerSync",
    "NativeSyncBatchNorm",
]
