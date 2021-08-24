from pytorch_lightning.plugins.base_plugin import Plugin
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.plugins.plugins_registry import (  # noqa: F401
    call_training_type_register_plugins,
    TrainingTypePluginsRegistry,
)
from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.cpu_native_amp import CPUNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.deepspeed_precision import DeepSpeedPrecisionPlugin
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin
from pytorch_lightning.plugins.precision.fully_sharded_native_amp import FullyShardedNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.ipu_precision import IPUPrecisionPlugin
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.tpu_bfloat import TPUHalfPrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.plugins.training_type.ddp2 import DDP2Plugin
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.plugins.training_type.deepspeed import DeepSpeedPlugin
from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin
from pytorch_lightning.plugins.training_type.fully_sharded import DDPFullyShardedPlugin
from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
from pytorch_lightning.plugins.training_type.ipu import IPUPlugin
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.plugins.training_type.sharded import DDPShardedPlugin
from pytorch_lightning.plugins.training_type.sharded_spawn import DDPSpawnShardedPlugin
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin

__all__ = [
    "CheckpointIO",
    "TorchCheckpointIO",
    "ApexMixedPrecisionPlugin",
    "CPUNativeMixedPrecisionPlugin",
    "DataParallelPlugin",
    "DDP2Plugin",
    "DDPPlugin",
    "DDPSpawnPlugin",
    "DDPFullyShardedPlugin",
    "DeepSpeedPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "HorovodPlugin",
    "IPUPlugin",
    "IPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "SingleDevicePlugin",
    "SingleTPUPlugin",
    "TPUHalfPrecisionPlugin",
    "TPUSpawnPlugin",
    "TrainingTypePlugin",
    "ParallelPlugin",
    "Plugin",
    "DDPShardedPlugin",
    "DDPSpawnShardedPlugin",
]

from pathlib import Path

FILE_ROOT = Path(__file__).parent
TRAINING_TYPE_BASE_MODULE = "pytorch_lightning.plugins.training_type"

call_training_type_register_plugins(FILE_ROOT, TRAINING_TYPE_BASE_MODULE)
