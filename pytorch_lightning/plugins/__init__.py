from pytorch_lightning.plugins.base_plugin import Plugin  # noqa: F401
from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.deepspeed_precision import DeepSpeedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.tpu_bfloat import TPUHalfPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ddp2 import DDP2Plugin  # noqa: F401
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.deepspeed import DeepSpeedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.rpc import RPCPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.rpc_sequential import RPCSequentialPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.sharded import DDPShardedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.sharded_spawn import DDPSpawnShardedPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin  # noqa: F401
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin  # noqa: F401

__all__ = [
    "ApexMixedPrecisionPlugin",
    "DataParallelPlugin",
    "DDP2Plugin",
    "DDPPlugin",
    "DDPSpawnPlugin",
    "DeepSpeedPlugin",
    "DeepSpeedPrecisionPlugin",
    "HorovodPlugin",
    "NativeMixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "SingleDevicePlugin",
    "SingleTPUPlugin",
    "TPUHalfPrecisionPlugin",
    "TPUSpawnPlugin",
    'RPCPlugin',
    'RPCSequentialPlugin',
    'TrainingTypePlugin',
    'ParallelPlugin',
    'Plugin',
    'DDPShardedPlugin',
    'DDPSpawnShardedPlugin',
]
