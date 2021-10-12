from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.deepspeed_precision import DeepSpeedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.fully_sharded_native_amp import (  # noqa: F401
    FullyShardedNativeMixedPrecisionPlugin,
)
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.tpu_bfloat import TPUHalfPrecisionPlugin  # noqa: F401
