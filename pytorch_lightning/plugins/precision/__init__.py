from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.bf16 import Bf16PrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.deepspeed_precision import DeepSpeedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.fully_sharded import (  # noqa: F401
    FullyShardedBf16PrecisionPlugin,
    FullyShardedNativeMixedPrecisionPlugin,
)
from pytorch_lightning.plugins.precision.ipu_precision import IPUPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.sharded import (  # noqa: F401
    ShardedBf16PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
)
from pytorch_lightning.plugins.precision.tpu import TPUPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin  # noqa: F401
