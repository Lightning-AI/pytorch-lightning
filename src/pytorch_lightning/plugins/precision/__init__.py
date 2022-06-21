# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.fully_sharded_native_amp import (  # noqa: F401
    FullyShardedNativeMixedPrecisionPlugin,
)
from pytorch_lightning.plugins.precision.hpu import HPUPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.ipu import IPUPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.tpu import TPUPrecisionPlugin  # noqa: F401
from pytorch_lightning.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin  # noqa: F401
