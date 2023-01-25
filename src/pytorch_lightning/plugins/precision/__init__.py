# Copyright The Lightning team.
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
from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.colossalai import ColossalAIPrecisionPlugin
from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from pytorch_lightning.plugins.precision.double import DoublePrecisionPlugin
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.fully_sharded_native_amp import FullyShardedNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.hpu import HPUPrecisionPlugin
from pytorch_lightning.plugins.precision.ipu import IPUPrecisionPlugin
from pytorch_lightning.plugins.precision.native_amp import MixedPrecisionPlugin, NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.plugins.precision.tpu import TPUPrecisionPlugin
from pytorch_lightning.plugins.precision.tpu_bf16 import TPUBf16PrecisionPlugin

__all__ = [
    "ApexMixedPrecisionPlugin",
    "ColossalAIPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "FullyShardedNativeNativeMixedPrecisionPlugin",
    "FullyShardedNativeMixedPrecisionPlugin",
    "HPUPrecisionPlugin",
    "IPUPrecisionPlugin",
    "NativeMixedPrecisionPlugin",
    "MixedPrecisionPlugin",
    "PrecisionPlugin",
    "ShardedNativeMixedPrecisionPlugin",
    "TPUPrecisionPlugin",
    "TPUBf16PrecisionPlugin",
]
