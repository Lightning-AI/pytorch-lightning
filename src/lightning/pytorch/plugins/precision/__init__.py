# Copyright The Lightning AI team.
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
from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin
from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin
from lightning.pytorch.plugins.precision.fsdp import FSDPMixedPrecisionPlugin, FSDPPrecisionPlugin
from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin
from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin

__all__ = [
    "BitsandbytesPrecisionPlugin",
    "DeepSpeedPrecisionPlugin",
    "DoublePrecisionPlugin",
    "FSDPMixedPrecisionPlugin",
    "FSDPPrecisionPlugin",
    "HalfPrecisionPlugin",
    "MixedPrecisionPlugin",
    "PrecisionPlugin",
    "TransformerEnginePrecisionPlugin",
    "XLAPrecisionPlugin",
]
