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
from lightning_pytorch.plugins.precision.amp import MixedPrecision
from lightning_pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision
from lightning_pytorch.plugins.precision.deepspeed import DeepSpeedPrecision
from lightning_pytorch.plugins.precision.double import DoublePrecision
from lightning_pytorch.plugins.precision.fsdp import FSDPPrecision
from lightning_pytorch.plugins.precision.half import HalfPrecision
from lightning_pytorch.plugins.precision.precision import Precision
from lightning_pytorch.plugins.precision.transformer_engine import TransformerEnginePrecision
from lightning_pytorch.plugins.precision.xla import XLAPrecision

__all__ = [
    "BitsandbytesPrecision",
    "DeepSpeedPrecision",
    "DoublePrecision",
    "FSDPPrecision",
    "HalfPrecision",
    "MixedPrecision",
    "Precision",
    "TransformerEnginePrecision",
    "XLAPrecision",
]
