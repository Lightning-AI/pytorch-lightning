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
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.plugins.io.xla import XLACheckpointIO
from lightning.fabric.plugins.precision.amp import MixedPrecision
from lightning.fabric.plugins.precision.bitsandbytes import BitsandbytesPrecision
from lightning.fabric.plugins.precision.deepspeed import DeepSpeedPrecision
from lightning.fabric.plugins.precision.double import DoublePrecision
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.plugins.precision.half import HalfPrecision
from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.transformer_engine import TransformerEnginePrecision
from lightning.fabric.plugins.precision.xla import XLAPrecision

__all__ = [
    "ClusterEnvironment",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "BitsandbytesPrecision",
    "Precision",
    "DeepSpeedPrecision",
    "DoublePrecision",
    "HalfPrecision",
    "MixedPrecision",
    "TransformerEnginePrecision",
    "XLAPrecision",
    "FSDPPrecision",
]
