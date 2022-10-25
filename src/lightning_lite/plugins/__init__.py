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
from lightning_lite.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_lite.plugins.io.checkpoint_io import CheckpointIO
from lightning_lite.plugins.io.torch_io import TorchCheckpointIO
from lightning_lite.plugins.io.xla import XLACheckpointIO
from lightning_lite.plugins.precision.deepspeed import DeepSpeedPrecision
from lightning_lite.plugins.precision.double import DoublePrecision
from lightning_lite.plugins.precision.native_amp import NativeMixedPrecision
from lightning_lite.plugins.precision.precision import Precision
from lightning_lite.plugins.precision.tpu import TPUPrecision
from lightning_lite.plugins.precision.tpu_bf16 import TPUBf16Precision

__all__ = [
    "ClusterEnvironment",
    "CheckpointIO",
    "TorchCheckpointIO",
    "XLACheckpointIO",
    "Precision",
    "DeepSpeedPrecision",
    "DoublePrecision",
    "NativeMixedPrecision",
    "TPUPrecision",
    "TPUBf16Precision",
]
