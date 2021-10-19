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
from pytorch_lightning.plugins.precision.bf16 import Bf16PrecisionPlugin
from pytorch_lightning.plugins.precision.mixin import ShardedMixin
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim.grad_scaler import ShardedGradScaler


class ShardedNativeMixedPrecisionPlugin(NativeMixedPrecisionPlugin, ShardedMixin):
    """Sharded support with native AMP."""

    def __init__(self) -> None:
        super().__init__()
        if not _FAIRSCALE_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for sharded AMP but you have not installed it."
                " Install `fairscale` using this guide: https://https://github.com/facebookresearch/fairscale"
            )
        self.scaler = ShardedGradScaler()


class ShardedBf16PrecisionPlugin(Bf16PrecisionPlugin, ShardedMixin):
    """Sharded support with bfloat16."""

    def __init__(self, use_cpu: bool) -> None:
        super().__init__(use_cpu)
        if not _FAIRSCALE_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for sharded AMP but you have not installed it."
                " Install `fairscale` using this guide: https://https://github.com/facebookresearch/fairscale"
            )
