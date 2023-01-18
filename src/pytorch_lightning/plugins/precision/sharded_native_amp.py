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
from typing import Optional, Union

from typing_extensions import Literal

from pytorch_lightning.overrides.fairscale import _FAIRSCALE_AVAILABLE
from pytorch_lightning.plugins.precision.native_amp import MixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler
else:
    OSS = ShardedGradScaler = object


class ShardedNativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Native AMP for Sharded Training."""

    def __init__(
        self, precision: Literal["16", 16, "bf16"], device: str, scaler: Optional[ShardedGradScaler] = None
    ) -> None:
        rank_zero_deprecation(
            "PyTorch Lightning's sharded implementation using FairScale has been deprecated in v1.9.0 and will be"
            " removed in v2.0.0. You can try using the `Trainer(strategy='fsdp_native')` instead."
            " The difference is that native FSDP uses PyTorch's implementation and the current strategy uses"
            " FairScale's implementation (which was upstreamed to PyTorch). After removal, `strategy='fsdp'` will use"
            " the native version by default."
        )
        if not _FAIRSCALE_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for sharded AMP but you have not installed it."
                " Install `fairscale` using this guide: https://https://github.com/facebookresearch/fairscale"
            )
        super().__init__(
            precision, device, scaler=(ShardedGradScaler() if scaler is None and str(precision) == "16" else None)
        )

    def clip_grad_by_norm(self, optimizer: "OSS", clip_val: Union[int, float]) -> None:
        optimizer.clip_grad_norm(clip_val)
