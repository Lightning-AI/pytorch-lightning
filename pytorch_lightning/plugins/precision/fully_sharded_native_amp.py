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

from torch.nn import Module
from torch.optim import Optimizer

from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.utilities import GradClipAlgorithmType


class FullyShardedNativeMixedPrecisionPlugin(ShardedNativeMixedPrecisionPlugin):
    """Mixed Precision for Full Sharded Training"""

    precision = "mixed"

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.VALUE,
        model: Optional[Module] = None,
    ) -> None:
        clip_val = float(clip_val)
        if clip_val <= 0:
            return
        # see https://fairscale.readthedocs.io/en/latest/api/nn/fsdp_tips.html
        # section `Gradient Clipping`, using `torch.nn.utils.clip_grad_norm_` is incorrect
        # for FSDP module. To overcome this, needs to call sharded_module.clip_grad_norm(clip_val)
        # however we rely on LightningModule's configure_sharded_model to wrap FSDP, it would be hard to
        # trace back the root FSDP. Now we only support clip by value.
        assert (
            gradient_clip_algorithm == GradClipAlgorithmType.VALUE
        ), "`gradient_clip_algorithm`: `norm` is currently not supported for `FullyShardedNativeMixedPrecisionPlugin`"
        self.clip_grad_by_value(optimizer, clip_val)
