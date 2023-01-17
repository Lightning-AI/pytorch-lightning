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
from typing import Any

from pytorch_lightning.plugins.precision.sharded_native_amp import ShardedNativeMixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class FullyShardedNativeMixedPrecisionPlugin(ShardedNativeMixedPrecisionPlugin):
    """Native AMP for Fully Sharded Training."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            "PyTorch Lightning's sharded implementation using FairScale has been deprecated in v1.9.0 and will be"
            " removed in v2.0.0. You can try using the `Trainer(strategy='fsdp_native')` instead."
            " The difference is that native FSDP uses PyTorch's implementation and the current strategy uses"
            " FairScale's implementation (which was upstreamed to PyTorch). After removal, `strategy='fsdp'` will use"
            " the native version by default."
        )
        super().__init__(*args, **kwargs)

    def clip_grad_by_norm(self, *_: Any, **__: Any) -> None:
        # see https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html
        # section `Gradient Clipping`, using `torch.nn.utils.clip_grad_norm_` is incorrect
        # for FSDP module. To overcome this, needs to call sharded_module.clip_grad_norm(clip_val)
        # however we rely on LightningModule's configure_sharded_model to wrap FSDP, it would be hard to
        # trace back the root FSDP. Now we only support clip by value.
        raise MisconfigurationException(
            f"`gradient_clip_algorithm='norm'` is currently not supported for `{self.__class__.__name__}`"
        )
