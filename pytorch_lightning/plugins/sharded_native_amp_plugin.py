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
from typing import Union, cast

from torch.optim import Optimizer

from pytorch_lightning.plugins.native_amp import NativeAMPPlugin
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE, _NATIVE_AMP_AVAILABLE

if _NATIVE_AMP_AVAILABLE and _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


class ShardedNativeAMPPlugin(NativeAMPPlugin):
    @property
    def scaler(self):
        return ShardedGradScaler()

    def clip_gradients(self,
                       optimizer: Optimizer,
                       grad_clip_val: Union[int, float],
                       gradient_clip_algorithm: str,
                       norm_type: Union[float, int]):
        if gradient_clip_algorithm == 'value':
            raise NotImplementedError("Value grad clipping with sharded ddp is not implemented yet")
        elif gradient_clip_algorithm.startswith('norm'):
            optimizer = cast(OSS, optimizer)
            optimizer.clip_grad_norm(grad_clip_val, norm_type=norm_type)
        else:
            raise ValueError(f'gradient_clip_algorithm [{gradient_clip_algorithm}] is not valid.')
