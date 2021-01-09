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
from typing import Union

from torch.optim import Optimizer

from pytorch_lightning.plugins.plugin import LightningPlugin


class PrecisionPlugin(LightningPlugin):
    """
    Abstract class to extend for precision support (32/16 etc).

    This is extended to cover any specific logic required for precision support such as AMP/APEX or sharded
    training.
    """

    def connect(self, model, optimizers):
        raise NotImplementedError

    def training_step(self, fx, args):
        raise NotImplementedError

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        raise NotImplementedError

    def clip_gradients(self, grad_clip_val: Union[int, float], optimizer: Optimizer, norm_type: float):
        raise NotImplementedError
