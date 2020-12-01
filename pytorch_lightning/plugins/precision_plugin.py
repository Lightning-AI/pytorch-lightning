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
import abc
from typing import Optional, Union

from torch.optim import Optimizer

from pytorch_lightning.utilities import AMPType


class PrecisionPlugin(abc.ABC):
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

    def required_plugins(self, trainer, amp_backend: AMPType) -> Optional[list]:
        """
            Override to define additional required plugins. This is useful for when custom plugins
            need to enforce override of other plugins.

        Returns: Optional list of plugins containing additional plugins.

        Example::
            class MyPlugin(DDPPlugin):
                def required_plugins(self):
                    return [MyCustomAMPPlugin()]

            # Will automatically add the necessary AMP plugin
            trainer = Trainer(plugins=[MyPlugin()])

            # Crash as MyPlugin enforces custom AMP plugin
            trainer = Trainer(plugins=[MyPlugin(), NativeAMPPlugin()])

        """
