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

"""Decorator for LightningModule methods."""

from functools import wraps
from typing import Callable

from pytorch_lightning.core.lightning import LightningModule


def auto_move_data(fn: Callable) -> Callable:
    """
    Decorator for :class:`~pytorch_lightning.core.lightning.LightningModule` methods for which
    input arguments should be moved automatically to the correct device.
    It as no effect if applied to a method of an object that is not an instance of
    :class:`~pytorch_lightning.core.lightning.LightningModule` and is typically applied to ``__call__``
    or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device
            the parameters are on.

    Example:

        .. code-block:: python

            # directly in the source code
            class LitModel(LightningModule):

                @auto_move_data
                def forward(self, x):
                    return x

            # or outside
            LitModel.forward = auto_move_data(LitModel.forward)

            model = LitModel()
            model = model.to('cuda')
            model(torch.zeros(1, 3))

            # input gets moved to device
            # tensor([[0., 0., 0.]], device='cuda:0')

    """
    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args = self.transfer_batch_to_device(args, self.device)
        kwargs = self.transfer_batch_to_device(kwargs, self.device)
        return fn(self, *args, **kwargs)

    return auto_transfer_args
