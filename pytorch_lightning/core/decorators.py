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

from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn


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

    Example::

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
        from pytorch_lightning.core.lightning import LightningModule
        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args, kwargs = self.transfer_batch_to_device((args, kwargs))
        return fn(self, *args, **kwargs)

    rank_zero_deprecation(
        "The `@auto_move_data` decorator is deprecated in v1.3 and will be removed in v1.5."
        f" Please use `trainer.predict` instead for inference. The decorator was applied to `{fn.__name__}`"
    )

    return auto_transfer_args


def parameter_validation(fn: Callable) -> Callable:
    """
    Decorator for :meth:`~pytorch_lightning.core.LightningModule.to` method.
    Validates that the module parameter lengths match after moving to the device. It is useful
    when tying weights on TPU's.

    Args:
        fn: ``.to`` method

    Note:
        TPU's require weights to be tied/shared after moving the module to the device.
        Failure to do this results in the initialization of new weights which are not tied.
        To overcome this issue, weights should be tied using the ``on_post_move_to_device`` model hook
        which is called after the module has been moved to the device.

    See Also:
        - `XLA Documentation <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks>`_
    """

    @wraps(fn)
    def inner_fn(self, *args, **kwargs):
        pre_layer_count = len(list(self.parameters()))
        module = fn(self, *args, **kwargs)
        self.on_post_move_to_device()
        post_layer_count = len(list(self.parameters()))

        if not pre_layer_count == post_layer_count:
            rank_zero_warn(
                f'The model layers do not match after moving to the target device.'
                ' If your model employs weight sharing on TPU,'
                ' please tie your weights using the `on_post_move_to_device` model hook.\n'
                f'Layer count: [Before: {pre_layer_count} After: {post_layer_count}]'
            )

        return module

    return inner_fn
