# Copyright The Lightning AI team.
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
r"""Lambda Callback ^^^^^^^^^^^^^^^

Create a simple callback on the fly using lambda functions.
"""

from __future__ import annotations

from typing import Callable

from lightning.pytorch.callbacks.callback import Callback


class LambdaCallback(Callback):
    r"""Create a simple callback on the fly using lambda functions.

    Args:
        **kwargs: hooks supported by :class:`~lightning.pytorch.callbacks.callback.Callback`

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import LambdaCallback
        >>> trainer = Trainer(callbacks=[LambdaCallback(setup=lambda *args: print('setup'))])
    """

    def __init__(
        self,
        setup: Callable | None = None,
        teardown: Callable | None = None,
        on_fit_start: Callable | None = None,
        on_fit_end: Callable | None = None,
        on_sanity_check_start: Callable | None = None,
        on_sanity_check_end: Callable | None = None,
        on_train_batch_start: Callable | None = None,
        on_train_batch_end: Callable | None = None,
        on_train_epoch_start: Callable | None = None,
        on_train_epoch_end: Callable | None = None,
        on_validation_epoch_start: Callable | None = None,
        on_validation_epoch_end: Callable | None = None,
        on_test_epoch_start: Callable | None = None,
        on_test_epoch_end: Callable | None = None,
        on_validation_batch_start: Callable | None = None,
        on_validation_batch_end: Callable | None = None,
        on_test_batch_start: Callable | None = None,
        on_test_batch_end: Callable | None = None,
        on_train_start: Callable | None = None,
        on_train_end: Callable | None = None,
        on_validation_start: Callable | None = None,
        on_validation_end: Callable | None = None,
        on_test_start: Callable | None = None,
        on_test_end: Callable | None = None,
        on_exception: Callable | None = None,
        on_save_checkpoint: Callable | None = None,
        on_load_checkpoint: Callable | None = None,
        on_before_backward: Callable | None = None,
        on_after_backward: Callable | None = None,
        on_before_optimizer_step: Callable | None = None,
        on_before_zero_grad: Callable | None = None,
        on_predict_start: Callable | None = None,
        on_predict_end: Callable | None = None,
        on_predict_batch_start: Callable | None = None,
        on_predict_batch_end: Callable | None = None,
        on_predict_epoch_start: Callable | None = None,
        on_predict_epoch_end: Callable | None = None,
    ):
        for k, v in locals().items():
            if k == "self":
                continue
            if v is not None:
                setattr(self, k, v)
