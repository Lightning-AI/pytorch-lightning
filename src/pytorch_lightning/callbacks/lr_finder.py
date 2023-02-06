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
r"""
LearningRateFinder
==================

Finds optimal learning rate
"""
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.tuner.lr_finder import _LRFinder, lr_find
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.utilities.seed import isolate_rng


class LearningRateFinder(Callback):
    """The ``LearningRateFinder`` callback enables the user to do a range test of good initial learning rates, to
    reduce the amount of guesswork in picking a good starting learning rate.

    Args:
        min_lr: Minimum learning rate to investigate

        max_lr: Maximum learning rate to investigate

        num_training_steps: Number of learning rates to test

        mode: Search strategy to update learning rate after each batch:

            - ``'exponential'`` (default): Increases the learning rate exponentially.
            - ``'linear'``: Increases the learning rate linearly.

        early_stop_threshold: Threshold for stopping the search. If the
            loss at any point is larger than early_stop_threshold*best_loss
            then the search is stopped. To disable, set to None.

        update_attr: Whether to update the learning rate attribute or not.

    Example::

        # Customize LearningRateFinder callback to run at different epochs.
        # This feature is useful while fine-tuning models.
        from pytorch_lightning.callbacks import LearningRateFinder


        class FineTuneLearningRateFinder(LearningRateFinder):
            def __init__(self, milestones, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.milestones = milestones

            def on_fit_start(self, *args, **kwargs):
                return

            def on_train_epoch_start(self, trainer, pl_module):
                if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                    self.lr_find(trainer, pl_module)


        trainer = Trainer(callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))])
        trainer.fit(...)

    Raises:
        MisconfigurationException:
            If learning rate/lr in ``model`` or ``model.hparams`` isn't overridden when ``auto_lr_find=True``,
            or if you are using more than one optimizer.
    """

    SUPPORTED_MODES = ("linear", "exponential")

    def __init__(
        self,
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training_steps: int = 100,
        mode: str = "exponential",
        early_stop_threshold: Optional[float] = 4.0,
        update_attr: bool = False,
    ) -> None:
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"`mode` should be either of {self.SUPPORTED_MODES}")

        self._min_lr = min_lr
        self._max_lr = max_lr
        self._num_training_steps = num_training_steps
        self._mode = mode
        self._early_stop_threshold = early_stop_threshold
        self._update_attr = update_attr

        self._early_exit = False
        self.lr_finder: Optional[_LRFinder] = None

    def lr_find(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with isolate_rng():
            self.optimal_lr = lr_find(
                trainer,
                pl_module,
                min_lr=self._min_lr,
                max_lr=self._max_lr,
                num_training=self._num_training_steps,
                mode=self._mode,
                early_stop_threshold=self._early_stop_threshold,
                update_attr=self._update_attr,
            )

        if self._early_exit:
            raise _TunerExitException()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.lr_find(trainer, pl_module)
