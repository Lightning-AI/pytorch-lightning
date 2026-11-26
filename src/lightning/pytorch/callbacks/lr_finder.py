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

from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.tuner.lr_finder import _lr_find, _LRFinder
from lightning.pytorch.utilities.exceptions import _TunerExitException
from lightning.pytorch.utilities.seed import isolate_rng


class LearningRateFinder(Callback):
    """The ``LearningRateFinder`` callback enables the user to do a range test of good initial learning rates, to
    reduce the amount of guesswork in picking a good starting learning rate.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

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
        attr_name: Name of the attribute which stores the learning rate. The names 'learning_rate' or 'lr' get
            automatically detected. Otherwise, set the name here.
        weights_only: Defaults to ``None``. If ``True``, restricts loading to ``state_dicts`` of plain
            ``torch.Tensor`` and other primitive types. If loading a checkpoint from a trusted source that contains
            an ``nn.Module``, use ``weights_only=False``. If loading checkpoint from an untrusted source, we
            recommend using ``weights_only=True``. For more information, please refer to the
            `PyTorch Developer Notes on Serialization Semantics <https://docs.pytorch.org/docs/main/notes/serialization.html#id3>`_.

    Example::

        # Customize LearningRateFinder callback to run at different epochs.
        # This feature is useful while fine-tuning models.
        from lightning.pytorch.callbacks import LearningRateFinder


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
            If learning rate/lr in ``model`` or ``model.hparams`` isn't overridden, or if you are using more than
            one optimizer.

    """

    SUPPORTED_MODES = ("linear", "exponential")

    def __init__(
        self,
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training_steps: int = 100,
        mode: str = "exponential",
        early_stop_threshold: Optional[float] = 4.0,
        update_attr: bool = True,
        attr_name: str = "",
        weights_only: Optional[bool] = None,
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
        self._attr_name = attr_name
        self._weights_only = weights_only

        self._early_exit = False
        self._lr_ignored = False
        self.optimal_lr: Optional[_LRFinder] = None

    def lr_find(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        old_lrs = _initial_lrs(trainer.optimizers)
        with isolate_rng():
            self.optimal_lr = _lr_find(
                trainer,
                pl_module,
                min_lr=self._min_lr,
                max_lr=self._max_lr,
                num_training=self._num_training_steps,
                mode=self._mode,
                early_stop_threshold=self._early_stop_threshold,
                update_attr=self._update_attr,
                attr_name=self._attr_name,
                weights_only=self._weights_only,
            )

        if self._early_exit:
            raise _TunerExitException()

        if self._update_attr and self.optimal_lr is not None and self.optimal_lr._optimal_idx is not None:
            self._lr_ignored = not _reconfigure_optimizers(trainer, old_lrs)

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.lr_find(trainer, pl_module)


def _initial_lrs(optimizers: list[Optimizer]) -> list:
    return [group.get("initial_lr", group["lr"]) for optimizer in optimizers for group in optimizer.param_groups]


def _reconfigure_optimizers(trainer: "pl.Trainer", old_lrs: Optional[list] = None) -> bool:
    """Runs ``configure_optimizers`` again after a tuner changed a value it may read, such as the learning rate or the
    batch size behind ``trainer.estimated_stepping_batches``.

    This only happens before the first training batch, so no state is lost. A checkpoint passed to ``fit`` is restored
    after ``on_fit_start``.

    Returns ``False`` if the new optimizers have the learning rates in ``old_lrs``, which means that
    ``configure_optimizers`` ignores the tuned one. The old optimizers, which already hold it, are then kept.

    """
    from lightning.pytorch.strategies import DeepSpeedStrategy

    strategy = trainer.strategy
    # the DeepSpeed engine owns the optimizer and the LR scheduler
    if trainer.state.fn != "fit" or trainer.fit_loop.total_batch_idx >= 0 or isinstance(strategy, DeepSpeedStrategy):
        return True

    old_optimizers, old_scheduler_configs = strategy.optimizers, strategy.lr_scheduler_configs
    current_lrs = [(group, group["lr"]) for optimizer in old_optimizers for group in optimizer.param_groups]
    strategy.setup_optimizers(trainer)
    if old_lrs == _initial_lrs(strategy.optimizers):
        for group, lr in current_lrs:
            group["lr"] = lr  # a new scheduler on a reused optimizer overwrites it
        strategy.optimizers, strategy.lr_scheduler_configs = old_optimizers, old_scheduler_configs
        return False
    _optimizers_to_device(strategy.optimizers, strategy.root_device)
    return True
