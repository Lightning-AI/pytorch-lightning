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
import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import torch
from lightning_utilities.core.imports import RequirementCache

import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import LRScheduler, LRSchedulerConfig, STEP_OUTPUT

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

_MATPLOTLIB_AVAILABLE = RequirementCache("matplotlib")
if TYPE_CHECKING and _MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
log = logging.getLogger(__name__)


def _determine_lr_attr_name(trainer: "pl.Trainer", model: "pl.LightningModule") -> str:
    if isinstance(trainer.auto_lr_find, str):
        if not lightning_hasattr(model, trainer.auto_lr_find):
            raise MisconfigurationException(
                f"`auto_lr_find` was set to {trainer.auto_lr_find}, however"
                " could not find this as a field in `model` or `model.hparams`."
            )
        return trainer.auto_lr_find

    attr_options = ("lr", "learning_rate")
    for attr in attr_options:
        if lightning_hasattr(model, attr):
            return attr

    raise MisconfigurationException(
        "When `auto_lr_find=True`, either `model` or `model.hparams` should"
        f" have one of these fields: {attr_options} overridden."
    )


class _LRFinder:
    """LR finder object. This object stores the results of lr_find().

    Args:
        mode: either `linear` or `exponential`, how to increase lr after each step

        lr_min: lr to start search from

        lr_max: lr to stop search

        num_training: number of steps to take between lr_min and lr_max

    Example::
        # Run lr finder
        lr_finder = trainer.lr_find(model)

        # Results stored in
        lr_finder.results

        # Plot using
        lr_finder.plot()

        # Get suggestion
        lr = lr_finder.suggestion()
    """

    def __init__(self, mode: str, lr_min: float, lr_max: float, num_training: int) -> None:
        assert mode in ("linear", "exponential"), "mode should be either `linear` or `exponential`"

        self.mode = mode
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_training = num_training

        self.results: Dict[str, Any] = {}
        self._total_batch_idx = 0  # for debug purpose

    def _exchange_scheduler(self, trainer: "pl.Trainer") -> None:
        # TODO: update docs here
        """Decorate `trainer.strategy.setup_optimizers` method such that it sets the user's originally specified
        optimizer together with a new scheduler that takes care of the learning rate search."""
        from pytorch_lightning.core.optimizer import _set_scheduler_opt_idx

        optimizers = trainer.strategy.optimizers

        if len(optimizers) != 1:
            raise MisconfigurationException(
                f"`model.configure_optimizers()` returned {len(optimizers)}, but"
                " learning rate finder only works with single optimizer"
            )

        optimizer = optimizers[0]

        new_lrs = [self.lr_min] * len(optimizer.param_groups)
        for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr

        args = (optimizer, self.lr_max, self.num_training)
        scheduler = _LinearLR(*args) if self.mode == "linear" else _ExponentialLR(*args)
        scheduler = cast(LRScheduler, scheduler)

        trainer.strategy.optimizers = [optimizer]
        trainer.strategy.lr_scheduler_configs = [LRSchedulerConfig(scheduler, interval="step", opt_idx=0)]
        _set_scheduler_opt_idx(trainer.optimizers, trainer.lr_scheduler_configs)

    def plot(self, suggest: bool = False, show: bool = False, ax: Optional["Axes"] = None) -> Optional["plt.Figure"]:
        """Plot results from lr_find run
        Args:
            suggest: if True, will mark suggested lr to use with a red point

            show: if True, will show figure

            ax: Axes object to which the plot is to be drawn. If not provided, a new figure is created.
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise MisconfigurationException(
                "To use the `plot` method, you must have Matplotlib installed."
                " Install it by running `pip install -U matplotlib`."
            )
        import matplotlib.pyplot as plt

        lrs = self.results["lr"]
        losses = self.results["loss"]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        if self.mode == "exponential":
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        if suggest:
            _ = self.suggestion()
            if self._optimal_idx:
                ax.plot(lrs[self._optimal_idx], losses[self._optimal_idx], markersize=10, marker="o", color="red")

        if show:
            plt.show()

        return fig

    def suggestion(self, skip_begin: int = 10, skip_end: int = 1) -> Optional[float]:
        """This will propose a suggestion for an initial learning rate based on the point with the steepest
        negative gradient.

        Args:
            skip_begin: how many samples to skip in the beginning; helps to avoid too naive estimates
            skip_end: how many samples to skip in the end; helps to avoid too optimistic estimates

        Returns:
            The suggested initial learning rate to use, or `None` if a suggestion is not possible due to too few
            loss samples.
        """
        losses = np.array(self.results["loss"][skip_begin:-skip_end])
        losses = losses[np.isfinite(losses)]
        if len(losses) < 2:
            # computing np.gradient requires at least 2 points
            log.error(
                "Failed to compute suggestion for learning rate because there are not enough points. Increase the loop"
                " iteration limits or the size of your dataset/dataloader."
            )
            self._optimal_idx = None
            return None

        # TODO: When computing the argmin here, and some losses are non-finite, the expected indices could be
        #   incorrectly shifted by an offset
        min_grad = np.gradient(losses).argmin()
        self._optimal_idx = min_grad + skip_begin
        return self.results["lr"][self._optimal_idx]


def lr_find(
    trainer: "pl.Trainer",
    model: "pl.LightningModule",
    min_lr: float = 1e-8,
    max_lr: float = 1,
    num_training: int = 100,
    mode: str = "exponential",
    early_stop_threshold: Optional[float] = 4.0,
    update_attr: bool = False,
) -> Optional[_LRFinder]:
    """See :meth:`~pytorch_lightning.tuner.tuning.Tuner.lr_find`"""
    if trainer.fast_dev_run:
        rank_zero_warn("Skipping learning rate finder since `fast_dev_run` is enabled.")
        return None

    # Determine lr attr
    if update_attr:
        lr_attr_name = _determine_lr_attr_name(trainer, model)

    # Save initial model, that is loaded after learning rate is found
    ckpt_path = os.path.join(trainer.default_root_dir, f".lr_find_{uuid.uuid4()}.ckpt")
    ckpt_path = trainer.strategy.broadcast(ckpt_path)
    trainer.save_checkpoint(ckpt_path)

    start_steps = trainer.global_step

    # Arguments we adjust during the lr finder, save for restoring
    params = __lr_finder_dump_params(trainer)

    # Set to values that are required by the algorithm
    __lr_finder_reset_params(trainer, num_training, early_stop_threshold)

    # Disable standard progress bar for fit
    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    # Initialize lr finder object (stores results)
    lr_finder = _LRFinder(mode, min_lr, max_lr, num_training)

    # Configure optimizer and scheduler
    lr_finder._exchange_scheduler(trainer)

    # Fit, lr & loss logged in callback
    _try_loop_run(trainer, params)

    # Prompt if we stopped early
    if trainer.global_step != num_training + start_steps:
        log.info(f"LR finder stopped early after {trainer.global_step} steps due to diverging loss.")

    # Transfer results from callback to lr finder object
    lr_finder.results.update({"lr": trainer.callbacks[0].lrs, "loss": trainer.callbacks[0].losses})
    lr_finder._total_batch_idx = trainer.fit_loop.total_batch_idx  # for debug purpose

    __lr_finder_restore_params(trainer, params)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    # Update lr attr if required
    lr_finder.results = trainer.strategy.broadcast(lr_finder.results)
    if update_attr:
        lr = lr_finder.suggestion()

        # TODO: log lr.results to self.logger
        if lr is not None:
            lightning_setattr(model, lr_attr_name, lr)
            log.info(f"Learning rate set to {lr}")

    # Restore initial state of model
    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)
    trainer.fit_loop.restarting = False  # reset restarting flag as checkpoint restoring sets it to True

    return lr_finder


def __lr_finder_dump_params(trainer: "pl.Trainer") -> Dict[str, Any]:
    return {
        "optimizers": trainer.strategy.optimizers,
        "lr_scheduler_configs": trainer.strategy.lr_scheduler_configs,
        "optimizer_frequencies": trainer.strategy.optimizer_frequencies,
        "callbacks": trainer.callbacks,
        "loggers": trainer.loggers,
        # TODO: check if this is required
        "auto_lr_find": trainer.auto_lr_find,
        "max_steps": trainer.fit_loop.max_steps,
        "limit_val_batches": trainer.limit_val_batches,
        "loop_state_dict": deepcopy(trainer.fit_loop.state_dict()),
    }


def __lr_finder_reset_params(trainer: "pl.Trainer", num_training: int, early_stop_threshold: Optional[float]) -> None:
    from pytorch_lightning.loggers.logger import DummyLogger

    trainer.strategy.lr_scheduler_configs = []
    trainer.strategy.optimizer_frequencies = []
    # avoid lr find being called multiple times
    trainer.auto_lr_find = False
    # Use special lr logger callback
    trainer.callbacks = [_LRCallback(num_training, early_stop_threshold, progress_bar_refresh_rate=1)]
    # No logging
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    # Max step set to number of iterations starting at current number of iterations
    trainer.fit_loop.max_steps = num_training + trainer.global_step
    trainer.limit_val_batches = num_training


def __lr_finder_restore_params(trainer: "pl.Trainer", params: Dict[str, Any]) -> None:
    trainer.strategy.optimizers = params["optimizers"]
    trainer.strategy.lr_scheduler_configs = params["lr_scheduler_configs"]
    trainer.strategy.optimizer_frequencies = params["optimizer_frequencies"]
    trainer.auto_lr_find = params["auto_lr_find"]
    trainer.callbacks = params["callbacks"]
    trainer.loggers = params["loggers"]
    trainer.fit_loop.max_steps = params["max_steps"]
    trainer.limit_val_batches = params["limit_val_batches"]

    loop = trainer.fit_loop
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    trainer.should_stop = False


class _LRCallback(Callback):
    """Special callback used by the learning rate finder. This callback logs the learning rate before each batch
    and logs the corresponding loss after each batch.

    Args:
        num_training: number of iterations done by the learning rate finder
        early_stop_threshold: threshold for stopping the search. If the
            loss at any point is larger than ``early_stop_threshold*best_loss``
            then the search is stopped. To disable, set to ``None``.
        progress_bar_refresh_rate: rate to refresh the progress bar for
            the learning rate finder
        beta: smoothing value, the loss being logged is a running average of
            loss values logged until now. ``beta`` controls the forget rate i.e.
            if ``beta=0`` all past information is ignored.
    """

    def __init__(
        self,
        num_training: int,
        early_stop_threshold: Optional[float] = 4.0,
        progress_bar_refresh_rate: int = 0,
        beta: float = 0.98,
    ):
        self.num_training = num_training
        self.early_stop_threshold = early_stop_threshold
        self.beta = beta
        self.losses: List[float] = []
        self.lrs: List[float] = []
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.progress_bar = None

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called before each training batch, logs the lr that will be used."""
        if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar_refresh_rate and self.progress_bar is None:
            self.progress_bar = tqdm(desc="Finding best initial lr", total=self.num_training)

        self.lrs.append(trainer.lr_scheduler_configs[0].scheduler.lr[0])  # type: ignore[union-attr]

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the training batch ends, logs the calculated loss."""
        if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        if self.progress_bar:
            self.progress_bar.update()

        loss_tensor = trainer.fit_loop.running_loss.last()
        assert loss_tensor is not None
        current_loss = loss_tensor.item()
        current_step = trainer.global_step

        # Avg loss (loss with momentum) + smoothing
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (current_step + 1))

        # Check if we diverging
        if self.early_stop_threshold is not None:
            if current_step > 1 and smoothed_loss > self.early_stop_threshold * self.best_loss:
                trainer.should_stop = True  # stop signal
                if self.progress_bar:
                    self.progress_bar.close()

        trainer.should_stop = trainer.strategy.broadcast(trainer.should_stop)

        # Save best loss for diverging checking
        if smoothed_loss < self.best_loss or current_step == 1:
            self.best_loss = smoothed_loss

        self.losses.append(smoothed_loss)


class _LinearLR(_TORCH_LRSCHEDULER):
    """Linearly increases the learning rate between two boundaries over a number of iterations.

    Args:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self) -> Union[float, List[float]]:
        return self._lr


class _ExponentialLR(_TORCH_LRSCHEDULER):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self) -> Union[float, List[float]]:
        return self._lr


def _try_loop_run(trainer: "pl.Trainer", params: Dict[str, Any]) -> None:
    loop = trainer.fit_loop
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    loop.run()
