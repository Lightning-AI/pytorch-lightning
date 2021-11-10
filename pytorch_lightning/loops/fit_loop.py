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
import logging
from typing import Optional

from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.epoch import TrainingEpochLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


class FitLoop(Loop):
    """This Loop iterates over the epochs to run the training.

    Args:
        min_epochs: The minimum number of epochs
        max_epochs: The maximum number of epochs, can be set -1 to turn this limit off
    """

    def __init__(
        self,
        min_epochs: Optional[int] = 1,
        max_epochs: int = 1000,
    ) -> None:
        super().__init__()
        if max_epochs < -1:
            # Allow max_epochs to be zero, since this will be handled by fit_loop.done
            raise MisconfigurationException(
                f"`max_epochs` must be a non-negative integer or -1. You passed in {max_epochs}."
            )

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.epoch_loop: Optional[TrainingEpochLoop] = None
        self.epoch_progress = Progress()
        self._is_fresh_start_epoch: bool = True

    @property
    def current_epoch(self) -> int:
        """Return the current epoch."""
        return self.epoch_progress.current.completed

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        """Setter for the current epoch."""
        self.epoch_progress.current.completed = value

    @property
    def global_step(self) -> int:
        """Returns the global step."""
        return self.epoch_loop.global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        """Sets the global step (forwards to epoch_loop)"""
        self.epoch_loop.global_step = value

    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        return self.epoch_loop.total_batch_idx

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        return self.epoch_loop.batch_idx

    @property
    def split_idx(self) -> int:
        """Returns the index of the current batch split (within the current batch) for bptt."""
        return self.epoch_loop.batch_loop.split_idx

    @property
    def min_steps(self) -> int:
        # TODO(@justusschock): Why aren't we using the attribute in this class?
        """Returns the minimum numnber of steps to run."""
        return self.epoch_loop.min_steps

    @min_steps.setter
    def min_steps(self, value: int) -> None:
        """Sets the minimum number of steps (forwards to epoch_loop)"""
        # TODO(@awaelchli): This setter is required by debugging connector (fast dev run), should be avoided
        self.epoch_loop.min_steps = value

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to run."""
        return self.epoch_loop.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        """Sets the maximum number of steps (forwards to epoch_loop)"""
        # TODO(@awaelchli): This setter is required by debugging connector (fast dev run), should be avoided
        if value is None:
            rank_zero_deprecation(
                "Setting `max_steps = None` is deprecated in v1.5 and will no longer be supported in v1.7."
                " Use `max_steps = -1` instead."
            )
            value = -1
        elif value < -1:
            raise MisconfigurationException(
                f"`max_steps` must be a non-negative integer or -1 (infinite steps). You passed in {value}."
            )
        self.epoch_loop.max_steps = value

    @property
    def running_loss(self) -> TensorRunningAccum:
        """Returns the running loss."""
        return self.epoch_loop.batch_loop.running_loss

    @property
    def _skip_backward(self) -> bool:
        """Determines whether the loop will skip backward during automatic optimization."""
        assert self.epoch_loop.batch_loop is not None
        assert self.epoch_loop.batch_loop.optimizer_loop is not None
        return self.epoch_loop.batch_loop.optimizer_loop._skip_backward

    @_skip_backward.setter
    def _skip_backward(self, value: bool) -> None:
        """Determines whether the loop will skip backward during automatic optimization."""
        assert self.epoch_loop.batch_loop is not None
        assert self.epoch_loop.batch_loop.optimizer_loop is not None
        self.epoch_loop.batch_loop.optimizer_loop._skip_backward = value

    @property
    def _results(self) -> ResultCollection:
        if self.trainer.training:
            return self.epoch_loop._results
        if self.trainer.validating:
            return self.epoch_loop.val_loop._results
        raise RuntimeError("`FitLoop._results` property isn't defined. Accessed outside of scope")

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop.

        Returns True if trainer.should_stop was set (e.g. by early stopping) or if the maximum number of steps or epochs
        is reached.
        """
        # TODO(@awaelchli): Move track steps inside training loop and move part of these condition inside training loop
        stop_steps = _is_max_limit_reached(self.global_step, self.max_steps)
        stop_epochs = _is_max_limit_reached(self.current_epoch, self.max_epochs)

        should_stop = False
        if self.trainer.should_stop:
            # early stopping
            met_min_epochs = self.current_epoch >= self.min_epochs if self.min_epochs else True
            met_min_steps = self.global_step >= self.min_steps if self.min_steps else True
            if met_min_epochs and met_min_steps:
                should_stop = True
            else:
                log.info(
                    "Trainer was signaled to stop but required minimum epochs"
                    f" ({self.min_epochs}) or minimum steps ({self.min_steps}) has"
                    " not been met. Training will continue..."
                )
        self.trainer.should_stop = should_stop

        return stop_steps or should_stop or stop_epochs or self.trainer.num_training_batches == 0

    @property
    def skip(self) -> bool:
        """Whether we should skip the training and immediately return from the call to :meth:`run`."""
        # since `trainer.num_training_batches` depends on the `train_dataloader` but that won't be called
        # until `on_run_start`, we use `limit_train_batches` instead
        return self.done or self.trainer.limit_train_batches == 0

    def connect(self, epoch_loop: TrainingEpochLoop):
        """Connects a training epoch loop to this fit loop."""
        self.epoch_loop = epoch_loop

    def reset(self) -> None:
        """Resets the internal state of this loop."""
        if self.restarting:
            self.epoch_progress.reset_on_restart()

    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        # reset train dataloader and val dataloader
        self.trainer.reset_train_val_dataloaders(self.trainer.lightning_module)
        self._is_fresh_start_epoch = True
        self._results.to(device=self.trainer.lightning_module.device)
        self.trainer.call_hook("on_train_start")

    def on_advance_start(self) -> None:
        """Prepares the dataloader for training and calls the hooks ``on_epoch_start`` and
        ``on_train_epoch_start``"""
        model = self.trainer.lightning_module

        # reset train dataloader
        if not self._is_fresh_start_epoch and self.trainer._should_reload_dl_epoch:
            self.trainer.reset_train_dataloader(model)
        self._is_fresh_start_epoch = False

        if self.trainer.train_dataloader is not None and callable(
            getattr(self.trainer.train_dataloader.sampler, "set_epoch", None)
        ):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.epoch_loop.batch_loop.accumulated_loss = TensorRunningAccum(
            window_length=self.trainer.accumulate_grad_batches
        )

        self.epoch_progress.increment_ready()

    def advance(self) -> None:
        """Runs one whole epoch."""
        dataloader = self.trainer.training_type_plugin.process_dataloader(self.trainer.train_dataloader)
        data_fetcher = self.trainer._data_connector.get_profiled_dataloader(dataloader)

        with self.trainer.profiler.profile("run_training_epoch"):
            self.epoch_loop.run(data_fetcher)

            # the global step is manually decreased here due to backwards compatibility with existing loggers
            # as they expect that the same step is used when logging epoch end metrics even when the batch loop has
            # finished. this means the attribute does not exactly track the number of optimizer steps applied.
            # TODO(@carmocca): deprecate and rename so users don't get confused
            self.global_step -= 1
            # log epoch metrics
            self.trainer.logger_connector.update_train_epoch_metrics()
            self.global_step += 1

    def on_advance_end(self) -> None:
        self.epoch_progress.increment_completed()

    def on_run_end(self) -> None:
        """Calls the ``on_train_end`` hook."""
        # NOTE: the current_epoch is already incremented
        # Lightning today does not increment the current epoch at the last epoch run in Trainer.fit
        # To simulate that current behavior, we decrement here.
        # TODO: must be fixed by https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
        self.current_epoch = max(self.current_epoch - 1, 0)

        # hook
        self.trainer.call_hook("on_train_end")

        # give accelerators a chance to finish
        self.trainer.training_type_plugin.on_train_end()

    def teardown(self) -> None:
        self.epoch_loop.teardown()

    def _should_accumulate(self) -> bool:
        """Whether the gradients should be accumulated."""
        return self.epoch_loop._should_accumulate()
