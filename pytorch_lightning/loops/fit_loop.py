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
from contextlib import suppress
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.epoch import TrainingEpochLoop
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import FitLoopProgress
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import rank_zero_info

log = logging.getLogger(__name__)


class FitLoop(Loop):
    """This Loop iterates over the epochs to run the training

    Args:
        min_epochs: The minimum number of epochs
        max_epochs: The maximum number of epochs
        min_steps: The minimum number of steps
        max_steps: The maximum number of epoch

    .. note::
        If neither the minimum epochs nor steps are specified the minimum number of epochs is set to 1
        and if neither the maximum steps nor epochs are specified, the maximum epochs are set to 1000.
    """

    def __init__(
        self,
        min_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None
    ):
        super().__init__()
        self.max_epochs = 1000 if (max_epochs is None and max_steps is None) else max_epochs
        self.min_epochs = 1 if (min_epochs is None and min_steps is None) else min_epochs
        self.progress = FitLoopProgress()

        self.epoch_loop = TrainingEpochLoop(min_steps, max_steps)

    @property
    def current_epoch(self) -> int:
        """Return the current epoch"""
        return self.iteration_count

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        """Setter for the current epoch"""
        self.iteration_count = value

    @property
    def global_step(self) -> int:
        """Returns the global step"""
        return self.epoch_loop.global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        """Sets the global step (forwards to epoch_loop)"""
        self.epoch_loop.global_step = value

    @property
    def total_batch_idx(self) -> int:
        """Returns the total number of batches already run (across all epochs)"""
        return self.epoch_loop.total_batch_idx

    @property
    def batch_idx(self) -> int:
        """Returns the number of batches already run within this epoch"""
        return self.epoch_loop.iteration_count

    @property
    def split_idx(self) -> int:
        """Returns the index of the current batch split (within the current batch) for bptt"""
        return self.epoch_loop.split_idx

    @property
    def min_steps(self) -> int:
        # TODO(@justusschock): Why aren't we using the attribute in this class?
        """Returns the minimum numnber of steps to run"""
        return self.epoch_loop.min_steps

    @min_steps.setter
    def min_steps(self, value: int) -> None:
        """Sets the minimum number of steps (forwards to epoch_loop)"""
        # TODO(@awaelchli): This setter is required by debugging connector (fast dev run), should be avoided
        self.epoch_loop.min_steps = value

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to run"""
        return self.epoch_loop.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        """Sets the maximum number of steps (forwards to epoch_loop)"""
        # TODO(@awaelchli): This setter is required by debugging connector (fast dev run), should be avoided
        self.epoch_loop.max_steps = value

    @property
    def running_loss(self) -> TensorRunningAccum:
        """Returns the running loss"""
        return self.epoch_loop.batch_loop.running_loss

    @property
    def _skip_backward(self) -> bool:
        """ Determines whether the loop will skip backward during automatic optimization. """
        return self.epoch_loop.batch_loop._skip_backward

    @_skip_backward.setter
    def _skip_backward(self, value: bool) -> None:
        """ Determines whether the loop will skip backward during automatic optimization. """
        self.epoch_loop.batch_loop._skip_backward = value

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

        Returns True if trainer.should_stop was set (e.g. by early stopping)
        or if the maximum number of steps or epochs is reached.
        """
        # TODO(@awaelchli): Move track steps inside training loop and move part of these condition inside training loop
        stop_steps = self.max_steps is not None and self.global_step >= self.max_steps
        stop_epochs = self.max_epochs is not None and self.current_epoch >= self.max_epochs

        should_stop = False
        if self.trainer.should_stop:
            # early stopping
            met_min_epochs = self.current_epoch >= self.min_epochs if self.min_epochs else True
            met_min_steps = self.global_step >= self.min_steps if self.min_steps else True
            if met_min_epochs and met_min_steps:
                should_stop = True
            else:
                log.info(
                    'Trainer was signaled to stop but required minimum epochs'
                    f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                    ' not been met. Training will continue...'
                )
        self.trainer.should_stop = should_stop

        return stop_steps or should_stop or stop_epochs

    @property
    def skip(self) -> bool:
        """Whether we should skip the training and immediately return from the call to :meth:`run`."""
        return self.done or self.trainer.num_training_batches == 0

    def connect(
        self, trainer: 'pl.Trainer', *args: Any, progress: Optional[FitLoopProgress] = None, **kwargs: Any
    ) -> None:
        """Connects the loop with necessary arguments like the trainer"""
        super().connect(trainer, *args, **kwargs)
        if progress is not None:
            self.progress = progress
        self.epoch_loop.connect(trainer, progress=self.progress.epoch)

    def reset(self) -> None:
        """Resets the internal state of this loop"""

    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        self._results.to(device=self.trainer.lightning_module.device)
        self.trainer.call_hook("on_train_start")

    def on_advance_start(self) -> None:
        """Prepares the dataloader for training and calls the hooks ``on_epoch_start`` and ``on_train_epoch_start``"""
        model = self.trainer.lightning_module

        # reset train dataloader
        if self.current_epoch != 0 and self.trainer._should_reload_dl_epoch:
            self.trainer.reset_train_dataloader(model)

        # TODO: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.epoch_loop.batch_loop.accumulated_loss = TensorRunningAccum(
            window_length=self.trainer.accumulate_grad_batches
        )

    def advance(self) -> None:
        """Runs one whole epoch."""
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)
        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)

        with self.trainer.profiler.profile("run_training_epoch"):
            # run train epoch
            epoch_output = self.epoch_loop.run(train_dataloader)

            if epoch_output is None:
                return

            # the global step is manually decreased here due to backwards compatibility with existing loggers
            # as they expect that the same step is used when logging epoch end metrics even when the batch loop has
            # finished. this means the attribute does not exactly track the number of optimizer steps applied.
            # TODO(@carmocca): deprecate and rename so users don't get confused
            self.global_step -= 1
            # log epoch metrics
            self.trainer.logger_connector.update_train_epoch_metrics()
            self.global_step += 1

    def on_advance_end(self) -> None:
        """Updates the LR schedulers and does some internal bookkeeping"""
        if self.epoch_loop.batches_seen == 0:
            return

        did_train_only = not self.trainer.enable_validation or self.epoch_loop.val_loop.skip
        if did_train_only:
            self.global_step -= 1
            self._check_checkpoint_callback(True)
            self.global_step += 1

    def on_run_end(self) -> None:
        """Calls the ``on_train_end`` hook"""
        # NOTE: the iteration_count/current_epoch is already incremented
        # Lightning today does not increment the current epoch at the last epoch run in Trainer.fit
        # To simulate that current behavior, we decrement here.
        # TODO: must be fixed by https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
        self.current_epoch -= 1

        # trigger checkpoint check. need to temporarily decrease the global step to avoid saving duplicates
        # when a checkpoint was saved at the last step
        self.epoch_loop.global_step -= 1
        # TODO: see discussion/rework https://github.com/PyTorchLightning/pytorch-lightning/issues/7406
        self._check_checkpoint_callback(should_update=True, is_last=True)
        self.epoch_loop.global_step += 1

        # hook
        self.trainer.call_hook("on_train_end")

        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu
        # kill loggers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize("success")

        # summarize profile results
        self.trainer.profiler.describe()

        # give accelerators a chance to finish
        self.trainer.accelerator.on_train_end()

    def should_accumulate(self) -> bool:
        """Whether the gradients should be accumulated"""
        return self.epoch_loop.batch_loop.should_accumulate()

    def _check_checkpoint_callback(self, should_update: bool, is_last: bool = False):
        """Checks if checkpointing needs to be done"""
        # TODO: bake this logic into the ModelCheckpoint callback
        if should_update:
            callbacks = self.trainer.checkpoint_callbacks

            if is_last and any(cb.save_last and cb.verbose for cb in callbacks):
                rank_zero_info("Saving latest checkpoint...")

            model = self.trainer.lightning_module

            for cb in callbacks:
                cb.on_validation_end(self.trainer, model)

    def state_dict(self) -> Dict:
        return {"epoch_loop": self.epoch_loop.state_dict()}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.epoch_loop.load_state_dict(state_dict["epoch_loop"])

    def teardown(self) -> None:
        self.epoch_loop.teardown()
