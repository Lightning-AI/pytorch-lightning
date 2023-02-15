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
import logging
import os
from typing import Any, Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.epoch import TrainingEpochLoop
from pytorch_lightning.loops.epoch.training_epoch_loop import _OUTPUTS_TYPE as _EPOCH_OUTPUTS_TYPE
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _set_sampler_epoch
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.supporters import CombinedLoader, TensorRunningAccum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import (
    AbstractDataFetcher,
    DataFetcher,
    DataLoaderIterDataFetcher,
    InterBatchParallelDataFetcher,
)
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature

log = logging.getLogger(__name__)


class FitLoop(Loop[None]):
    """This Loop iterates over the epochs to run the training.

    Args:
        min_epochs: The minimum number of epochs
        max_epochs: The maximum number of epochs, can be set -1 to turn this limit off
    """

    def __init__(
        self,
        min_epochs: Optional[int] = 0,
        max_epochs: Optional[int] = None,
    ) -> None:
        super().__init__()
        if isinstance(max_epochs, int) and max_epochs < -1:
            # Allow max_epochs to be zero, since this will be handled by fit_loop.done
            raise MisconfigurationException(
                f"`max_epochs` must be a non-negative integer or -1. You passed in {max_epochs}."
            )

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.epoch_loop = TrainingEpochLoop()
        self.epoch_progress = Progress()

        self._is_fresh_start_epoch: bool = True
        self._outputs: _EPOCH_OUTPUTS_TYPE = []
        self._data_fetcher: Optional[AbstractDataFetcher] = None

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
    def min_steps(self) -> Optional[int]:
        # TODO(@justusschock): Why aren't we using the attribute in this class?
        """Returns the minimum number of steps to run."""
        return self.epoch_loop.min_steps

    @min_steps.setter
    def min_steps(self, value: Optional[int]) -> None:
        """Sets the minimum number of steps (forwards to epoch_loop)"""
        # TODO: This setter is required by debugging connector (fast dev run), should be avoided
        self.epoch_loop.min_steps = value

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to run."""
        return self.epoch_loop.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        """Sets the maximum number of steps (forwards to epoch_loop)"""
        # TODO: This setter is required by debugging connector (fast dev run), should be avoided
        if value < -1:
            raise MisconfigurationException(
                f"`max_steps` must be a non-negative integer or -1 (infinite steps). You passed in {value}."
            )
        self.epoch_loop.max_steps = value

    @property
    def running_loss(self) -> TensorRunningAccum:
        """Returns the running loss."""
        return self.epoch_loop.batch_loop.running_loss

    @Loop.restarting.setter
    def restarting(self, restarting: bool) -> None:
        # if the last epoch completely finished, we are not actually restarting
        values = self.epoch_progress.current.ready, self.epoch_progress.current.started
        epoch_unfinished = any(v != self.epoch_progress.current.processed for v in values)
        restarting = restarting and epoch_unfinished or self._iteration_based_training()
        Loop.restarting.fset(self, restarting)  # call the parent setter

    @property
    def prefetch_batches(self) -> int:
        is_unsized = self.trainer.num_training_batches == float("inf")
        inter_batch_parallelism = os.getenv("PL_INTER_BATCH_PARALLELISM", "0") == "1"
        return 1 if is_unsized or inter_batch_parallelism else 0

    @property
    def _skip_backward(self) -> bool:
        """Determines whether the loop will skip backward during automatic optimization."""
        return self.epoch_loop.batch_loop.optimizer_loop._skip_backward

    @_skip_backward.setter
    def _skip_backward(self, value: bool) -> None:
        """Determines whether the loop will skip backward during automatic optimization."""
        self.epoch_loop.batch_loop.optimizer_loop._skip_backward = value

    @property
    def _results(self) -> _ResultCollection:
        if self.trainer.training:
            return self.epoch_loop._results
        if self.trainer.validating:
            return self.epoch_loop.val_loop._results
        raise RuntimeError("`FitLoop._results` property isn't defined. Accessed outside of scope")

    @property
    def _can_stop_early(self) -> bool:
        met_min_epochs = self.epoch_progress.current.processed >= self.min_epochs if self.min_epochs else True
        met_min_steps = self.epoch_loop.global_step >= self.min_steps if self.min_steps else True
        return met_min_epochs and met_min_steps

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop."""
        if self.trainer.num_training_batches == 0:
            rank_zero_info("`Trainer.fit` stopped: No training batches.")
            return True

        # TODO: Move track steps inside training loop and move part of these condition inside training loop
        stop_steps = _is_max_limit_reached(self.epoch_loop.global_step, self.max_steps)
        if stop_steps:
            rank_zero_info(f"`Trainer.fit` stopped: `max_steps={self.max_steps!r}` reached.")
            return True

        # `processed` is increased before `on_train_epoch_end`, the hook where checkpoints are typically saved.
        # we use it here because the checkpoint data won't have `completed` increased yet
        assert isinstance(self.max_epochs, int)
        stop_epochs = _is_max_limit_reached(self.epoch_progress.current.processed, self.max_epochs)
        if stop_epochs:
            # in case they are not equal, override so `trainer.current_epoch` has the expected value
            self.epoch_progress.current.completed = self.epoch_progress.current.processed
            rank_zero_info(f"`Trainer.fit` stopped: `max_epochs={self.max_epochs!r}` reached.")
            return True

        if self.trainer.should_stop and self._can_stop_early:
            rank_zero_debug("`Trainer.fit` stopped: `trainer.should_stop` was set.")
            return True

        return False

    @property
    def skip(self) -> bool:
        """Whether we should skip the training and immediately return from the call to :meth:`run`."""
        # since `trainer.num_training_batches` depends on the `train_dataloader` but that won't be called
        # until `on_run_start`, we use `limit_train_batches` instead
        return self.done or self.trainer.limit_train_batches == 0

    def connect(self, epoch_loop: TrainingEpochLoop) -> None:  # type: ignore[override]
        """Connects a training epoch loop to this fit loop."""
        self.epoch_loop = epoch_loop

    def reset(self) -> None:
        """Resets the internal state of this loop."""
        if self.restarting:
            self.epoch_progress.reset_on_restart()

    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        # update the current_epoch in-case of checkpoint reload
        if not self._iteration_based_training():
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        self.trainer.reset_train_dataloader(self.trainer.lightning_module)
        # reload the evaluation dataloaders too for proper display in the progress bar
        if self.epoch_loop._should_check_val_epoch():
            self.epoch_loop.val_loop._reload_evaluation_dataloaders()

        data_fetcher_cls = _select_data_fetcher(self.trainer)
        self._data_fetcher = data_fetcher_cls(prefetch_batches=self.prefetch_batches)

        self._is_fresh_start_epoch = True
        self._results.to(device=self.trainer.lightning_module.device)

        self.trainer._call_callback_hooks("on_train_start")
        self.trainer._call_lightning_module_hook("on_train_start")
        self.trainer._call_strategy_hook("on_train_start")

    def on_advance_start(self) -> None:
        """Prepares the dataloader for training and calls the hook ``on_train_epoch_start``"""
        model = self.trainer.lightning_module

        # reset train dataloader
        if not self._is_fresh_start_epoch and self.trainer._data_connector._should_reload_train_dl:
            log.detail(f"{self.__class__.__name__}: resetting train dataloader")
            self.trainer.reset_train_dataloader(model)
        self._is_fresh_start_epoch = False

        # reset outputs here instead of in `reset` as they are not accumulated between epochs
        self._outputs = []

        if self.trainer.train_dataloader is not None:
            assert isinstance(self.trainer.train_dataloader, CombinedLoader)
            _set_sampler_epoch(self.trainer.train_dataloader, self.epoch_progress.current.processed)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.epoch_loop.batch_loop.accumulated_loss.reset(window_length=self.trainer.accumulate_grad_batches)

        self.epoch_progress.increment_ready()

        self.trainer._logger_connector.on_epoch_start()

        self.trainer._call_callback_hooks("on_train_epoch_start")
        self.trainer._call_lightning_module_hook("on_train_epoch_start")

        self.epoch_progress.increment_started()

    def advance(self) -> None:
        """Runs one whole epoch."""
        log.detail(f"{self.__class__.__name__}: advancing loop")
        assert self.trainer.train_dataloader is not None
        dataloader = self.trainer.train_dataloader

        def batch_to_device(batch: Any) -> Any:
            batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=0)
            batch = self.trainer._call_strategy_hook("batch_to_device", batch, dataloader_idx=0)
            return batch

        assert self._data_fetcher is not None
        self._data_fetcher.setup(dataloader, batch_to_device=batch_to_device)
        with self.trainer.profiler.profile("run_training_epoch"):
            self._outputs = self.epoch_loop.run(self._data_fetcher)

    def on_advance_end(self) -> None:
        # inform logger the batch loop has finished
        self.trainer._logger_connector.epoch_end_reached()

        # get the model and call model.training_epoch_end
        model = self.trainer.lightning_module
        if is_overridden("training_epoch_end", model) and self._outputs:
            epoch_end_outputs = self.epoch_loop._prepare_outputs_training_epoch_end(
                self._outputs,
                lightning_module=model,
                num_optimizers=len(self.trainer.optimizers),
            )
            # run lightning module hook training_epoch_end
            # refresh the result for custom logging at the epoch level
            epoch_end_outputs = self.trainer._call_lightning_module_hook("training_epoch_end", epoch_end_outputs)
            if epoch_end_outputs is not None:
                raise MisconfigurationException(
                    "`training_epoch_end` expects a return of None. "
                    "HINT: remove the return statement in `training_epoch_end`."
                )
        # free memory
        self._outputs = []

        self.epoch_progress.increment_processed()

        # call train epoch end hooks
        self.trainer._call_callback_hooks("on_train_epoch_end")
        self.trainer._call_lightning_module_hook("on_train_epoch_end")

        self.trainer._logger_connector.on_epoch_end()

        if self.epoch_loop._num_ready_batches_reached():
            # if we are restarting and the above condition holds, it's because we are reloading an epoch-end checkpoint.
            # since metric-based schedulers require access to metrics and those are not currently saved in the
            # checkpoint, the plateau schedulers shouldn't be updated
            self.epoch_loop.update_lr_schedulers("epoch", update_plateau_schedulers=not self.restarting)

        # we manually decrease here because loggers expect that the same step is used when logging epoch-end metrics
        # even when the batch loop has finished
        self.epoch_loop._batches_that_stepped -= 1
        # log epoch metrics
        self.trainer._logger_connector.update_train_epoch_metrics()
        self.epoch_loop._batches_that_stepped += 1

        self.epoch_progress.increment_completed()

        # if fault tolerant is enabled and process has been notified, exit.
        self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> None:
        """Calls the ``on_train_end`` hook."""
        log.detail(f"{self.__class__.__name__}: train run ended")

        # hook
        self.trainer._call_callback_hooks("on_train_end")
        self.trainer._call_lightning_module_hook("on_train_end")
        self.trainer._call_strategy_hook("on_train_end")

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None
        self.epoch_loop.teardown()

    def _should_accumulate(self) -> bool:
        """Whether the gradients should be accumulated."""
        return self.epoch_loop._should_accumulate()

    def _iteration_based_training(self) -> bool:
        return self.trainer.max_steps != -1


def _select_data_fetcher(trainer: "pl.Trainer") -> Type[AbstractDataFetcher]:
    training_step_fx = getattr(trainer.lightning_module, "training_step")
    if is_param_in_hook_signature(training_step_fx, "dataloader_iter", explicit=True):
        rank_zero_warn(
            "Found `dataloader_iter` argument in the `training_step`. Note that the support for "
            "this signature is experimental and the behavior is subject to change."
        )
        return DataLoaderIterDataFetcher
    elif os.getenv("PL_INTER_BATCH_PARALLELISM", "0") == "1":
        if not isinstance(trainer.accelerator, CUDAAccelerator):
            raise MisconfigurationException("Inter batch parallelism is available only when using Nvidia GPUs.")
        return InterBatchParallelDataFetcher
    return DataFetcher
