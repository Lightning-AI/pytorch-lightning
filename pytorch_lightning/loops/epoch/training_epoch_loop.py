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
import math
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, overload, Tuple, Union

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning import loops  # import as loops to avoid circular imports
from pytorch_lightning.loops.batch import TrainingBatchLoop
from pytorch_lightning.loops.batch.training_batch_loop import _OUTPUTS_TYPE as _BATCH_OUTPUTS_TYPE
from pytorch_lightning.loops.utilities import _get_active_optimizers, _is_max_limit_reached, _v1_8_output_format
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.progress import BatchProgress, SchedulerProgress
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.auto_restart import _collect_states_on_rank_zero_over_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.warnings import WarningCache

_OUTPUTS_TYPE = List[_BATCH_OUTPUTS_TYPE]


class TrainingEpochLoop(loops.Loop[_OUTPUTS_TYPE]):
    """Runs over all batches in a dataloader (one epoch).

    Args:
        min_steps: The minimum number of steps (batches) to process
        max_steps: The maximum number of steps (batches) to process
    """

    def __init__(self, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__()
        if max_steps is None:
            rank_zero_deprecation(
                "Setting `max_steps = None` is deprecated in v1.5 and will no longer be supported in v1.7."
                " Use `max_steps = -1` instead."
            )
            max_steps = -1
        elif max_steps < -1:
            raise MisconfigurationException(
                f"`max_steps` must be a non-negative integer or -1 (infinite steps). You passed in {max_steps}."
            )
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.batch_progress = BatchProgress()
        self.scheduler_progress = SchedulerProgress()

        self.batch_loop = TrainingBatchLoop()
        self.val_loop = loops.EvaluationLoop(verbose=False)

        self._results = _ResultCollection(training=True)
        self._outputs: _OUTPUTS_TYPE = []
        self._warning_cache = WarningCache()
        # caches the loaded dataloader state until dataloader objects are available
        self._dataloader_state_dict: Dict[str, Any] = {}
        self._batches_that_stepped: int = 0

    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return self.batch_progress.total.ready - 1

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return self.batch_progress.current.ready - 1

    @property
    def global_step(self) -> int:
        lightning_module = self.trainer.lightning_module
        if lightning_module is None or lightning_module.automatic_optimization:
            return self.batch_loop.optimizer_loop.optim_progress.optimizer_steps
        return self.batch_loop.manual_loop.optim_step_progress.total.completed

    @property
    def _is_training_done(self) -> bool:
        max_steps_reached = _is_max_limit_reached(self.global_step, self.max_steps)
        return max_steps_reached or self._num_ready_batches_reached()

    @property
    def _is_validation_done(self) -> bool:
        # when we are restarting we want to check whether the val loop has finished
        return not self.restarting or self.val_loop.done

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop."""
        return (self._is_training_done and self._is_validation_done) or self.trainer.should_stop

    def connect(  # type: ignore[override]
        self,
        batch_loop: Optional[TrainingBatchLoop] = None,
        val_loop: Optional["loops.EvaluationLoop"] = None,
    ) -> None:
        """Optionally connect a custom batch or validation loop to this training epoch loop."""
        if batch_loop is not None:
            self.batch_loop = batch_loop
        if val_loop is not None:
            self.val_loop = val_loop

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        if self.restarting:
            self.batch_progress.reset_on_restart()
            self.scheduler_progress.reset_on_restart()
            self.batch_loop.optimizer_loop.optim_progress.reset_on_restart()

            trainer = self.trainer
            if not trainer.state._fault_tolerant_mode.is_enabled and trainer.num_training_batches != float("inf"):
                expected_steps = math.ceil(trainer.num_training_batches / trainer.accumulate_grad_batches)
                if self.global_step % expected_steps != 0:
                    rank_zero_warn(
                        "You're resuming from a checkpoint that ended before the epoch ended. This can cause unreliable"
                        " results if further training is done. Consider using an end-of-epoch checkpoint or enabling"
                        " fault-tolerant training:"
                        " https://pytorch-lightning.readthedocs.io/en/stable/advanced/fault_tolerant_training.html"
                    )
        else:
            self.batch_progress.reset_on_run()
            self.scheduler_progress.reset_on_run()
            self.batch_loop.optimizer_loop.optim_progress.reset_on_run()
            # when the epoch starts, the total val batch progress should be reset as it's supposed to count the batches
            # seen per epoch, this is useful for tracking when validation is run multiple times per epoch
            self.val_loop.epoch_loop.batch_progress.total.reset()

        self._outputs = []

    def on_run_start(self, data_fetcher: AbstractDataFetcher) -> None:  # type: ignore[override]
        self._reload_dataloader_state_dict(data_fetcher)
        _ = iter(data_fetcher)  # creates the iterator inside the fetcher
        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready

    def advance(self, data_fetcher: AbstractDataFetcher) -> None:  # type: ignore[override]
        """Runs a single training batch.

        Raises:
            StopIteration: When the epoch is canceled by the user returning -1
        """
        if self.restarting and self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch):
            # skip training and run validation in `on_advance_end`
            return
        # we are going to train first so the val loop does not need to restart
        self.val_loop.restarting = False

        if not isinstance(data_fetcher, DataLoaderIterDataFetcher):
            batch_idx = self.batch_idx + 1
            batch = next(data_fetcher)
        else:
            batch_idx, batch = next(data_fetcher)
        self.batch_progress.is_last_batch = data_fetcher.done

        self.batch_progress.increment_ready()

        self.trainer._logger_connector.on_batch_start(batch, batch_idx)

        if batch is None:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            batch_output = []
        else:
            # hook
            self.trainer._call_callback_hooks("on_batch_start")

            # TODO: Update this in v1.7 (deprecation: #9816)
            model_fx = self.trainer.lightning_module.on_train_batch_start
            extra_kwargs = (
                {"dataloader_idx": 0}
                if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
                else {}
            )

            # hook
            self.trainer._call_callback_hooks("on_train_batch_start", batch, batch_idx, **extra_kwargs)
            response = self.trainer._call_lightning_module_hook(
                "on_train_batch_start", batch, batch_idx, **extra_kwargs
            )
            self.trainer._call_strategy_hook("on_train_batch_start", batch, batch_idx, **extra_kwargs)
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            self.batch_progress.increment_started()

            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.batch_loop.run(batch, batch_idx)

        self.batch_progress.increment_processed()

        # update non-plateau LR schedulers
        # update epoch-interval ones only when we are at the end of training epoch
        self.update_lr_schedulers("step", update_plateau_schedulers=False)
        if self._num_ready_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

        batch_end_outputs = self._prepare_outputs_training_batch_end(
            batch_output,
            lightning_module=self.trainer.lightning_module,
            num_optimizers=len(self.trainer.optimizers),
        )

        # TODO: Update this in v1.7 (deprecation: #9816)
        model_fx = self.trainer.lightning_module.on_train_batch_end
        extra_kwargs = (
            {"dataloader_idx": 0}
            if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
            else {}
        )
        self.trainer._call_callback_hooks("on_train_batch_end", batch_end_outputs, batch, batch_idx, **extra_kwargs)
        self.trainer._call_lightning_module_hook(
            "on_train_batch_end", batch_end_outputs, batch, batch_idx, **extra_kwargs
        )
        self.trainer._call_callback_hooks("on_batch_end")
        self.trainer._logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        if is_overridden("training_epoch_end", self.trainer.lightning_module):
            self._outputs.append(batch_output)

        # -----------------------------------------
        # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
        # -----------------------------------------
        self.trainer._logger_connector.update_train_step_metrics()

    def on_advance_end(self) -> None:
        # -----------------------------------------
        # VALIDATE IF NEEDED
        # -----------------------------------------
        should_check_val = self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch)
        if should_check_val:
            self.trainer.validating = True
            self._run_validation()
            self.trainer.training = True

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # this is increased once per batch disregarding multiple optimizers or tbptt on purpose for loggers
            self._batches_that_stepped += 1
        # this will save based on the `batches_that_stepped` value
        self._save_loggers_on_train_batch_end()

        # if training finished, defer exit to the parent. this assumes there will be enough time in between
        # which might not be the case depending on what's in the `*_epoch_end` hooks
        if not self._is_training_done:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> _OUTPUTS_TYPE:
        outputs, self._outputs = self._outputs, []
        return outputs

    def teardown(self) -> None:
        self._results.cpu()
        self.batch_loop.teardown()
        self.val_loop.teardown()

    def on_save_checkpoint(self) -> Dict:
        state_dict = super().on_save_checkpoint()

        if (
            self.trainer is not None
            and self.trainer.state._fault_tolerant_mode.is_enabled
            and self.trainer.train_dataloader is not None
            and not self._num_completed_batches_reached()  # did not finish
            # TODO: fault-tolerance requires a minimum number of batches so probably should be > 0
            and self.batch_progress.current.ready  # did start
        ):
            loader: CombinedLoader = self.trainer.train_dataloader
            state = loader.state_dict(has_completed=self._has_completed())
            if state:
                state_dict["dataloader_state_dict"] = _collect_states_on_rank_zero_over_collection(state)

        return state_dict

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        # cache the dataloader state dict until the dataloader objects are available
        self._dataloader_state_dict = state_dict.get("dataloader_state_dict")

    def _run_validation(self) -> None:
        # reload dataloaders
        self.val_loop._reload_evaluation_dataloaders()

        with torch.no_grad():
            self.val_loop.run()

    def _accumulated_batches_reached(self) -> bool:
        """Determine if accumulation will be finished by the end of the current batch."""
        return self.batch_progress.current.ready % self.trainer.accumulate_grad_batches == 0

    def _num_ready_batches_reached(self) -> bool:
        """Checks if we are in the last batch or if there are more batches to follow."""
        epoch_finished_on_ready = self.batch_progress.current.ready == self.trainer.num_training_batches
        return epoch_finished_on_ready or self.batch_progress.is_last_batch

    def _num_completed_batches_reached(self) -> bool:
        epoch_finished_on_completed = self.batch_progress.current.completed == self.trainer.num_training_batches
        dataloader_consumed_successfully = self.batch_progress.is_last_batch and self._has_completed()
        return epoch_finished_on_completed or dataloader_consumed_successfully

    def _has_completed(self) -> bool:
        return self.batch_progress.current.ready == self.batch_progress.current.completed

    def _should_accumulate(self) -> bool:
        """Checks if the optimizer step should be performed or gradients should be accumulated for the current
        step."""
        accumulation_done = self._accumulated_batches_reached()
        # Lightning steps on the final batch
        is_final_batch = self._num_ready_batches_reached()
        # but the strategy might not
        strategy_accumulates_on_final_batch = self.trainer.strategy.handles_gradient_accumulation or not is_final_batch
        return not accumulation_done and strategy_accumulates_on_final_batch

    @staticmethod
    def _prepare_outputs_training_batch_end(
        batch_output: _BATCH_OUTPUTS_TYPE,
        lightning_module: "pl.LightningModule",
        num_optimizers: int,
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Processes the outputs from the batch loop into the format passed to the ``on_train_batch_end`` hook."""
        if not batch_output:
            return []

        # convert optimizer dicts to list
        if lightning_module.automatic_optimization:
            batch_output = apply_to_collection(
                batch_output, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )

        array = np.array(batch_output, dtype=object)
        # TODO: remove in v1.8
        if (
            num_optimizers > 1
            and lightning_module.truncated_bptt_steps > 0
            and not _v1_8_output_format(lightning_module.on_train_batch_end)
        ):
            rank_zero_deprecation(
                "You are training with multiple optimizers AND truncated backpropagation through time enabled."
                " The current format of the `on_train_batch_end(outputs, ...)` is a 2d list with sizes"
                " (n_optimizers, tbptt_steps), however, this has been deprecated and will change in version v1.8 to"
                " (tbptt_steps, n_optimizers). You can update your code by adding the following parameter to your"
                " hook signature: `on_train_batch_end(outputs, ..., new_format=True)`."
            )
            # (tbptt_steps, n_opt) -> (n_opt, tbptt_steps)
            if array.ndim == 1:
                array = np.expand_dims(array, 1)
            array = array.transpose((1, 0))
        # squeeze all single-element dimensions
        array = array.squeeze()
        array = array.tolist()
        array = _recursive_unpad(array)
        return array

    @staticmethod
    def _prepare_outputs_training_epoch_end(
        batch_outputs: _OUTPUTS_TYPE,
        lightning_module: "pl.LightningModule",
        num_optimizers: int,
    ) -> Union[List[List[List[Dict[str, Any]]]], List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Processes the outputs from the batch loop into the format passed to the ``training_epoch_end`` hook."""
        # `batch_outputs` (plural) is the same as `epoch_end_output` (singular)
        if not batch_outputs:
            return []

        # convert optimizer dicts to list
        if lightning_module.automatic_optimization:
            batch_outputs = apply_to_collection(
                batch_outputs, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )

        array = _recursive_pad(batch_outputs)
        # TODO: remove in v1.8
        if (
            num_optimizers > 1
            and lightning_module.truncated_bptt_steps > 0
            and not _v1_8_output_format(lightning_module.on_train_epoch_end)
        ):
            rank_zero_deprecation(
                "You are training with multiple optimizers AND truncated backpropagation through time enabled."
                " The current format of the `training_epoch_end(outputs)` is a 3d list with sizes"
                " (n_optimizers, n_batches, tbptt_steps), however, this has been deprecated and will change in version"
                " v1.8 to (n_batches, tbptt_steps, n_optimizers). You can update your code by adding the following"
                " parameter to your hook signature: `training_epoch_end(outputs, new_format=True)`."
            )
            # (n_batches, tbptt_steps, n_opt) -> (n_opt, n_batches, tbptt_steps)
            if array.ndim == 2:
                array = np.expand_dims(array, 2)
            array = array.transpose((2, 0, 1))
        # squeeze all single-element dimensions
        array = array.squeeze()
        array = array.tolist()
        array = _recursive_unpad(array)
        # in case we squeezed from 1-element array to a 0-dim array
        array = array if isinstance(array, list) else [array]
        # remove residual empty lists
        array = [item for item in array if not isinstance(item, list) or len(item)]
        return array

    def update_lr_schedulers(self, interval: str, update_plateau_schedulers: bool) -> None:
        """updates the lr schedulers based on the given interval."""
        if interval == "step" and self._should_accumulate():
            return
        active_optimizers = _get_active_optimizers(
            self.trainer.optimizers, self.trainer.optimizer_frequencies, self.total_batch_idx
        )
        self._update_learning_rates(
            interval=interval,
            update_plateau_schedulers=update_plateau_schedulers,
            opt_indices=[opt_idx for opt_idx, _ in active_optimizers],
        )

    def _update_learning_rates(
        self, interval: str, update_plateau_schedulers: bool, opt_indices: Optional[List[int]] = None
    ) -> None:
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.
            opt_indices: indices of the optimizers to update.
        """
        if not self.trainer.lr_scheduler_configs or not self.trainer.lightning_module.automatic_optimization:
            return

        if opt_indices is None:
            opt_indices = []

        for config in self.trainer.lr_scheduler_configs:
            if config.opt_idx not in opt_indices:
                continue

            if update_plateau_schedulers ^ config.reduce_on_plateau:
                continue

            current_idx = self.batch_idx if interval == "step" else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if config.interval == interval and current_idx % config.frequency == 0:
                monitor_val = None
                if config.reduce_on_plateau:
                    # If instance of ReduceLROnPlateau, we need a monitor
                    monitor_key = config.monitor
                    monitor_val = self._get_monitor_value(monitor_key)
                    if monitor_val is None:
                        if config.strict:
                            avail_metrics = list(self.trainer.callback_metrics)
                            raise MisconfigurationException(
                                f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                                f" which is not available. Available metrics are: {avail_metrics}."
                                " Condition can be set using `monitor` key in lr scheduler dict"
                            )
                        rank_zero_warn(
                            f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                            " which is not available but strict is set to `False`."
                            " Skipping learning rate update.",
                            category=RuntimeWarning,
                        )
                        continue

                self.scheduler_progress.increment_ready()

                # update LR
                self.trainer._call_lightning_module_hook(
                    "lr_scheduler_step",
                    config.scheduler,
                    config.opt_idx,
                    monitor_val,
                )
                self.scheduler_progress.increment_completed()

    def _get_monitor_value(self, key: str) -> Any:
        # this is a separate method to aid in testing
        return self.trainer.callback_metrics.get(key)

    def _should_check_val_epoch(self):
        return (
            self.trainer.enable_validation
            and (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        )

    def _should_check_val_fx(self, batch_idx: int, is_last_batch: bool) -> bool:
        """Decide if we should run validation."""
        if not self._should_check_val_epoch():
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self.trainer.val_check_batch == float("inf")
        if is_last_batch and is_infinite_dataset:
            return True

        if self.trainer.should_stop:
            return True

        # TODO(@awaelchli): let training/eval loop handle logic around limit_*_batches and val_check_batch
        is_val_check_batch = is_last_batch
        if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
            is_val_check_batch = (batch_idx + 1) % self.trainer.limit_train_batches == 0
        elif self.trainer.val_check_batch != float("inf"):
            is_val_check_batch = (batch_idx + 1) % self.trainer.val_check_batch == 0
        return is_val_check_batch

    def _save_loggers_on_train_batch_end(self) -> None:
        """Flushes loggers to disk."""
        # this assumes that `batches_that_stepped` was increased before
        should_flush = self._batches_that_stepped % self.trainer.flush_logs_every_n_steps == 0
        if should_flush or self.trainer.should_stop:
            for logger in self.trainer.loggers:
                logger.save()

    def _reload_dataloader_state_dict(self, data_fetcher: AbstractDataFetcher) -> None:
        if self._dataloader_state_dict:
            data_fetcher.dataloader.load_state_dict(self._dataloader_state_dict)
            self._dataloader_state_dict = None


def _convert_optim_dict(outs: Dict[int, Dict[str, Any]], num_optimizers: int) -> List[Optional[Dict[str, Any]]]:
    """Converts an optimizer dict to a list in which the key of the dict determines the position of the element.

    Example::
        >>> _convert_optim_dict({0: {"loss": 0.0}, 2: {"loss": 0.2}}, num_optimizers=3)
        [{'loss': 0.0}, None, {'loss': 0.2}]
    """
    return [outs[opt_idx] if opt_idx in outs else None for opt_idx in range(num_optimizers)]


@overload
def _recursive_unpad(nested: Any, value: Optional[Any] = None) -> Any:
    ...


@overload
def _recursive_unpad(nested: List[Any], value: Optional[Any] = None) -> List[Any]:
    ...


def _recursive_unpad(nested: Union[Any, List[Any]], value: Optional[Any] = None) -> Union[Any, List[Any]]:
    """Removes the given pad value from the nested list. Not strictly the reverse operation of
    :func:`_recursive_pad` because it removes the padding element everywhere, not just from the end of a list.

    Example::
        >>> _recursive_unpad([[[0, 1, 0]], [2], [0, 0]], value=0)
        [[[1]], [2], []]
    """
    if not isinstance(nested, list):
        return nested

    return [_recursive_unpad(item, value) for item in nested if item != value]


def _recursive_pad(nested: List[Any], fill_value: Optional[Any] = None) -> np.array:
    """Pads a jagged nested list of lists with the given value such that a proper multi-dimensional array can be
    formed with rectangular shape. The padding appends to the incomplete lists.

    Example::
        >>> _recursive_pad([[], [1], [2, 3], [4]], fill_value=0)  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 0], [1, 0], [2, 3], [4, 0]], dtype=object)
    """
    # code adapted from stackexchange:
    # https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape
    dimensions = _get_max_shape(nested)
    result = np.full(dimensions, fill_value, dtype=object)
    for index, value in _iterate_nested_array(nested):
        result[index] = value
    return result


def _get_dimensions(array: List[Any], level: int = 0) -> Generator:
    yield level, len(array)
    if all(isinstance(row, list) for row in array):
        for row in array:
            yield from _get_dimensions(row, level + 1)


def _get_max_shape(array: List[Any]) -> List[int]:
    """Calculates the max size in each dimension of a jagged (non-rectangular) nested list of lists.

    Example::
        >>> _get_max_shape([[], [[1], [2]], []])
        [3, 2, 1]
    """
    dimensions = defaultdict(int)
    for level, length in _get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def _iterate_nested_array(array: List[Any], index: Tuple = ()) -> Generator:
    if all(isinstance(item, list) for item in array):
        for idx, row in enumerate(array):
            yield from _iterate_nested_array(row, (*index, idx))
    else:  # final level
        yield (*index, slice(len(array))), array
