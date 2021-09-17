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
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch

from pytorch_lightning import loops  # import as loops to avoid circular imports
from pytorch_lightning.loops.batch import TrainingBatchLoop
from pytorch_lightning.loops.batch.training_batch_loop import _OUTPUTS_TYPE as _BATCH_OUTPUTS_TYPE
from pytorch_lightning.loops.utilities import _prepare_dataloader_iter
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import Progress, SchedulerProgress
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden

_OUTPUTS_TYPE = List[_BATCH_OUTPUTS_TYPE]


class TrainingEpochLoop(loops.Loop[_OUTPUTS_TYPE]):
    """Runs over all batches in a dataloader (one epoch).

    Args:
        min_steps: The minimum number of steps (batches) to process
        max_steps: The maximum number of steps (batches) to process
    """

    def __init__(self, min_steps: int, max_steps: int):
        super().__init__()
        self.min_steps: int = min_steps

        if max_steps and max_steps < -1:
            raise MisconfigurationException(f"`max_steps` must be a positive integer or -1. You passed in {max_steps}.")
        self.max_steps: int = max_steps

        self.global_step: int = 0
        # manually tracking which is the last batch is necessary for iterable dataset support
        self.is_last_batch: Optional[bool] = None
        self.batch_progress = Progress()
        self.scheduler_progress = SchedulerProgress()

        self.batch_loop: Optional[TrainingBatchLoop] = None
        self.val_loop: Optional["loops.EvaluationLoop"] = None

        self._results = ResultCollection(training=True)
        self._outputs: _OUTPUTS_TYPE = []

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
    def done(self) -> bool:
        """Returns whether the training should be stopped.

        The criteria are that the number of steps reached the max steps, the last batch is reached or the trainer
        signals to stop (e.g. by early stopping).
        """
        max_steps_reached = self.max_steps is not None and self.global_step >= self.max_steps
        return max_steps_reached or self.trainer.should_stop or self._num_training_batches_reached()

    def connect(
        self,
        batch_loop: TrainingBatchLoop = None,
        val_loop: Optional["loops.EvaluationLoop"] = None,
    ) -> None:
        """Optionally connect a custom batch or validation loop to this training epoch loop."""
        if batch_loop is not None:
            self.batch_loop = batch_loop
        if val_loop is not None:
            self.val_loop = val_loop

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        self.is_last_batch = False
        self._outputs = []

        if not self.restarting or self._num_training_batches_reached():
            self.batch_progress.current.reset()
            self.scheduler_progress.current.reset()
            assert self.batch_loop is not None
            assert self.batch_loop.optimizer_loop is not None
            self.batch_loop.optimizer_loop.optim_progress.reset_on_epoch()

    def on_run_start(self, dataloader_iter: Iterator, **kwargs: Any) -> None:
        # hook
        self.trainer.logger_connector.on_epoch_start()
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")
        self.trainer.fit_loop.epoch_progress.increment_started()

        self.dataloader_iter = _prepare_dataloader_iter(dataloader_iter, self.batch_idx + 1)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Runs a single training batch.

        Args:
            dataloader_iter: the iterator over the dataloader producing the new batch

        Raises:
            StopIteration: When the epoch is canceled by the user returning -1
        """
        batch_idx, (batch, is_last) = next(self.dataloader_iter)

        if not self.trainer.data_connector.train_data_fetcher.store_on_device:
            with self.trainer.profiler.profile("training_batch_to_device"):
                batch = self.trainer.accelerator.batch_to_device(batch)

        self.batch_progress.increment_ready()

        with self.trainer.profiler.profile("run_training_batch"):
            batch_output = self.batch_loop.run(batch, batch_idx)

        self.batch_progress.increment_processed()

        self.is_last_batch = is_last

        # when returning -1 from train_step, we end epoch early
        if batch_output.signal == -1:
            raise StopIteration

        # update non-plateau LR schedulers
        # update epoch-interval ones only when we are at the end of training epoch
        self.update_lr_schedulers("step", update_plateau_schedulers=False)
        if self._num_training_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

        # if is_overridden("on_train_batch_end", self.trainer.lightning_module):
        batch_end_outputs = self._prepare_outputs_training_batch_end(
            batch_output.outputs,
            automatic=self.trainer.lightning_module.trainer.lightning_module.automatic_optimization,
            num_optimizers=len(self.trainer.optimizers),
        )
        self.trainer.call_hook("on_train_batch_end", batch_end_outputs, batch, self.batch_idx, 0)
        self.trainer.call_hook("on_batch_end")
        self.trainer.logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        if is_overridden("training_epoch_end", self.trainer.lightning_module):
            self._outputs.append(batch_output.outputs)

        # -----------------------------------------
        # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
        # -----------------------------------------
        self.trainer.logger_connector.update_train_step_metrics()

    def on_advance_end(self):
        """Runs validation and Checkpointing if necessary.

        Raises:
            StopIteration: if :attr:`done` evaluates to ``True`` to finish this epoch
        """
        # -----------------------------------------
        # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
        # -----------------------------------------
        should_check_val = self._should_check_val_fx(self.batch_idx, self.is_last_batch)
        if should_check_val:
            self.trainer.validating = True
            self._run_validation()
            self.trainer.training = True

        # -----------------------------------------
        # SAVE LOGGERS (ie: Tensorboard, etc...)
        # -----------------------------------------
        self._save_loggers_on_train_batch_end()

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # progress global step according to grads progress
            self.global_step += 1

    def on_run_end(self) -> None:
        """Calls the on_epoch_end hook.

        Returns:
            The output of each training step for each optimizer

        Raises:
            MisconfigurationException: ``train_epoch_end`` does not return ``None``
        """
        # inform logger the batch loop has finished
        self.trainer.logger_connector.epoch_end_reached()

        # get the model and call model.training_epoch_end
        model = self.trainer.lightning_module
        if is_overridden("training_epoch_end", model) and self._outputs:
            epoch_end_outputs = self._prepare_outputs_training_epoch_end(
                self._outputs,
                automatic=model.automatic_optimization,
                num_optimizers=len(self.trainer.optimizers),
            )
            # check that the dataloader/iterator produced a batch
            # FIXME: still necessary?
            if epoch_end_outputs:
                # run training_epoch_end
                # refresh the result for custom logging at the epoch level
                model._current_fx_name = "training_epoch_end"

                # lightning module hook
                epoch_end_outputs = model.training_epoch_end(epoch_end_outputs)
                if epoch_end_outputs is not None:
                    raise MisconfigurationException(
                        "training_epoch_end expects a return of None. "
                        "HINT: remove the return statement in training_epoch_end"
                    )
        # free memory
        self._outputs = []

        self.trainer.fit_loop.epoch_progress.increment_processed()

        # call train epoch end hooks
        self.trainer.call_hook("on_train_epoch_end")
        self.trainer.call_hook("on_epoch_end")
        self.trainer.logger_connector.on_epoch_end()

        if self._num_training_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=True)

        self.dataloader_iter = None

    def teardown(self) -> None:
        self._results.cpu()
        self.batch_loop.teardown()
        self.val_loop.teardown()

    def _run_validation(self):
        # reload dataloaders
        self.val_loop.reload_evaluation_dataloaders()

        with torch.no_grad():
            self.val_loop.run()

    def _accumulated_batches_reached(self) -> bool:
        """Determine if accumulation will be finished by the end of the current batch."""
        return self.batch_progress.current.ready % self.trainer.accumulate_grad_batches == 0

    def _num_training_batches_reached(self) -> bool:
        """Checks if we are in the last batch or if there are more batches to follow."""
        return self.batch_progress.current.ready == self.trainer.num_training_batches or self.is_last_batch

    def _should_accumulate(self) -> bool:
        """Checks if the optimizer step should be performed or gradients should be accumulated for the current
        step."""
        accumulation_done = self._accumulated_batches_reached()
        # Lightning steps on the final batch
        is_final_batch = self._num_training_batches_reached()
        # but the TTP might not
        ttp_accumulates_on_final_batch = (
            self.trainer.training_type_plugin.handles_gradient_accumulation or not is_final_batch
        )
        return not accumulation_done and ttp_accumulates_on_final_batch

    @staticmethod
    def _prepare_outputs_training_batch_end(
        batch_output: _BATCH_OUTPUTS_TYPE,
        automatic: bool,
        num_optimizers: int,
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Processes the outputs from the batch loop into the format passed to the ``training_batch_end`` hook.

        ``(tbptt_steps, n_opt) -> (n_opt, tbptt_steps)``. The optimizer dimension might have been squeezed.
        """
        if not batch_output:
            return []

        # convert optimizer dicts to list
        if automatic:
            batch_output = apply_to_collection(
                batch_output, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )
        array = np.array(batch_output, dtype=object)
        if array.ndim == 1:
            array = np.expand_dims(array, 1)

        n_splits, _ = array.shape

        array = array.transpose((0, 1))
        array = array.squeeze()
        array = array.tolist()

        # remove residual empty lists
        array = [item for item in array if isinstance(item, list) and len(item)]

        array = _recursive_unpad(array)

        # in case we squeezed from 1-element array to a 0-dim array
        array = array if isinstance(array, list) else [array]
        return array

    @staticmethod
    def _prepare_outputs_training_epoch_end(
        batch_outputs: _OUTPUTS_TYPE,
        automatic: bool,
        num_optimizers: int,
    ) -> Union[List[List[List[Dict[str, Any]]]], List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Processes the outputs from the batch loop into the format passed to the ``training_epoch_end`` hook.

        ``(n_batches, tbptt_steps, n_opt) -> (n_opt, n_batches, tbptt_steps)``.
        All single-element dimensions might have been squeezed.

        This processing is necessary because the format of the inputs to the ``training_epoch_end`` hook does not
        match the loop structure and because empty dimensions are squeezed. This could break with loop customization.
        """
        # `batch_outputs` (plural) is the same as `epoch_end_output` (singular)
        if not batch_outputs:
            return []

        # convert optimizer dicts to list
        if automatic:
            batch_outputs = apply_to_collection(
                batch_outputs, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )
        array = np.array(batch_outputs, dtype=object)
        if array.ndim == 2:
            array = np.expand_dims(array, 2)

        n_batches, n_splits, _ = array.shape

        array = array.transpose((2, 0, 1))
        array = array.squeeze()
        array = array.tolist()

        # remove residual empty lists
        array = [item for item in array if isinstance(item, list) and len(item)]

        array = _recursive_unpad(array)

        # in case we squeezed from 1-element array to a 0-dim array
        array = array if isinstance(array, list) else [array]
        return array

    def update_lr_schedulers(self, interval: str, update_plateau_schedulers: bool) -> None:
        """updates the lr schedulers based on the given interval."""
        if interval == "step" and self._should_accumulate():
            return
        self.trainer.optimizer_connector.update_learning_rates(
            interval=interval,
            update_plateau_schedulers=update_plateau_schedulers,
            opt_indices=[opt_idx for opt_idx, _ in self.batch_loop.get_active_optimizers(self.total_batch_idx)],
        )

    def _should_check_val_fx(self, batch_idx: int, is_last_batch: bool) -> bool:
        """Decide if we should run validation."""
        if not self.trainer.enable_validation:
            return False

        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        if not is_val_check_epoch:
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
        # when loggers should save to disk
        should_flush_logs = self.trainer.logger_connector.should_flush_logs
        if should_flush_logs and self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.save()


def _convert_optim_dict(outs: Dict[int, Dict[str, Any]], num_optimizers: int) -> List[Dict[str, Any]]:
    # relevant test: test_step_scheduling_for_multiple_optimizers_with_frequency
    return [outs[opt_idx] if opt_idx in outs else None for opt_idx in range(num_optimizers)]


def _recursive_unpad(nested: List, value: Optional[Any] = None) -> List:
    if not isinstance(nested, list):
        return nested

    return [_recursive_unpad(item) for item in nested if item is not value]
