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

from collections import OrderedDict
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Union

from deprecate import void

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.utilities import _update_dataloader_iter
from pytorch_lightning.trainer.progress import BatchProgress
from pytorch_lightning.utilities.auto_restart import MergedIteratorState, reload_dataloader_state_dict
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class EvaluationEpochLoop(Loop):
    """This is the loop performing the evaluation.

    It mainly loops over the given dataloader and runs the validation or test step (depending on the trainer's current
    state).
    """

    def __init__(self) -> None:
        super().__init__()
        self.outputs: EPOCH_OUTPUT = []
        self.batch_progress = BatchProgress()

        self._dl_max_batches: Optional[int] = None
        self._num_dataloaders: Optional[int] = None
        self._dataloader_iter: Optional[Iterator] = None
        self._data_fetcher: Optional[DataFetcher] = None
        self._dataloader_state_dict: Dict[str, Any] = None

    @property
    def done(self) -> bool:
        """Returns ``True`` if the current iteration count reaches the number of dataloader batches."""
        return self.batch_progress.current.completed >= self._dl_max_batches

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def reset(self) -> None:
        """Resets the loop's internal state."""
        self._dl_max_batches = None
        self._num_dataloaders = None
        self._data_fetcher = None
        self.outputs = []

        if not self.restarting:
            self.batch_progress.reset_on_run()
        else:
            self.batch_progress.reset_on_restart()

    def on_run_start(
        self, data_fetcher: AbstractDataFetcher, dataloader_idx: int, dl_max_batches: int, num_dataloaders: int
    ) -> None:
        """Adds the passed arguments to the loop's state if necessary.

        Args:
            data_fetcher: the current data_fetcher wrapping the dataloader
            dataloader_idx: index of the current dataloader
            dl_max_batches: maximum number of batches the dataloader can produce
            num_dataloaders: the total number of dataloaders
        """
        void(dataloader_idx)
        self._dl_max_batches = dl_max_batches
        self._num_dataloaders = num_dataloaders
        self._data_fetcher = data_fetcher

        self._reload_dataloader_state_dict(data_fetcher)
        self._dataloader_iter = _update_dataloader_iter(data_fetcher, self.batch_progress.current.ready)

    def advance(
        self, data_fetcher: AbstractDataFetcher, dataloader_idx: int, dl_max_batches: int, num_dataloaders: int
    ) -> None:
        """Calls the evaluation step with the corresponding hooks and updates the logger connector.

        Args:
            data_fetcher: iterator over the dataloader
            dataloader_idx: index of the current dataloader
            dl_max_batches: maximum number of batches the dataloader can produce
            num_dataloaders: the total number of dataloaders

        Raises:
            StopIteration: If the current batch is None
        """
        void(data_fetcher, dl_max_batches, num_dataloaders)

        batch_idx, (batch, self.batch_progress.is_last_batch) = next(self._dataloader_iter)

        if batch is None:
            raise StopIteration

        if not self.trainer._data_connector.evaluation_data_fetcher.store_on_device:
            with self.trainer.profiler.profile("evaluation_batch_to_device"):
                batch = self.trainer.accelerator.batch_to_device(batch, dataloader_idx=dataloader_idx)

        self.batch_progress.increment_ready()

        # hook
        self._on_evaluation_batch_start(batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_started()

        # lightning module methods
        with self.trainer.profiler.profile("evaluation_step_and_end"):
            output = self._evaluation_step(batch, batch_idx, dataloader_idx)
            output = self._evaluation_step_end(output)

        self.batch_progress.increment_processed()

        # track loss history
        self._on_evaluation_batch_end(output, batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_completed()

        # log batch metrics
        self.trainer.logger_connector.update_eval_step_metrics()

        # track epoch level outputs
        if self._should_track_batch_outputs_for_epoch_end() and output is not None:
            self.outputs.append(output)

        if self.trainer.move_metrics_to_cpu:
            # the evaluation step output is not moved as they are not considered "metrics"
            assert self.trainer._results is not None
            self.trainer._results.cpu()

        if not self.batch_progress.is_last_batch:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> EPOCH_OUTPUT:
        """Returns the outputs of the whole run."""
        outputs = self.outputs
        # free memory
        self.outputs = []
        self._dataloader_iter = None
        self._data_fetcher = None
        return outputs

    def teardown(self) -> None:
        # in case the model changes
        self._should_track_batch_outputs_for_epoch_end.cache_clear()

    def on_save_checkpoint(self) -> Dict:
        state_dict = super().on_save_checkpoint()

        if (
            self._data_fetcher is None
            or self._num_completed_batches_reached()  # did not finish
            # TODO: fault-tolerance requires a minimum number of batches so probably should be > 0
            or self.batch_progress.current.ready == 0  # did not start
        ):
            return state_dict

        # TODO: this should use `pytorch_lightning/trainer/supporters.py::CombinedLoader._state_dict_fn`
        state_to_save = "state" if self._has_completed() else "previous_state"
        state: Optional[MergedIteratorState] = getattr(self._data_fetcher.dataloader_iter, state_to_save, None)
        if state:
            state_dict["dataloader_state_dict"] = asdict(state)
        return state_dict

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        # cache the dataloader state dict until the dataloader objects are available
        self._dataloader_state_dict = state_dict.get("dataloader_state_dict")

    def _reload_dataloader_state_dict(self, data_fetcher: AbstractDataFetcher):
        if not self.trainer.sanity_checking and self._dataloader_state_dict:
            reload_dataloader_state_dict(data_fetcher.dataloader, self._dataloader_state_dict)
            self._dataloader_state_dict = None

    def _num_completed_batches_reached(self) -> bool:
        epoch_finished_on_completed = self.batch_progress.current.completed == self._dl_max_batches
        dataloader_consumed_successfully = self.batch_progress.is_last_batch and self._has_completed()
        return epoch_finished_on_completed or dataloader_consumed_successfully

    def _has_completed(self) -> bool:
        return self.batch_progress.current.ready == self.batch_progress.current.completed

    def _evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Optional[STEP_OUTPUT]:
        """The evaluation step (validation_step or test_step depending on the trainer's state).

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the outputs of the step
        """
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        if self.trainer.testing:
            self.trainer.lightning_module._current_fx_name = "test_step"
            with self.trainer.profiler.profile("test_step"):
                output = self.trainer.accelerator.test_step(step_kwargs)
        else:
            self.trainer.lightning_module._current_fx_name = "validation_step"
            with self.trainer.profiler.profile("validation_step"):
                output = self.trainer.accelerator.validation_step(step_kwargs)

        return output

    def _evaluation_step_end(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """Calls the `{validation/test}_step_end` hook."""
        hook_name = "test_step_end" if self.trainer.testing else "validation_step_end"
        output = self.trainer.call_hook(hook_name, *args, **kwargs)
        return output

    def _on_evaluation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Calls the ``on_{validation/test}_batch_start`` hook.

        Args:
            batch: The current batch to run through the step
            batch_idx: The index of the current batch
            dataloader_idx: The index of the dataloader producing the current batch

        Raises:
            AssertionError: If the number of dataloaders is None (has not yet been set).
        """
        self.trainer.logger_connector.on_batch_start(batch_idx, batch)

        assert self._num_dataloaders is not None
        self.trainer.logger_connector.on_evaluation_batch_start(dataloader_idx, self._num_dataloaders)

        if self.trainer.testing:
            self.trainer.call_hook("on_test_batch_start", batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook("on_validation_batch_start", batch, batch_idx, dataloader_idx)

    def _on_evaluation_batch_end(
        self, output: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """The ``on_{validation/test}_batch_end`` hook.

        Args:
            output: The output of the performed step
            batch: The input batch for the step
            batch_idx: The index of the current batch
            dataloader_idx: Index of the dataloader producing the current batch
        """
        hook_name = "on_test_batch_end" if self.trainer.testing else "on_validation_batch_end"
        self.trainer.call_hook(hook_name, output, batch, batch_idx, dataloader_idx)

        self.trainer.logger_connector.on_batch_end()

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Union[Any, int]]:
        """Helper function to build the arguments for the current step.

        Args:
            batch: The current batch to run through the step
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the keyword arguments to pass to the step function
        """
        # make dataloader_idx arg in validation_step optional
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])

        multiple_val_loaders = not self.trainer.testing and self._num_dataloaders > 1
        multiple_test_loaders = self.trainer.testing and self._num_dataloaders > 1

        if multiple_test_loaders or multiple_val_loaders:
            step_kwargs["dataloader_idx"] = dataloader_idx

        return step_kwargs

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage."""
        model = self.trainer.lightning_module
        if self.trainer.testing:
            return is_overridden("test_epoch_end", model)
        return is_overridden("validation_epoch_end", model)
