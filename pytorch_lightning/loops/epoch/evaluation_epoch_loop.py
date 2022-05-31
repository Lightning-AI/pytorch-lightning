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
from functools import lru_cache
from typing import Any, Dict, Optional

from deprecate import void
from torch.utils.data import DataLoader

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.progress import BatchProgress
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.auto_restart import (
    _collect_states_on_rank_zero_over_collection,
    _reload_dataloader_state_dict,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class EvaluationEpochLoop(Loop):
    """This is the loop performing the evaluation.

    It mainly loops over the given dataloader and runs the validation or test step (depending on the trainer's current
    state).
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_progress = BatchProgress()

        self._outputs: EPOCH_OUTPUT = []
        self._dl_max_batches = 0
        self._data_fetcher: Optional[AbstractDataFetcher] = None
        self._dataloader_state_dict: Dict[str, Any] = {}
        self._dl_batch_idx = [0]

    @property
    def done(self) -> bool:
        """Returns ``True`` if the current iteration count reaches the number of dataloader batches."""
        return self.batch_progress.current.completed >= self._dl_max_batches

    def reset(self) -> None:
        """Resets the loop's internal state."""
        self._dl_max_batches = 0
        self._data_fetcher = None
        self._outputs = []

        if not self.restarting:
            self.batch_progress.reset_on_run()
        else:
            self.batch_progress.reset_on_restart()
        # when restarting, if we are running `validate` or `test` twice, since there's no concept of `max_epochs` we
        # need to reset the current state when the loop has finished running
        if self.done and self.trainer.state.fn != TrainerFn.FITTING:
            self.batch_progress.reset_on_run()

    def on_run_start(  # type: ignore[override]
        self, data_fetcher: AbstractDataFetcher, dl_max_batches: int, kwargs: OrderedDict
    ) -> None:
        """Adds the passed arguments to the loop's state if necessary.

        Args:
            data_fetcher: the current data_fetcher wrapping the dataloader
            dl_max_batches: maximum number of batches the dataloader can produce
            kwargs: the kwargs passed down to the hooks.
        """
        void(kwargs)
        self._dl_max_batches = dl_max_batches
        self._reload_dataloader_state_dict(data_fetcher)
        # creates the iterator inside the fetcher but returns `self`
        self._data_fetcher = iter(data_fetcher)
        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready

    def advance(  # type: ignore[override]
        self,
        data_fetcher: AbstractDataFetcher,
        dl_max_batches: int,
        kwargs: OrderedDict,
    ) -> None:
        """Calls the evaluation step with the corresponding hooks and updates the logger connector.

        Args:
            data_fetcher: iterator over the dataloader
            dl_max_batches: maximum number of batches the dataloader can produce
            kwargs: the kwargs passed down to the hooks.

        Raises:
            StopIteration: If the current batch is None
        """
        void(dl_max_batches)

        if not isinstance(data_fetcher, DataLoaderIterDataFetcher):
            batch_idx = self.batch_progress.current.ready
            batch = next(data_fetcher)
        else:
            batch_idx, batch = next(data_fetcher)
        self.batch_progress.is_last_batch = data_fetcher.done

        # configure step_kwargs
        kwargs = self._build_kwargs(kwargs, batch, batch_idx)

        self.batch_progress.increment_ready()

        # hook
        self._on_evaluation_batch_start(**kwargs)

        self.batch_progress.increment_started()

        # lightning module methods
        output = self._evaluation_step(**kwargs)
        output = self._evaluation_step_end(output)

        self.batch_progress.increment_processed()

        # track loss history
        self._on_evaluation_batch_end(output, **kwargs)

        self.batch_progress.increment_completed()

        # log batch metrics
        if not self.trainer.sanity_checking:
            dataloader_idx = kwargs.get("dataloader_idx", 0)
            self.trainer._logger_connector.update_eval_step_metrics(self._dl_batch_idx[dataloader_idx])
            self._dl_batch_idx[dataloader_idx] += 1

        # track epoch level outputs
        if self._should_track_batch_outputs_for_epoch_end() and output is not None:
            self._outputs.append(output)

        if self.trainer.move_metrics_to_cpu:
            # the evaluation step output is not moved as they are not considered "metrics"
            assert self.trainer._results is not None
            self.trainer._results.cpu()

        if not self.batch_progress.is_last_batch:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> EPOCH_OUTPUT:
        """Returns the outputs of the whole run."""
        outputs, self._outputs = self._outputs, []  # free memory
        self._data_fetcher = None
        return outputs

    def teardown(self) -> None:
        # in case the model changes
        self._should_track_batch_outputs_for_epoch_end.cache_clear()

    def on_save_checkpoint(self) -> Dict:
        state_dict = super().on_save_checkpoint()

        if (
            self.trainer is not None
            and self.trainer.state._fault_tolerant_mode.is_enabled
            and self._data_fetcher is not None
            and not self._num_completed_batches_reached()  # did not finish
            and self.batch_progress.current.ready  # did start
        ):
            state = CombinedLoader._state_dict_fn(self._data_fetcher.dataloader_iter, self._has_completed())
            if state:
                state_dict["dataloader_state_dict"] = _collect_states_on_rank_zero_over_collection(state)

        return state_dict

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        # cache the dataloader state dict until the dataloader objects are available
        # dataset states are collected across all ranks
        dataloader_state_dict = state_dict.get("dataloader_state_dict", None)
        if not _fault_tolerant_training() or not dataloader_state_dict:
            return
        self._dataloader_state_dict = dataloader_state_dict[self.trainer.global_rank]

    def _reload_dataloader_state_dict(self, data_fetcher: AbstractDataFetcher) -> None:
        if self.trainer.sanity_checking or not self._dataloader_state_dict:
            return
        dataloader = data_fetcher.dataloader
        if isinstance(dataloader, CombinedLoader):
            raise MisconfigurationException(
                "Reloading support hasn't been implemented for `CombinedLoader`. You can request it by opening an issue"
                " in `https://github.com/PyTorchLightning/pytorch-lightning/issues`."
            )
        assert isinstance(dataloader, DataLoader)
        _reload_dataloader_state_dict(dataloader, self._dataloader_state_dict)
        self._dataloader_state_dict = {}

    def _num_completed_batches_reached(self) -> bool:
        epoch_finished_on_completed = self.batch_progress.current.completed == self._dl_max_batches
        dataloader_consumed_successfully = self.batch_progress.is_last_batch and self._has_completed()
        return epoch_finished_on_completed or dataloader_consumed_successfully

    def _has_completed(self) -> bool:
        return self.batch_progress.current.ready == self.batch_progress.current.completed

    def _evaluation_step(self, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The evaluation step (validation_step or test_step depending on the trainer's state).

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the outputs of the step
        """
        if self.trainer.testing:
            output = self.trainer._call_strategy_hook("test_step", *kwargs.values())
        else:
            output = self.trainer._call_strategy_hook("validation_step", *kwargs.values())

        return output

    def _evaluation_step_end(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """Calls the `{validation/test}_step_end` hook."""
        hook_name = "test_step_end" if self.trainer.testing else "validation_step_end"
        model_output = self.trainer._call_lightning_module_hook(hook_name, *args, **kwargs)
        strategy_output = self.trainer._call_strategy_hook(hook_name, *args, **kwargs)
        output = strategy_output if model_output is None else model_output
        return output

    def _on_evaluation_batch_start(self, **kwargs: Any) -> None:
        """Calls the ``on_{validation/test}_batch_start`` hook.

        Args:
            batch: The current batch to run through the step
            batch_idx: The index of the current batch
            dataloader_idx: The index of the dataloader producing the current batch

        Raises:
            AssertionError: If the number of dataloaders is None (has not yet been set).
        """
        self.trainer._logger_connector.on_batch_start(**kwargs)

        kwargs.setdefault("dataloader_idx", 0)  # TODO: the argument should be keyword for these
        hook_name = "on_test_batch_start" if self.trainer.testing else "on_validation_batch_start"
        self.trainer._call_callback_hooks(hook_name, *kwargs.values())
        self.trainer._call_lightning_module_hook(hook_name, *kwargs.values())

    def _on_evaluation_batch_end(self, output: Optional[STEP_OUTPUT], **kwargs: Any) -> None:
        """The ``on_{validation/test}_batch_end`` hook.

        Args:
            output: The output of the performed step
            batch: The input batch for the step
            batch_idx: The index of the current batch
            dataloader_idx: Index of the dataloader producing the current batch
        """
        kwargs.setdefault("dataloader_idx", 0)  # TODO: the argument should be keyword for these
        hook_name = "on_test_batch_end" if self.trainer.testing else "on_validation_batch_end"
        self.trainer._call_callback_hooks(hook_name, output, *kwargs.values())
        self.trainer._call_lightning_module_hook(hook_name, output, *kwargs.values())

        self.trainer._logger_connector.on_batch_end()

    def _build_kwargs(self, kwargs: OrderedDict, batch: Any, batch_idx: int) -> OrderedDict:
        """Helper function to build the arguments for the current step.

        Args:
            kwargs: The kwargs passed down to the hooks.
            batch: The current batch to run through the step.

        Returns:
            The kwargs passed down to the hooks.
        """
        kwargs.update({"batch": batch, "batch_idx": batch_idx})
        kwargs.move_to_end("batch_idx", last=False)
        kwargs.move_to_end("batch", last=False)
        return kwargs

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage."""
        model = self.trainer.lightning_module
        if self.trainer.testing:
            return is_overridden("test_epoch_end", model)
        return is_overridden("validation_epoch_end", model)

    def _reset_dl_batch_idx(self, num_dataloaders: int) -> None:
        self._dl_batch_idx = [0] * num_dataloaders
