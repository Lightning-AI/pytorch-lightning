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

from collections import OrderedDict
from typing import Any, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.progress import BatchProgress
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import SIGTERMException
from lightning.pytorch.utilities.types import STEP_OUTPUT


class _EvaluationEpochLoop(_Loop):
    """This is the loop performing the evaluation.

    It mainly loops over the given dataloader and runs the validation or test step (depending on the trainer's current
    state).
    """

    def __init__(self, trainer: "pl.Trainer") -> None:
        super().__init__(trainer)
        self.batch_progress = BatchProgress()

        self._dl_max_batches: Union[int, float] = 0
        self._data_fetcher: Optional[_DataFetcher] = None
        self._dl_batch_idx = [0]

    @property
    def done(self) -> bool:
        """Returns ``True`` if the current iteration count reaches the number of dataloader batches."""
        return self.batch_progress.current.completed >= self._dl_max_batches

    def run(self, data_fetcher: _DataFetcher, dl_max_batches: Union[int, float], kwargs: OrderedDict) -> None:
        self.reset()
        self.on_run_start(data_fetcher, dl_max_batches, kwargs)
        while not self.done:
            try:
                self.advance(data_fetcher, kwargs)
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        self.on_run_end()

    def reset(self) -> None:
        """Resets the loop's internal state."""
        self._dl_max_batches = 0
        self._data_fetcher = None

        if not self.restarting:
            self.batch_progress.reset_on_run()
        else:
            self.batch_progress.reset_on_restart()
        # when restarting, if we are running `validate` or `test` twice, since there's no concept of `max_epochs` we
        # need to reset the current state when the loop has finished running
        if self.done and self.trainer.state.fn != TrainerFn.FITTING:
            self.batch_progress.reset_on_run()

    def on_run_start(self, data_fetcher: _DataFetcher, dl_max_batches: Union[int, float], kwargs: OrderedDict) -> None:
        """Adds the passed arguments to the loop's state if necessary.

        Args:
            data_fetcher: the current data_fetcher wrapping the dataloader
            dl_max_batches: maximum number of batches the dataloader can produce
            kwargs: the kwargs passed down to the hooks.
        """
        self._dl_max_batches = dl_max_batches
        # creates the iterator inside the fetcher but returns `self`
        self._data_fetcher = iter(data_fetcher)
        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready

        stage = self.trainer.state.stage
        assert stage is not None
        stage = stage.dataloader_prefix
        self._profiler_fetch_action = (
            f"[{self.__class__.__name__}].{stage}_dataloader_idx_{kwargs.get('dataloader_idx', 0)}_next"
        )
        data_fetcher._start_profiler = self._on_before_fetch
        data_fetcher._stop_profiler = self._on_after_fetch

    def _on_before_fetch(self) -> None:
        self.trainer.profiler.start(self._profiler_fetch_action)

    def _on_after_fetch(self) -> None:
        self.trainer.profiler.stop(self._profiler_fetch_action)

    def advance(
        self,
        data_fetcher: _DataFetcher,
        kwargs: OrderedDict,
    ) -> None:
        """Calls the evaluation step with the corresponding hooks and updates the logger connector.

        Args:
            data_fetcher: iterator over the dataloader
            kwargs: the kwargs passed down to the hooks.

        Raises:
            StopIteration: If the current batch is None
        """
        batch_idx = (
            data_fetcher.fetched
            if isinstance(data_fetcher, _DataLoaderIterDataFetcher)
            else self.batch_progress.current.ready
        )
        batch = next(data_fetcher)
        self.batch_progress.is_last_batch = data_fetcher.done

        dataloader_idx = kwargs.get("dataloader_idx", 0)
        batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = call._call_strategy_hook(self.trainer, "batch_to_device", batch, dataloader_idx=dataloader_idx)

        # configure step_kwargs
        kwargs = self._build_kwargs(kwargs, batch, batch_idx)

        self.batch_progress.increment_ready()

        # hook
        self._on_evaluation_batch_start(**kwargs)

        self.batch_progress.increment_started()

        # step hook
        output = self._evaluation_step(**kwargs)

        self.batch_progress.increment_processed()

        # track loss history
        self._on_evaluation_batch_end(output, **kwargs)

        self.batch_progress.increment_completed()

        # log batch metrics
        if not self.trainer.sanity_checking:
            self.trainer._logger_connector.update_eval_step_metrics(self._dl_batch_idx[dataloader_idx])
            self._dl_batch_idx[dataloader_idx] += 1

        if not self.batch_progress.is_last_batch and self.trainer.received_sigterm:
            raise SIGTERMException

    def on_run_end(self) -> None:
        self._data_fetcher = None

    def _evaluation_step(self, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The evaluation step (validation_step or test_step depending on the trainer's state).

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the outputs of the step
        """
        trainer = self.trainer
        hook_name = "test_step" if trainer.testing else "validation_step"
        return call._call_strategy_hook(trainer, hook_name, *kwargs.values())

    def _on_evaluation_batch_start(self, **kwargs: Any) -> None:
        """Calls the ``on_{validation/test}_batch_start`` hook.

        Args:
            batch: The current batch to run through the step
            batch_idx: The index of the current batch
            dataloader_idx: The index of the dataloader producing the current batch

        Raises:
            AssertionError: If the number of dataloaders is None (has not yet been set).
        """
        trainer = self.trainer
        trainer._logger_connector.on_batch_start(**kwargs)

        hook_name = "on_test_batch_start" if trainer.testing else "on_validation_batch_start"
        call._call_callback_hooks(trainer, hook_name, *kwargs.values())
        call._call_lightning_module_hook(trainer, hook_name, *kwargs.values())

    def _on_evaluation_batch_end(self, output: Optional[STEP_OUTPUT], **kwargs: Any) -> None:
        """The ``on_{validation/test}_batch_end`` hook.

        Args:
            output: The output of the performed step
            batch: The input batch for the step
            batch_idx: The index of the current batch
            dataloader_idx: Index of the dataloader producing the current batch
        """
        trainer = self.trainer

        hook_name = "on_test_batch_end" if trainer.testing else "on_validation_batch_end"
        call._call_callback_hooks(trainer, hook_name, output, *kwargs.values())
        call._call_lightning_module_hook(trainer, hook_name, output, *kwargs.values())

        trainer._logger_connector.on_batch_end()

    def _build_kwargs(self, kwargs: OrderedDict, batch: Any, batch_idx: int) -> OrderedDict:
        """Helper method to build the arguments for the current step.

        Args:
            kwargs: The kwargs passed down to the hooks.
            batch: The current batch to run through the step.

        Returns:
            The kwargs passed down to the hooks.
        """
        kwargs.update(batch=batch, batch_idx=batch_idx)
        # `dataloader_idx` should be last so we need to push these to the front
        kwargs.move_to_end("batch_idx", last=False)
        kwargs.move_to_end("batch", last=False)
        return kwargs

    def _reset_dl_batch_idx(self, num_dataloaders: int) -> None:
        self._dl_batch_idx = [0] * num_dataloaders
