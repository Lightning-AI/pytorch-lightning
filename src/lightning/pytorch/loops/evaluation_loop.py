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
import os
import shutil
import sys
from collections import ChainMap, defaultdict, OrderedDict
from typing import Any, DefaultDict, Iterable, List, Optional, Tuple, Union

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor

import lightning.pytorch as pl
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.progress import _BatchProgress
from lightning.pytorch.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.data_connector import (
    _check_dataloader_iterable,
    _DataLoaderSource,
    _parse_num_batches,
    _process_dataloader,
    _request_dataloader,
    _resolve_overfit_batches,
)
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.combined_loader import _Sequential, CombinedLoader
from lightning.pytorch.utilities.data import has_len_all_ranks
from lightning.pytorch.utilities.exceptions import SIGTERMException
from lightning.pytorch.utilities.model_helpers import is_overridden

if _RICH_AVAILABLE:
    from rich import get_console
    from rich.table import Column, Table


class _EvaluationLoop(_Loop):
    """Top-level loop where validation/testing starts."""

    def __init__(
        self,
        trainer: "pl.Trainer",
        trainer_fn: TrainerFn,
        stage: RunningStage,
        verbose: bool = True,
        inference_mode: bool = True,
    ) -> None:
        super().__init__(trainer)
        self.verbose = verbose
        self.inference_mode = inference_mode
        self.batch_progress = _BatchProgress()  # across dataloaders
        #  list in "sequential" mode, number otherwise
        self._max_batches: Union[int, float, List[Union[int, float]]] = []

        self._results = _ResultCollection(training=False)
        self._logged_outputs: List[_OUT_DICT] = []
        self._has_run: bool = False
        self._trainer_fn = trainer_fn
        self._stage = stage
        self._data_source = _DataLoaderSource(None, f"{stage.dataloader_prefix}_dataloader")
        self._combined_loader: Optional[CombinedLoader] = None
        self._data_fetcher: Optional[_DataFetcher] = None
        self._seen_batches_per_dataloader: DefaultDict[int, int] = defaultdict(int)
        self._last_val_dl_reload_epoch = float("-inf")

    @property
    def num_dataloaders(self) -> int:
        """Returns the number of prediction dataloaders."""
        combined_loader = self._combined_loader
        assert combined_loader is not None
        return len(combined_loader.flattened)

    @property
    def max_batches(self) -> Union[int, float, List[Union[int, float]]]:
        """In "sequential" mode, the max number of batches to run per dataloader.

        Otherwise, the max batches to run.

        """
        max_batches = self._max_batches
        if not self.trainer.sanity_checking:
            return max_batches
        sanity_val_steps = self.trainer.num_sanity_val_steps
        if isinstance(max_batches, list):
            return [min(sanity_val_steps, batches) for batches in max_batches]
        return min(sanity_val_steps, max_batches)

    @property
    def skip(self) -> bool:
        """Returns whether the evaluation should be skipped."""
        return sum(self.max_batches) == 0 if isinstance(self.max_batches, list) else self.max_batches == 0

    @property
    def _should_reload_val_dl(self) -> bool:
        """Check if validation dataloader should be reloaded."""
        n_epochs = self.trainer.reload_dataloaders_every_n_epochs
        return bool(n_epochs and self.trainer.current_epoch - self._last_val_dl_reload_epoch >= n_epochs)

    @property
    def _is_sequential(self) -> bool:
        assert self._combined_loader is not None
        return self._combined_loader._mode == "sequential"

    @_no_grad_context
    def run(self) -> List[_OUT_DICT]:
        self.setup_data()
        if self.skip:
            return []
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None
        previous_dataloader_idx = 0
        while True:
            try:
                if self._is_sequential:
                    batch, batch_idx, dataloader_idx = next(data_fetcher)

                    if previous_dataloader_idx != dataloader_idx:
                        # the dataloader has changed, notify the logger connector
                        self._store_dataloader_outputs()
                    previous_dataloader_idx = dataloader_idx
                else:
                    batch_idx = (
                        data_fetcher.fetched
                        if isinstance(data_fetcher, _DataLoaderIterDataFetcher)
                        else self.batch_progress.current.ready
                    )
                    batch = next(data_fetcher)
                    dataloader_idx = 0
                self.batch_progress.is_last_batch = data_fetcher.done

                # run step hooks
                self._evaluation_step(batch, batch_idx, dataloader_idx)
            except StopIteration:
                # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
                break
            finally:
                self._restarting = False
        self._store_dataloader_outputs()
        return self.on_run_end()

    def setup_data(self) -> None:
        trainer = self.trainer
        trainer_fn = self._trainer_fn
        if self._combined_loader is not None and trainer_fn == "fit" and not self._should_reload_val_dl:
            return

        pl_module = trainer.lightning_module
        limit_batches = trainer.limit_test_batches if trainer.testing else trainer.limit_val_batches
        hook_name = "test_step" if trainer.testing else "validation_step"
        if limit_batches == 0 or not is_overridden(hook_name, pl_module):
            return

        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        # it should not reload again if it has already reloaded during sanity_check
        if trainer_fn == "fit" and (
            (trainer.sanity_checking and trainer.fit_loop.epoch_loop._should_check_val_epoch())
            or not trainer.sanity_checking
        ):
            self._last_val_dl_reload_epoch = trainer.current_epoch

        stage = self._stage
        source = self._data_source
        dataloaders = _request_dataloader(source)
        trainer.strategy.barrier(f"{stage.dataloader_prefix}_dataloader()")

        if not isinstance(dataloaders, CombinedLoader):
            combined_loader = CombinedLoader(dataloaders, "sequential")
        else:
            combined_loader = dataloaders

        if trainer_fn == "fit" and trainer.overfit_batches > 0:
            _resolve_overfit_batches(combined_loader, stage)

        dataloaders = []
        for dl in combined_loader.flattened:
            _check_dataloader_iterable(dl, source, trainer_fn)
            dl = _process_dataloader(trainer, trainer_fn, stage, dl)
            dataloaders.append(dl)
        combined_loader.flattened = dataloaders
        self._combined_loader = combined_loader

        allow_zero_length = pl_module.allow_zero_length_dataloader_with_multiple_devices
        if trainer.datamodule is not None:
            allow_zero_length |= trainer.datamodule.allow_zero_length_dataloader_with_multiple_devices

        if self._is_sequential:
            self._max_batches = []
            for dl in combined_loader.flattened:
                # determine number of batches
                length = len(dl) if has_len_all_ranks(dl, trainer.strategy, allow_zero_length) else float("inf")
                limit_batches = getattr(trainer, f"limit_{stage.dataloader_prefix}_batches")
                num_batches = _parse_num_batches(stage, length, limit_batches)
                self._max_batches.append(num_batches)
        else:
            has_len_all_ranks_ = has_len_all_ranks(combined_loader, trainer.strategy, allow_zero_length)
            self._max_batches = len(combined_loader) if has_len_all_ranks_ else float("inf")

        # this depends on the data used, so reset it too
        self._seen_batches_per_dataloader = defaultdict(int)

    def reset(self) -> None:
        """Resets the internal state of the loop."""
        trainer = self.trainer

        self._has_run = False
        self._logged_outputs = []

        if not self.restarting:
            self.batch_progress.reset_on_run()
        else:
            self.batch_progress.reset_on_restart()
        fn = trainer.state.fn
        assert fn is not None
        # when restarting, if we are running `validate` or `test` twice, since there's no concept of `max_epochs` we
        # need to reset the current state when the loop has finished running
        if fn != TrainerFn.FITTING:
            self.batch_progress.reset_on_run()

        data_fetcher = _select_data_fetcher(trainer)
        if isinstance(data_fetcher, _DataLoaderIterDataFetcher) and self.num_dataloaders > 1:
            raise NotImplementedError(
                "Using `dataloader_iter` in your step method is not supported with multiple dataloaders"
            )
        combined_loader = self._combined_loader
        assert combined_loader is not None

        if fn == TrainerFn.FITTING:
            for i, dl in enumerate(combined_loader.flattened):
                # some users want validation shuffling based on the training progress
                _set_sampler_epoch(dl, trainer.fit_loop.epoch_progress.current.processed)

        data_fetcher.setup(combined_loader)
        iter(data_fetcher)  # creates the iterator inside the fetcher
        if isinstance(combined_loader._iterator, _Sequential):
            # set the per-dataloader limits
            max_batches = self.max_batches
            assert isinstance(max_batches, list)
            combined_loader._iterator.limits = max_batches

        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready
        data_fetcher._start_profiler = self._on_before_fetch
        data_fetcher._stop_profiler = self._on_after_fetch
        self._data_fetcher = data_fetcher

    def on_run_start(self) -> None:
        """Runs the ``_on_evaluation_model_eval``, ``_on_evaluation_start`` and ``_on_evaluation_epoch_start``
        hooks."""
        self._verify_dataloader_idx_requirement()

        self._on_evaluation_model_eval()
        self.trainer.lightning_module.zero_grad()
        self._on_evaluation_start()
        self._on_evaluation_epoch_start()

    def on_run_end(self) -> List[_OUT_DICT]:
        """Runs the ``_on_evaluation_epoch_end`` hook."""
        # if `done` returned True before any iterations were done, this won't have been called in `on_advance_end`
        self.trainer._logger_connector.epoch_end_reached()
        self.trainer._logger_connector._evaluation_epoch_end()

        # hook
        self._on_evaluation_epoch_end()

        logged_outputs, self._logged_outputs = self._logged_outputs, []  # free memory
        # include any logged outputs on epoch_end
        epoch_end_logged_outputs = self.trainer._logger_connector.update_eval_epoch_metrics()
        all_logged_outputs = dict(ChainMap(*logged_outputs))  # list[dict] -> dict
        all_logged_outputs.update(epoch_end_logged_outputs)
        for dl_outputs in logged_outputs:
            dl_outputs.update(epoch_end_logged_outputs)

        # log metrics
        self.trainer._logger_connector.log_eval_end_metrics(all_logged_outputs)

        # hook
        self._on_evaluation_end()

        # enable train mode again
        self._on_evaluation_model_train()

        if self.verbose and self.trainer.is_global_zero:
            self._print_results(logged_outputs, self._stage)

        return logged_outputs

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None
        self._results.cpu()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        trainer = self.trainer

        hook_name = "on_test_start" if trainer.testing else "on_validation_start"
        call._call_callback_hooks(trainer, hook_name, *args, **kwargs)
        call._call_lightning_module_hook(trainer, hook_name, *args, **kwargs)
        call._call_strategy_hook(trainer, hook_name, *args, **kwargs)

    def _on_evaluation_model_eval(self) -> None:
        """Sets model to eval mode."""
        trainer = self.trainer
        hook_name = "on_test_model_eval" if trainer.testing else "on_validation_model_eval"
        call._call_lightning_module_hook(trainer, hook_name)

    def _on_evaluation_model_train(self) -> None:
        """Sets model to train mode."""
        trainer = self.trainer
        hook_name = "on_test_model_train" if trainer.testing else "on_validation_model_train"
        call._call_lightning_module_hook(trainer, hook_name)

    def _on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_end`` hook."""
        trainer = self.trainer
        hook_name = "on_test_end" if trainer.testing else "on_validation_end"
        call._call_callback_hooks(trainer, hook_name, *args, **kwargs)
        call._call_lightning_module_hook(trainer, hook_name, *args, **kwargs)
        call._call_strategy_hook(trainer, hook_name, *args, **kwargs)

        # reset the logger connector state
        trainer._logger_connector.reset_results()

    def _on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs the ``on_{validation/test}_epoch_start`` hooks."""
        trainer = self.trainer

        trainer._logger_connector.on_epoch_start()

        hook_name = "on_test_epoch_start" if trainer.testing else "on_validation_epoch_start"
        call._call_callback_hooks(trainer, hook_name, *args, **kwargs)
        call._call_lightning_module_hook(trainer, hook_name, *args, **kwargs)

    def _on_evaluation_epoch_end(self) -> None:
        """Runs ``on_{validation/test}_epoch_end`` hook."""
        trainer = self.trainer

        hook_name = "on_test_epoch_end" if trainer.testing else "on_validation_epoch_end"
        call._call_callback_hooks(trainer, hook_name)
        call._call_lightning_module_hook(trainer, hook_name)

        trainer._logger_connector.on_epoch_end()

    def _store_dataloader_outputs(self) -> None:
        trainer = self.trainer
        trainer._logger_connector.epoch_end_reached()
        self._logged_outputs.append(trainer._logger_connector.update_eval_epoch_metrics())

    def _on_before_fetch(self) -> None:
        self.trainer.profiler.start(f"[{type(self).__name__}].{self._stage.dataloader_prefix}_next")

    def _on_after_fetch(self) -> None:
        # the dataloader_idx cannot be easily included here because it might be different from the index used on
        # profiler start, since the `__next__` call might use a different iterator
        self.trainer.profiler.stop(f"[{type(self).__name__}].{self._stage.dataloader_prefix}_next")

    def _evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Runs the actual evaluation step together with all the necessary bookkeeping and the hooks tied to it.

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        """
        trainer = self.trainer

        batch = trainer.precision_plugin.convert_input(batch)
        batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = call._call_strategy_hook(trainer, "batch_to_device", batch, dataloader_idx=dataloader_idx)

        step_kwargs = self._build_kwargs(
            batch, batch_idx, dataloader_idx if self._is_sequential and self.num_dataloaders > 1 else None
        )

        self.batch_progress.increment_ready()

        trainer._logger_connector.on_batch_start(**step_kwargs)

        hook_name = "on_test_batch_start" if trainer.testing else "on_validation_batch_start"
        call._call_callback_hooks(trainer, hook_name, *step_kwargs.values())
        call._call_lightning_module_hook(trainer, hook_name, *step_kwargs.values())

        self.batch_progress.increment_started()

        hook_name = "test_step" if trainer.testing else "validation_step"
        output = call._call_strategy_hook(trainer, hook_name, *step_kwargs.values())

        self.batch_progress.increment_processed()

        hook_name = "on_test_batch_end" if trainer.testing else "on_validation_batch_end"
        call._call_callback_hooks(trainer, hook_name, output, *step_kwargs.values())
        call._call_lightning_module_hook(trainer, hook_name, output, *step_kwargs.values())

        trainer._logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        if not trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

            # log batch metrics
            trainer._logger_connector.update_eval_step_metrics(self._seen_batches_per_dataloader[dataloader_idx])
            self._seen_batches_per_dataloader[dataloader_idx] += 1

        if not self.batch_progress.is_last_batch and trainer.received_sigterm:
            raise SIGTERMException

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> OrderedDict:
        """Helper method to build the arguments for the current step.

        Args:
            batch: the current batch to run through the step.
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch. None if not multiple dataloaders.

        Returns:
            the dictionary containing all the keyboard arguments for the step

        """
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])
        if dataloader_idx is not None:
            step_kwargs["dataloader_idx"] = dataloader_idx
        return step_kwargs

    def _verify_dataloader_idx_requirement(self) -> None:
        trainer = self.trainer
        step_hook = "test_step" if trainer.testing else "validation_step"
        batch_start_hook = "on_test_batch_start" if trainer.testing else "on_validation_batch_start"
        batch_end_hook = "on_test_batch_end" if trainer.testing else "on_validation_batch_end"
        assert self._combined_loader is not None
        _verify_dataloader_idx_requirement(
            (step_hook, batch_start_hook, batch_end_hook),
            self._is_sequential and self.num_dataloaders > 1,
            self._stage,
            trainer.lightning_module,
        )

    @staticmethod
    def _get_keys(data: dict) -> Iterable[Tuple[str, ...]]:
        for k, v in data.items():
            if isinstance(v, dict):
                for new_key in apply_to_collection(v, dict, _EvaluationLoop._get_keys):
                    yield (k, *new_key)  # this need to be in parenthesis for older python versions
            else:
                yield k,

    @staticmethod
    def _find_value(data: dict, target: Iterable[str]) -> Optional[Any]:
        target_start, *rest = target
        if target_start not in data:
            return None
        result = data[target_start]
        if not rest:
            return result
        return _EvaluationLoop._find_value(result, rest)

    @staticmethod
    def _print_results(results: List[_OUT_DICT], stage: str) -> None:
        # remove the dl idx suffix
        results = [{k.split("/dataloader_idx_")[0]: v for k, v in result.items()} for result in results]
        metrics_paths = {k for keys in apply_to_collection(results, dict, _EvaluationLoop._get_keys) for k in keys}
        if not metrics_paths:
            return

        metrics_strs = [":".join(metric) for metric in metrics_paths]
        # sort both lists based on metrics_strs
        metrics_strs, metrics_paths = zip(*sorted(zip(metrics_strs, metrics_paths)))

        headers = [f"DataLoader {i}" for i in range(len(results))]

        # fallback is useful for testing of printed output
        term_size = shutil.get_terminal_size(fallback=(120, 30)).columns or 120
        max_length = int(min(max(len(max(metrics_strs, key=len)), len(max(headers, key=len)), 25), term_size / 2))

        rows: List[List[Any]] = [[] for _ in metrics_paths]

        for result in results:
            for metric, row in zip(metrics_paths, rows):
                val = _EvaluationLoop._find_value(result, metric)
                if val is not None:
                    if isinstance(val, Tensor):
                        val = val.item() if val.numel() == 1 else val.tolist()
                    row.append(f"{val}")
                else:
                    row.append(" ")

        # keep one column with max length for metrics
        num_cols = int((term_size - max_length) / max_length)

        for i in range(0, len(headers), num_cols):
            table_headers = headers[i : (i + num_cols)]
            table_rows = [row[i : (i + num_cols)] for row in rows]

            table_headers.insert(0, f"{stage} Metric".capitalize())

            if _RICH_AVAILABLE:
                columns = [Column(h, justify="center", style="magenta", width=max_length) for h in table_headers]
                columns[0].style = "cyan"

                table = Table(*columns)
                for metric, row in zip(metrics_strs, table_rows):
                    row.insert(0, metric)
                    table.add_row(*row)

                console = get_console()
                console.print(table)
            else:
                row_format = f"{{:^{max_length}}}" * len(table_headers)
                half_term_size = int(term_size / 2)

                try:
                    # some terminals do not support this character
                    if sys.stdout.encoding is not None:
                        "─".encode(sys.stdout.encoding)
                except UnicodeEncodeError:
                    bar_character = "-"
                else:
                    bar_character = "─"
                bar = bar_character * term_size

                lines = [bar, row_format.format(*table_headers).rstrip(), bar]
                for metric, row in zip(metrics_strs, table_rows):
                    # deal with column overflow
                    if len(metric) > half_term_size:
                        while len(metric) > half_term_size:
                            row_metric = metric[:half_term_size]
                            metric = metric[half_term_size:]
                            lines.append(row_format.format(row_metric, *row).rstrip())
                        lines.append(row_format.format(metric, " ").rstrip())
                    else:
                        lines.append(row_format.format(metric, *row).rstrip())
                lines.append(bar)
                print(os.linesep.join(lines))
