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
from collections import ChainMap, OrderedDict
from typing import Any, Iterable, List, Optional, Tuple, Union

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor

from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.progress import BatchProgress
from lightning.pytorch.loops.utilities import _no_grad_context, _select_data_fetcher
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import SIGTERMException

if _RICH_AVAILABLE:
    from rich import get_console
    from rich.table import Column, Table


class _EvaluationLoop(_Loop):
    """Top-level loop where validation/testing starts."""

    def __init__(self, verbose: bool = True, inference_mode: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.inference_mode = inference_mode
        self.batch_progress = BatchProgress()  # across dataloaders

        self._results = _ResultCollection(training=False)
        self._logged_outputs: List[_OUT_DICT] = []
        self._has_run: bool = False
        self._data_fetcher: Optional[_DataFetcher] = None

    @property
    def num_dataloaders(self) -> int:
        """Returns the number of prediction dataloaders."""
        combined_loader = self.trainer.test_dataloaders if self.trainer.testing else self.trainer.val_dataloaders
        assert combined_loader is not None
        return len(combined_loader._flattened)

    @property
    def current_dataloader_idx(self) -> int:
        """Returns the index of the current dataloader."""
        combined_loader = self.trainer.test_dataloaders if self.trainer.testing else self.trainer.val_dataloaders
        assert combined_loader is not None
        if combined_loader._mode != "sequential":
            raise ValueError(f'`{type(self).__name__}` only supports the `CombinedLoader(mode="sequential")` mode.')
        if combined_loader._iterator is None:
            raise RuntimeError("The iterator has not been created yet.")
        return combined_loader._iterator._iterator_idx

    @property
    def current_dataloader(self) -> Iterable:
        """Returns the current dataloader."""
        combined_loader = self.trainer.test_dataloaders if self.trainer.testing else self.trainer.val_dataloaders
        assert combined_loader is not None
        return combined_loader._flattened[self.current_dataloader_idx]

    @property
    def max_batches(self) -> List[Union[int, float]]:
        """The max number of batches this loop will run for each dataloader."""
        if self.trainer.testing:
            return self.trainer.num_test_batches
        elif self.trainer.sanity_checking:
            return self.trainer.num_sanity_val_batches
        elif self.trainer.validating:
            return self.trainer.num_val_batches
        raise RuntimeError(f"Unexpected stage: {self.trainer.state.stage}")

    @property
    def skip(self) -> bool:
        """Returns whether the evaluation should be skipped."""
        return sum(self.max_batches) == 0

    @_no_grad_context
    def run(self) -> List[_OUT_DICT]:
        if self.skip:
            return []
        self.reset()
        self.on_run_start()
        while True:
            try:
                if isinstance(self._data_fetcher, _DataLoaderIterDataFetcher):
                    batch_idx = self._data_fetcher._dataloader._iterator._idx
                    batch = next(self._data_fetcher)
                else:
                    batch_idx, batch = next(self._data_fetcher)
                self.batch_progress.is_last_batch = self._data_fetcher.done
                self._evaluation_step(batch, batch_idx, self.current_dataloader_idx)
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        self.on_advance_end()
        return self.on_run_end()

    def reset(self) -> None:
        """Resets the internal state of the loop."""
        trainer = self.trainer

        self._has_run = False
        self._logged_outputs = []
        if not self.restarting:
            self.batch_progress.reset_on_run()
        else:
            self.batch_progress.reset_on_restart()
        # when restarting, if we are running `validate` or `test` twice, since there's no concept of `max_epochs` we
        # need to reset the current state when the loop has finished running
        if trainer.state.fn != TrainerFn.FITTING:
            self.batch_progress.reset_on_run()

        data_fetcher = _select_data_fetcher(trainer)
        if isinstance(data_fetcher, _DataLoaderIterDataFetcher) and self.num_dataloaders > 1:
            raise NotImplementedError(
                "Using `dataloader_iter` in your step method is not supported with multiple dataloaders"
            )
        combined_loader = trainer.test_dataloaders if trainer.testing else trainer.val_dataloaders
        assert combined_loader is not None
        data_fetcher.setup(combined_loader)
        iter(data_fetcher)  # creates the iterator inside the fetcher
        # set the per-dataloader limits
        combined_loader._iterator.limits = self.max_batches
        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready
        data_fetcher._start_profiler = self._on_before_fetch
        data_fetcher._stop_profiler = self._on_after_fetch
        self._data_fetcher = data_fetcher

    def on_run_start(self) -> None:
        """Runs the ``_on_evaluation_model_eval``, ``_on_evaluation_start`` and ``_on_evaluation_epoch_start``
        hooks."""
        self._on_evaluation_model_eval()
        self.trainer.lightning_module.zero_grad()
        self._on_evaluation_start()
        self._on_evaluation_epoch_start()

    def on_advance_end(self) -> None:
        trainer = self.trainer
        if not trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True

        trainer._logger_connector.epoch_end_reached()

        self._logged_outputs.append(trainer._logger_connector.update_eval_epoch_metrics())

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
            assert self.trainer.state.stage is not None
            self._print_results(logged_outputs, self.trainer.state.stage)

        return logged_outputs

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None
        self._results.cpu()

    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        if self.trainer.testing:
            self.trainer.reset_test_dataloader()
        elif self.trainer.val_dataloaders is None or self.trainer._data_connector._should_reload_val_dl:
            self.trainer.reset_val_dataloader()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks."""
        trainer = self.trainer

        assert self._results is not None
        self._results.to(device=trainer.lightning_module.device)

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

    def _on_before_fetch(self) -> None:
        stage = self.trainer.state.stage
        assert stage is not None
        stage = stage.dataloader_prefix
        self.trainer.profiler.start(f"[{type(self).__name__}].{stage}_next")

    def _on_after_fetch(self) -> None:
        stage = self.trainer.state.stage
        assert stage is not None
        stage = stage.dataloader_prefix
        # the dataloader_idx cannot be easily included here because it might be different from the index used on
        # profiler start, since the `__next__` call might use a different iterator
        self.trainer.profiler.stop(f"[{type(self).__name__}].{stage}_next")

    def _evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Runs the actual evaluation step together with all the necessary bookkeeping and the hooks tied to it.

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        trainer = self.trainer

        batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = call._call_strategy_hook(trainer, "batch_to_device", batch, dataloader_idx=dataloader_idx)

        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self.num_dataloaders > 1 else None)

        self.batch_progress.increment_ready()

        trainer._logger_connector.on_batch_start(**step_kwargs)

        hook_name = "on_test_batch_start" if trainer.testing else "on_validation_batch_start"
        call._call_callback_hooks(trainer, hook_name, *step_kwargs.values())
        call._call_lightning_module_hook(trainer, hook_name, *step_kwargs.values())

        self.batch_progress.increment_started()

        hook_name = "test_step" if trainer.testing else "validation_step"
        output = call._call_strategy_hook(trainer, hook_name, *step_kwargs.values())

        hook_name = "test_step_end" if trainer.testing else "validation_step_end"
        model_output = call._call_lightning_module_hook(trainer, hook_name, output)
        strategy_output = call._call_strategy_hook(trainer, hook_name, output)
        output = strategy_output if model_output is None else model_output

        self.batch_progress.increment_processed()

        hook_name = "on_test_batch_end" if trainer.testing else "on_validation_batch_end"
        call._call_callback_hooks(trainer, hook_name, output, *step_kwargs.values())
        call._call_lightning_module_hook(trainer, hook_name, output, *step_kwargs.values())

        trainer._logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        # log batch metrics
        if not trainer.sanity_checking:
            trainer._logger_connector.update_eval_step_metrics(batch_idx)

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
