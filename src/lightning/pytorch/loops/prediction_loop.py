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
from typing import Any, Iterator, List, Optional, Union

import torch
from lightning_utilities import WarningCache

import lightning.pytorch as pl
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.progress import _Progress
from lightning.pytorch.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.data_connector import (
    _check_dataloader_iterable,
    _DataLoaderSource,
    _parse_num_batches,
    _process_dataloader,
    _request_dataloader,
)
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.data import has_len_all_ranks
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import _ModuleMode
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature
from lightning.pytorch.utilities.types import _PREDICT_OUTPUT


class _PredictionLoop(_Loop):
    """Top-level loop where prediction starts."""

    def __init__(self, trainer: "pl.Trainer", inference_mode: bool = True) -> None:
        super().__init__(trainer)
        self.inference_mode = inference_mode
        # dataloaders x batches x samples. used by PredictionWriter
        self.epoch_batch_indices: List[List[List[int]]] = []
        self.current_batch_indices: List[int] = []  # used by PredictionWriter
        self.batch_progress = _Progress()  # across dataloaders
        self.max_batches: List[Union[int, float]] = []

        self._warning_cache = WarningCache()
        self._data_source = _DataLoaderSource(None, "predict_dataloader")
        self._combined_loader: Optional[CombinedLoader] = None
        self._data_fetcher: Optional[_DataFetcher] = None
        self._results = None  # for `trainer._results` access
        self._predictions: List[List[Any]] = []  # dataloaders x batches
        self._return_predictions = False
        self._module_mode = _ModuleMode()

    @property
    def return_predictions(self) -> bool:
        """Whether to return the predictions or not."""
        return self._return_predictions

    @return_predictions.setter
    def return_predictions(self, return_predictions: Optional[bool] = None) -> None:
        # Strategies that spawn or fork don't support returning predictions
        return_supported = not isinstance(self.trainer.strategy.launcher, _MultiProcessingLauncher)
        if return_predictions and not return_supported:
            raise MisconfigurationException(
                "`return_predictions` should be set to `False` when using the strategies that spawn or fork."
                f" Found {return_predictions} with strategy {type(self.trainer.strategy)}."
            )
        # For strategies that support it, `return_predictions` is True by default unless user decide otherwise.
        self._return_predictions = return_supported if return_predictions is None else return_predictions

    @property
    def predictions(self) -> List[Any]:
        """The cached predictions."""
        if self._predictions == []:
            return self._predictions
        return self._predictions[0] if self.num_dataloaders == 1 else self._predictions

    @property
    def num_dataloaders(self) -> int:
        """Returns the number of prediction dataloaders."""
        combined_loader = self._combined_loader
        assert combined_loader is not None
        return len(combined_loader.flattened)

    @property
    def skip(self) -> bool:
        return sum(self.max_batches) == 0

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        self.setup_data()
        if self.skip:
            return None
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None
        while True:
            try:
                if isinstance(data_fetcher, _DataLoaderIterDataFetcher):
                    dataloader_iter = next(data_fetcher)
                    # hook's batch_idx and dataloader_idx arguments correctness cannot be guaranteed in this setting
                    batch = data_fetcher._batch
                    batch_idx = data_fetcher._batch_idx
                    dataloader_idx = data_fetcher._dataloader_idx
                else:
                    dataloader_iter = None
                    batch, batch_idx, dataloader_idx = next(data_fetcher)
                self.batch_progress.is_last_batch = data_fetcher.done
                # run step hooks
                self._predict_step(batch, batch_idx, dataloader_idx, dataloader_iter)
            except StopIteration:
                # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
                break
            finally:
                self._restarting = False
        return self.on_run_end()

    def setup_data(self) -> None:
        trainer = self.trainer
        # a default `predict_step` exists in the LightningModule, so no need to check if it's overridden
        if trainer.limit_predict_batches == 0:
            return

        source = self._data_source
        dataloaders = _request_dataloader(source)
        trainer.strategy.barrier("predict_dataloader()")

        if not isinstance(dataloaders, CombinedLoader):
            combined_loader = CombinedLoader(dataloaders, "sequential")
        else:
            combined_loader = dataloaders

        allow_zero_length = trainer.lightning_module.allow_zero_length_dataloader_with_multiple_devices
        if trainer.datamodule is not None:
            allow_zero_length |= trainer.datamodule.allow_zero_length_dataloader_with_multiple_devices

        trainer_fn = TrainerFn.PREDICTING
        stage = RunningStage.PREDICTING
        dataloaders = []
        self.max_batches = []
        for dl in combined_loader.flattened:
            _check_dataloader_iterable(dl, source, trainer_fn)
            dl = _process_dataloader(trainer, trainer_fn, stage, dl)
            dataloaders.append(dl)

            # determine number of batches
            length = len(dl) if has_len_all_ranks(dl, trainer.strategy, allow_zero_length) else float("inf")
            num_batches = _parse_num_batches(stage, length, trainer.limit_predict_batches)
            self.max_batches.append(num_batches)
        combined_loader.flattened = dataloaders
        self._combined_loader = combined_loader

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        self.batch_progress.reset_on_run()

        assert self.trainer.state.stage is not None
        data_fetcher = _select_data_fetcher(self.trainer, self.trainer.state.stage)
        combined_loader = self._combined_loader
        assert combined_loader is not None
        if combined_loader._mode != "sequential":
            raise ValueError('`trainer.predict()` only supports the `CombinedLoader(mode="sequential")` mode.')

        # set the per-dataloader limits
        combined_loader.limits = self.max_batches
        data_fetcher.setup(combined_loader)
        iter(data_fetcher)  # creates the iterator inside the fetcher

        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready
        data_fetcher._start_profiler = self._on_before_fetch
        data_fetcher._stop_profiler = self._on_after_fetch
        self._data_fetcher = data_fetcher

        num_dataloaders = self.num_dataloaders
        self.epoch_batch_indices = [[] for _ in range(num_dataloaders)]
        self._predictions = [[] for _ in range(num_dataloaders)]

    def on_run_start(self) -> None:
        """Calls ``_on_predict_model_eval``, ``_on_predict_start`` and ``_on_predict_epoch_start`` hooks."""
        self._verify_dataloader_idx_requirement()
        self._on_predict_model_eval()
        self._on_predict_start()
        self._on_predict_epoch_start()

    def on_run_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` and ``on_predict_end`` hooks and returns results from all dataloaders."""
        results = self._on_predict_epoch_end()
        self._on_predict_end()
        self._on_predict_model_train()
        return results

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None

    def _predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int, dataloader_iter: Optional[Iterator]
    ) -> None:
        """Runs the actual predict step together with all the necessary bookkeeping and the hooks tied to it.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: The index of the current batch.
            dataloader_idx: the index of the dataloader producing the current batch.
            dataloader_iter: The iterator if using this step flavor.

        """
        trainer = self.trainer
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None

        if not (using_dataloader_iter := isinstance(data_fetcher, _DataLoaderIterDataFetcher)):
            batch = trainer.precision_plugin.convert_input(batch)
            batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
            batch = call._call_strategy_hook(trainer, "batch_to_device", batch, dataloader_idx=dataloader_idx)

        self.batch_progress.increment_ready()

        if not using_dataloader_iter:
            any_on_epoch = self._store_data_for_prediction_writer(batch_idx, dataloader_idx)

        # the `_step` methods don't take a batch_idx when `dataloader_iter` is used, but all other hooks still do,
        # so we need different kwargs
        hook_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self.num_dataloaders > 1 else None)

        call._call_callback_hooks(trainer, "on_predict_batch_start", *hook_kwargs.values())
        call._call_lightning_module_hook(trainer, "on_predict_batch_start", *hook_kwargs.values())

        self.batch_progress.increment_started()

        # configure step_kwargs
        step_args = (
            self._build_step_args_from_hook_kwargs(hook_kwargs, "predict_step")
            if not using_dataloader_iter
            else (dataloader_iter,)
        )
        predictions = call._call_strategy_hook(trainer, "predict_step", *step_args)
        if predictions is None:
            self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.batch_progress.increment_processed()

        if using_dataloader_iter:
            # update the hook kwargs now that the step method might have consumed the iterator
            batch = data_fetcher._batch
            batch_idx = data_fetcher._batch_idx
            dataloader_idx = data_fetcher._dataloader_idx
            hook_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self.num_dataloaders > 1 else None)

        call._call_callback_hooks(trainer, "on_predict_batch_end", predictions, *hook_kwargs.values())
        call._call_lightning_module_hook(trainer, "on_predict_batch_end", predictions, *hook_kwargs.values())

        self.batch_progress.increment_completed()

        if self._return_predictions or any_on_epoch:
            self._predictions[dataloader_idx].append(move_data_to_device(predictions, torch.device("cpu")))

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> OrderedDict:
        """Assembles the keyword arguments for the ``predict_step``

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch.
            dataloader_idx: the index of the dataloader producing the current batch. None if not multiple dataloaders
                in sequential mode.

        Returns:
            the dictionary containing all the keyboard arguments for the predict step

        """
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])
        if dataloader_idx is not None:
            step_kwargs["dataloader_idx"] = dataloader_idx
        return step_kwargs

    def _build_step_args_from_hook_kwargs(self, hook_kwargs: OrderedDict, step_hook_name: str) -> tuple:
        """Helper method to build args for `predict_step`."""
        kwargs = hook_kwargs.copy()
        step_hook_fx = getattr(self.trainer.lightning_module, step_hook_name)
        if not is_param_in_hook_signature(step_hook_fx, "batch_idx", min_args=2):
            kwargs.pop("batch_idx", None)
        return tuple(kwargs.values())

    def _get_batch_indices(self, dataloader: object) -> List[List[int]]:  # batches x samples
        """Returns a reference to the seen batch indices if the dataloader has a batch sampler wrapped by our
        :class:`~lightning.pytorch.overrides.distributed._IndexBatchSamplerWrapper`."""
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if not isinstance(batch_sampler, _IndexBatchSamplerWrapper):
            self._warning_cache.warn(
                f"Couldn't infer the batch indices fetched from your dataloader: `{type(dataloader).__name__}`"
            )
            return []
        return batch_sampler.seen_batch_indices

    def _store_data_for_prediction_writer(self, batch_idx: int, dataloader_idx: int) -> bool:
        prediction_writers = [cb for cb in self.trainer.callbacks if isinstance(cb, BasePredictionWriter)]
        any_on_epoch = any(cb.interval.on_epoch for cb in prediction_writers)
        any_on_batch = any(cb.interval.on_batch for cb in prediction_writers)
        if any_on_batch or any_on_epoch:
            combined_loader = self._combined_loader
            assert combined_loader is not None
            dataloader = combined_loader.flattened[dataloader_idx]
            batch_indices = self._get_batch_indices(dataloader)
            if not batch_indices:
                # this is only available with `_IndexBatchSamplerWrapper`, but it's only used on DataLoaders, if this is
                # reached, it's likely because a non-DataLoader was passed
                return any_on_epoch
            batch_indices = batch_indices[batch_idx]
            if any_on_epoch:
                self.epoch_batch_indices[dataloader_idx].append(batch_indices)
            if any_on_batch:
                self.current_batch_indices = batch_indices
        return any_on_epoch

    def _on_before_fetch(self) -> None:
        self.trainer.profiler.start(f"[{type(self).__name__}].predict_next")

    def _on_after_fetch(self) -> None:
        # the dataloader_idx cannot be easily included here because it might be different from the index used on
        # profiler start, since the `__next__` call might use a different iterator
        self.trainer.profiler.stop(f"[{type(self).__name__}].predict_next")

    def _on_predict_start(self) -> None:
        """Calls ``on_predict_start`` hooks."""
        trainer = self.trainer
        call._call_callback_hooks(trainer, "on_predict_start")
        call._call_lightning_module_hook(trainer, "on_predict_start")
        call._call_strategy_hook(trainer, "on_predict_start")

    def _on_predict_model_eval(self) -> None:
        self._module_mode.capture(self.trainer.lightning_module)
        call._call_lightning_module_hook(self.trainer, "on_predict_model_eval")

    def _on_predict_model_train(self) -> None:
        self._module_mode.restore(self.trainer.lightning_module)

    def _on_predict_epoch_start(self) -> None:
        """Calls ``on_predict_epoch_start`` hooks."""
        trainer = self.trainer
        call._call_callback_hooks(trainer, "on_predict_epoch_start")
        call._call_lightning_module_hook(trainer, "on_predict_epoch_start")

    def _on_predict_epoch_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` hook.

        Returns:
            the results for all dataloaders

        """
        trainer = self.trainer
        call._call_callback_hooks(trainer, "on_predict_epoch_end")
        call._call_lightning_module_hook(trainer, "on_predict_epoch_end")

        if self.return_predictions:
            return self.predictions
        return None

    def _on_predict_end(self) -> None:
        """Resets previous gradient status and calls ``on_predict_end`` hook."""
        if not self.return_predictions:
            self._predictions = []
        self.epoch_batch_indices = []

        trainer = self.trainer
        # hook
        call._call_callback_hooks(trainer, "on_predict_end")
        call._call_lightning_module_hook(trainer, "on_predict_end")
        call._call_strategy_hook(trainer, "on_predict_end")

    def _verify_dataloader_idx_requirement(self) -> None:
        trainer = self.trainer
        assert self._combined_loader is not None
        _verify_dataloader_idx_requirement(
            ("predict_step",),
            self._combined_loader._mode == "sequential"
            and self.num_dataloaders > 1
            and not isinstance(self._data_fetcher, _DataLoaderIterDataFetcher),
            RunningStage.PREDICTING,
            trainer.lightning_module,
        )
        _verify_dataloader_idx_requirement(
            ("on_predict_batch_start", "on_predict_batch_end"),
            self._combined_loader._mode == "sequential" and self.num_dataloaders > 1,
            RunningStage.PREDICTING,
            trainer.lightning_module,
        )
