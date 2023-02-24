from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch
from lightning_utilities import WarningCache

import lightning.pytorch as pl
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.data import _set_sampler_epoch
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.progress import Progress
from lightning.pytorch.loops.utilities import _no_grad_context, _select_data_fetcher
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper
from lightning.pytorch.strategies import DDPSpawnStrategy
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.data_connector import _DataLoaderSource
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.combined_loader import _Sequential, CombinedLoader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import _PREDICT_OUTPUT


class _PredictionLoop(_Loop):
    """Top-level loop where prediction starts."""

    def __init__(self, trainer: "pl.Trainer", inference_mode: bool = True) -> None:
        super().__init__(trainer)
        self.inference_mode = inference_mode
        # dataloaders x batches x samples. used by PredictionWriter
        self.epoch_batch_indices: List[List[List[int]]] = []
        self.current_batch_indices: List[int] = []  # used by PredictionWriter
        self.batch_progress = Progress()  # across dataloaders
        self.max_batches: List[Union[int, float]] = []

        self._warning_cache = WarningCache()
        self._data_source = _DataLoaderSource(None, "predict_dataloader")
        self._combined_loader: Optional[CombinedLoader] = None
        self._data_fetcher: Optional[_DataFetcher] = None
        self._results = None  # for `trainer._results` access
        self._predictions: List[List[Any]] = []  # dataloaders x batches
        self._return_predictions = False

    @property
    def return_predictions(self) -> bool:
        """Whether to return the predictions or not."""
        return self._return_predictions

    @return_predictions.setter
    def return_predictions(self, return_predictions: Optional[bool] = None) -> None:
        # `DDPSpawnStrategy` plugins and derivatives don't support return predictions.
        is_ddp_spawn = isinstance(self.trainer.strategy, DDPSpawnStrategy)
        if return_predictions and is_ddp_spawn:
            raise MisconfigurationException(
                "`return_predictions` should be set to `False` when using the `DDPSpawnStrategy` or children class. "
                f"Found {return_predictions} with strategy {type(self.trainer.strategy)}."
            )
        # For non `DDPSpawnStrategy` plugin, the `return_predictions` is True by default unless user decide otherwise.
        self._return_predictions = not is_ddp_spawn if return_predictions is None else return_predictions

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
                batch, batch_idx, dataloader_idx = next(data_fetcher)
                self.batch_progress.is_last_batch = data_fetcher.done
                self._predict_step(batch, batch_idx, dataloader_idx)
            except StopIteration:
                # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
                break
            finally:
                self._restarting = False
        return self.on_run_end()

    def setup_data(self) -> None:
        trainer = self.trainer
        source = self._data_source
        pl_module = trainer.lightning_module
        # a dfault `predict_step` exists in the LightningModule, so no need to check if it's overridden
        if not source.is_defined() or trainer.limit_predict_batches == 0:
            return

        self.max_batches, combined_loader = trainer._data_connector._reset_eval_dataloader(
            RunningStage.PREDICTING, model=pl_module
        )
        for dl in combined_loader.flattened:
            # some users want prediction shuffling based on the training progress
            _set_sampler_epoch(dl, trainer.fit_loop.epoch_progress.current.processed)
        self._combined_loader = combined_loader

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        self.batch_progress.reset_on_run()

        data_fetcher = _select_data_fetcher(self.trainer)
        if isinstance(data_fetcher, _DataLoaderIterDataFetcher) and self.num_dataloaders > 1:
            raise NotImplementedError(
                "Using `dataloader_iter` in your step method is not supported with multiple dataloaders"
            )
        combined_loader = self._combined_loader
        assert combined_loader is not None
        data_fetcher.setup(combined_loader)
        iter(data_fetcher)  # creates the iterator inside the fetcher
        assert isinstance(combined_loader._iterator, _Sequential)
        # set the per-dataloader limits
        combined_loader._iterator.limits = self.max_batches
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
        trainer = self.trainer
        call._call_lightning_module_hook(trainer, "on_predict_model_eval")
        trainer.lightning_module.zero_grad()
        self._on_predict_start()
        self._on_predict_epoch_start()

    def on_run_end(self) -> Optional[_PREDICT_OUTPUT]:
        """Calls ``on_predict_epoch_end`` and ``on_predict_end`` hooks and returns results from all dataloaders."""
        results = self._on_predict_epoch_end()
        self._on_predict_end()
        return results

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Runs the actual predict step together with all the necessary bookkeeping and the hooks tied to it.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        trainer = self.trainer
        batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = call._call_strategy_hook(trainer, "batch_to_device", batch, dataloader_idx=dataloader_idx)

        self.batch_progress.increment_ready()

        any_on_epoch = self._store_data_for_prediction_writer(batch_idx, dataloader_idx)

        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if self.num_dataloaders > 1 else None)

        call._call_callback_hooks(trainer, "on_predict_batch_start", *step_kwargs.values())
        call._call_lightning_module_hook(trainer, "on_predict_batch_start", *step_kwargs.values())

        self.batch_progress.increment_started()

        # configure step_kwargs
        predictions = call._call_strategy_hook(trainer, "predict_step", *step_kwargs.values())

        self.batch_progress.increment_processed()

        if predictions is None:
            self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        call._call_callback_hooks(trainer, "on_predict_batch_end", predictions, *step_kwargs.values())
        call._call_lightning_module_hook(trainer, "on_predict_batch_end", predictions, *step_kwargs.values())

        self.batch_progress.increment_completed()

        if self._return_predictions or any_on_epoch:
            self._predictions[dataloader_idx].append(move_data_to_device(predictions, torch.device("cpu")))

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int]) -> Dict[str, Any]:
        """Assembles the keyword arguments for the ``predict_step``

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch. None if not multiple dataloaders.

        Returns:
            the dictionary containing all the keyboard arguments for the predict step
        """
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])
        if dataloader_idx is not None:
            step_kwargs["dataloader_idx"] = dataloader_idx
        return step_kwargs

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
