from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.loops.fetchers import _DataFetcher
from lightning.pytorch.loops.loop import _Loop
from lightning.pytorch.loops.progress import Progress
from lightning.pytorch.loops.utilities import _select_data_fetcher
from lightning.pytorch.overrides.distributed import IndexBatchSamplerWrapper
from lightning.pytorch.trainer.supporters import _Sequential, CombinedLoader
from lightning.pytorch.utilities.rank_zero import WarningCache

warning_cache = WarningCache()


class _PredictionEpochLoop(_Loop):
    """Loop performing prediction on arbitrary sequentially used dataloaders."""

    def __init__(self) -> None:
        super().__init__()
        self.return_predictions = False
        self.predictions: List[Any] = []
        self.current_batch_indices: List[int] = []
        self.batch_progress = Progress()

        self._dl_max_batches: Union[int, float] = 0
        self._warning_cache = WarningCache()
        self._seen_batch_indices: List[List[int]] = []
        self._data_fetcher: Optional[_DataFetcher] = None

    @property
    def done(self) -> bool:
        """Ends prediction when the iteration count exceeds the total number of available batches."""
        return self.batch_progress.current.completed >= self._dl_max_batches

    @property
    def should_store_predictions(self) -> bool:
        """Whether the predictions should be stored for later usage (e.g. aggregation or returning)"""
        any_pred = any(cb.interval.on_epoch for cb in self.trainer.prediction_writer_callbacks)
        return self.return_predictions or any_pred

    def run(
        self,
        combined_loader: CombinedLoader,
        dl_max_batches: Union[int, float],
        num_dataloaders: int,
    ) -> Tuple[List[Any], List[List[int]]]:
        assert isinstance(combined_loader._iterator, _Sequential)

        self.reset()
        self.on_run_start(combined_loader, dl_max_batches)

        for batch_idx, batch in combined_loader:
            dataloader_idx = combined_loader._iterator._iterator_idx
            if batch_idx >= dl_max_batches:
                break

            try:
                self.advance(batch, batch_idx, dataloader_idx, num_dataloaders)
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        return self.on_run_end()

    def reset(self) -> None:
        """Resets the loops internal state."""
        self._seen_batch_indices = []
        self.predictions = []
        self.batch_progress.reset_on_run()

    def on_run_start(
        self,
        iterator: Iterator,
        dl_max_batches: Union[int, float],
    ) -> None:
        """Prepares the loops internal state.

        Args:
            iterator: The iterator to wrap.
            dl_max_batches: the maximum number of batches the current loader can produce
        """
        self._dl_max_batches = dl_max_batches
        # this call requires that `self.return_predictions` is set
        self._seen_batch_indices = self._get_batch_indices() if self.should_store_predictions else []

        data_fetcher = _select_data_fetcher(self.trainer)
        # FIXME(carlos): batch_to_device? where is the batch moved?
        data_fetcher.setup(iterator)
        iter(data_fetcher)  # creates the iterator inside the fetcher
        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready
        data_fetcher._start_profiler = self._on_before_fetch
        data_fetcher._stop_profiler = self._on_after_fetch
        self._data_fetcher = data_fetcher

    def advance(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        num_dataloaders: int,
    ) -> None:
        """Runs one prediction step.

        Args:
            batch: FIXME
            batch_idx: FIXME
            dataloader_idx: FIXME
            num_dataloaders: the total number of dataloaders
        """
        self._seen_batch_indices = self._get_batch_indices() if self.should_store_predictions else []
        # we need to truncate the list of batch indices due to prefetching in the dataloader and Lightning
        self._seen_batch_indices = self._seen_batch_indices[: (self.batch_progress.current.completed + 1)]

        batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = self.trainer._call_strategy_hook("batch_to_device", batch, dataloader_idx=dataloader_idx)

        self.batch_progress.increment_ready()

        self._predict_step(batch, batch_idx, dataloader_idx, num_dataloaders)

    def on_run_end(self) -> Tuple[List[Any], List[List[int]]]:
        """Returns the predictions and the corresponding batch indices."""
        predictions, all_batch_indices = self.predictions, self._seen_batch_indices
        self.predictions, self._seen_batch_indices = [], []  # free memory
        return predictions, all_batch_indices

    def teardown(self) -> None:
        if self._data_fetcher is not None:
            self._data_fetcher.teardown()
            self._data_fetcher = None

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int, num_dataloaders: int) -> None:
        """Runs the actual predict step together with all the necessary bookkeeping and the hooks tied to the
        predict step.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
            num_dataloaders: the total number of dataloaders
        """
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx if num_dataloaders > 1 else None)

        # extract batch_indices and store them
        batch_indices = self._get_batch_indices()
        self.current_batch_indices = batch_indices[batch_idx] if batch_indices else []

        self.trainer._call_callback_hooks("on_predict_batch_start", batch, batch_idx, dataloader_idx)
        self.trainer._call_lightning_module_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_started()

        predictions = self.trainer._call_strategy_hook("predict_step", *step_kwargs.values())

        self.batch_progress.increment_processed()

        if predictions is None:
            self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.trainer._call_callback_hooks("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)
        self.trainer._call_lightning_module_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_completed()

        if self.should_store_predictions:
            self.predictions.append(move_data_to_device(predictions, torch.device("cpu")))

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

    def _get_batch_indices(self) -> List[List[int]]:
        """Returns a reference to the seen batch indices if the dataloader has a batch sampler wrapped by our
        :class:`~lightning.pytorch.overrides.distributed.IndexBatchSamplerWrapper`."""
        combined_loader = self.trainer.predict_dataloaders
        assert combined_loader is not None
        # TODO(carlos): avoid referencing parent
        current_dataloader = self.trainer.predict_loop.current_dataloader
        batch_sampler = getattr(current_dataloader, "batch_sampler", None)
        if isinstance(batch_sampler, IndexBatchSamplerWrapper):
            return batch_sampler.seen_batch_indices
        warning_cache.warn("Lightning couldn't infer the indices fetched for your dataloader.")
        return []

    def _on_before_fetch(self) -> None:
        # FIXME(carlos): add dl_idx as before
        self.trainer.profiler.start(f"[{self.__class__.__name__}].predict_dataloader_next")

    def _on_after_fetch(self) -> None:
        # FIXME(carlos): add dl_idx as before
        self.trainer.profiler.stop(f"[{self.__class__.__name__}].predict_dataloader_next")
