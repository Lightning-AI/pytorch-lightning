from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Tuple

import torch

from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.utilities.rank_zero import WarningCache

warning_cache = WarningCache()


class PredictionEpochLoop(Loop):
    """Loop performing prediction on arbitrary sequentially used dataloaders."""

    def __init__(self) -> None:
        super().__init__()
        self.return_predictions = False
        self.predictions: List[Any] = []
        self.current_batch_indices: List[int] = []
        self.batch_progress = Progress()

        self._dl_max_batches = 0
        self._num_dataloaders = 0
        self._warning_cache = WarningCache()
        self._seen_batch_indices: List[List[int]] = []

    @property
    def done(self) -> bool:
        """Ends prediction when the iteration count exceeds the total number of available batches."""
        return self.batch_progress.current.completed >= self._dl_max_batches

    @property
    def should_store_predictions(self) -> bool:
        """Whether the predictions should be stored for later usage (e.g. aggregation or returning)"""
        any_pred = any(cb.interval.on_epoch for cb in self.trainer.prediction_writer_callbacks)
        return self.return_predictions or any_pred

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def reset(self) -> None:
        """Resets the loops internal state."""
        self._seen_batch_indices = []
        self.predictions = []
        self.batch_progress.reset_on_run()

    def on_run_start(
        self,
        dataloader_iter: Iterator,
        dataloader_idx: int,
        dl_max_batches: int,
        num_dataloaders: int,
    ) -> None:
        """Prepares the loops internal state.

        Args:
            dataloader_iter: the iterator over the current dataloader
            dataloader_idx: the index of the current dataloader
            dl_max_batches: the maximum number of batches the current loader can produce
            num_dataloaders: the total number of dataloaders
        """
        self._dl_max_batches = dl_max_batches
        self._num_dataloaders = num_dataloaders
        # this call requires that `self.return_predictions` is set
        self._seen_batch_indices = self._get_batch_indices(dataloader_idx) if self.should_store_predictions else []

    def advance(
        self,
        dataloader_iter: Iterator,
        dataloader_idx: int,
        dl_max_batches: int,
        num_dataloaders: int,
    ) -> None:
        """Runs one prediction step.

        Args:
            dataloader_iter: the iterator over the current dataloader
            dataloader_idx: the index of the current dataloader
            dl_max_batches: the maximum number of batches the current loader can produce
            num_dataloaders: the total number of dataloaders
        """
        action_name = f"[{self.__class__.__name__}].predict_dataloader_idx_{dataloader_idx}_next"
        with self.trainer.profiler.profile(action_name):
            batch_idx, batch = next(dataloader_iter)
        self._seen_batch_indices = self._get_batch_indices(dataloader_idx) if self.should_store_predictions else []
        # we need to truncate the list of batch indices due to prefetching in the dataloader and Lightning
        self._seen_batch_indices = self._seen_batch_indices[: (self.batch_progress.current.completed + 1)]

        if batch is None:
            raise StopIteration

        batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
        batch = self.trainer._call_strategy_hook("batch_to_device", batch, dataloader_idx=dataloader_idx)

        self.batch_progress.increment_ready()

        self._predict_step(batch, batch_idx, dataloader_idx)

    def on_run_end(self) -> Tuple[List[Any], List[List[int]]]:
        """Returns the predictions and the corresponding batch indices."""
        predictions, all_batch_indices = self.predictions, self._seen_batch_indices
        self.predictions, self._seen_batch_indices = [], []  # free memory
        return predictions, all_batch_indices

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Runs the actual predict step together with all the necessary bookkeeping and the hooks tied to the
        predict step.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        # extract batch_indices and store them
        batch_indices = self._get_batch_indices(dataloader_idx)
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

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Any]:
        """Assembles the keyword arguments for the ``predict_step``

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the dictionary containing all the keyboard arguments for the predict step
        """
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])
        if self._num_dataloaders > 1:
            step_kwargs["dataloader_idx"] = dataloader_idx
        return step_kwargs

    def _get_batch_indices(self, dataloader_idx: int) -> List[List[int]]:
        """Returns a reference to the seen batch indices if the dataloader has a batch sampler wrapped by our
        :class:`~pytorch_lightning.overrides.distributed.IndexBatchSamplerWrapper`."""
        # the batch_sampler is not be defined in case of CombinedDataLoaders
        assert self.trainer.predict_dataloaders
        batch_sampler = getattr(
            self.trainer.predict_dataloaders[dataloader_idx],
            "batch_sampler",
            None,
        )
        if isinstance(batch_sampler, IndexBatchSamplerWrapper):
            return batch_sampler.seen_batch_indices

        warning_cache.warn("Lightning couldn't infer the indices fetched for your dataloader.")
        return []
