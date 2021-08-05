from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

from deprecate import void

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.utilities.warnings import WarningCache


class PredictionEpochLoop(Loop):
    """Loop performing prediction on arbitrary sequentially used dataloaders."""

    def __init__(self) -> None:
        super().__init__()
        self.return_predictions: bool = False
        self.predictions: List[Any] = []
        self.current_batch_indices: List[int] = []
        self.batch_progress = Progress()

        self._dl_max_batches: Optional[int] = None
        self._num_dataloaders: Optional[int] = None
        self._warning_cache = WarningCache()
        self._all_batch_indices: List[int] = []

    @property
    def done(self) -> bool:
        """Ends prediction when the iteration count exceeds the total number of available batches"""
        return self.batch_progress.current.completed >= self._dl_max_batches

    @property
    def should_store_predictions(self) -> bool:
        """Whether the predictions should be stored for later usage (e.g. aggregation or returning)"""
        any_pred = any(cb.interval.on_epoch for cb in self.trainer.prediction_writer_callbacks)
        return self.return_predictions or any_pred

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def reset(self) -> None:
        """Resets the loops internal state"""
        self._all_batch_indices: List[int] = []
        self.predictions: List[Any] = []
        self.batch_progress.current.reset()

    def on_run_start(
        self,
        dataloader_iter: Iterator,
        dataloader_idx: int,
        dl_max_batches: int,
        num_dataloaders: int,
        return_predictions: bool = False,
    ) -> None:
        """
        Prepares the loops internal state

        Args:
            dataloader_iter: the iterator over the current dataloader
            dataloader_idx: the index of the current dataloader
            dl_max_batches: the maximum number of batches the current loader can produce
            num_dataloaders: the total number of dataloaders
            return_predictions: whether to return the obtained predictions
        """
        void(dataloader_iter, dataloader_idx)
        self._dl_max_batches = dl_max_batches
        self._num_dataloaders = num_dataloaders
        self.return_predictions = return_predictions

    def advance(
        self,
        dataloader_iter: Iterator,
        dataloader_idx: int,
        dl_max_batches: int,
        num_dataloaders: int,
        return_predictions: bool = False,
    ) -> None:
        """
        Runs one prediction step.

        Args:
            dataloader_iter: the iterator over the current dataloader
            dataloader_idx: the index of the current dataloader
            dl_max_batches: the maximum number of batches the current loader can produce
            num_dataloaders: the total number of dataloaders
            return_predictions: whether to return the obtained predictions
        """
        batch_idx, batch = next(dataloader_iter)
        if batch is None:
            raise StopIteration

        with self.trainer.profiler.profile("predict_batch_to_device"):
            batch = self.trainer.accelerator.batch_to_device(batch, dataloader_idx=dataloader_idx)

        self.batch_progress.increment_ready()

        with self.trainer.profiler.profile("predict_step"):
            self._predict_step(batch, batch_idx, dataloader_idx)

    def on_run_end(self) -> Tuple[Any, Any]:
        """Returns the predictions and the corresponding batch indices"""
        predictions = self.predictions
        all_batch_indices = self._all_batch_indices
        # free memory
        self.predictions = []
        self._all_batch_indices = []
        return predictions, all_batch_indices

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Runs the actual predict step together with all the
        necessary bookkeeping and the hooks tied to the predict step.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        # extract batch_indices and store them
        self._store_batch_indices(dataloader_idx)

        model_ref = self.trainer.lightning_module

        self.trainer.call_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_started()

        model_ref._current_fx_name = "predict_step"
        predictions = self.trainer.accelerator.predict_step(step_kwargs)

        self.batch_progress.increment_processed()

        if predictions is None:
            self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.trainer.call_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_completed()

        if self.should_store_predictions:
            self.predictions.append(predictions)

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Any]:
        """
        Assembles the keyword arguments for the ``predict_step``

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

    def _store_batch_indices(self, dataloader_idx: int) -> None:
        """Stores the batch indices if the predictions should be stored"""
        batch_sampler = self.trainer.predict_dataloaders[dataloader_idx].batch_sampler
        if isinstance(batch_sampler, IndexBatchSamplerWrapper):
            self.current_batch_indices = batch_sampler.batch_indices
            if self.should_store_predictions:
                self._all_batch_indices.append(batch_sampler.batch_indices)
