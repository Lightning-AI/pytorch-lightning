from collections import OrderedDict
from typing import Any, List

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.utilities.warnings import WarningCache


class PredictionLoop(Loop):

    def __init__(self):
        super().__init__()
        self.warning_cache = WarningCache()
        self.dl_max_batches = None
        self.num_dataloaders = None
        self.return_predictions = False
        self.predictions: List[Any] = []
        self.current_batch_indices: [List[int]] = []
        self.all_batch_indices: [List[int]] = []

    @property
    def should_store_predictions(self) -> bool:
        any_pred = any(cb.interval.on_epoch for cb in self.trainer.prediction_writer_callbacks)
        return self.return_predictions or any_pred

    @property
    def done(self) -> bool:
        return self.iteration_count >= self.dl_max_batches

    def reset(self) -> None:
        self.iteration_count = 0
        self.all_batch_indices: List[int] = []
        self.predictions: List[Any] = []

    def on_run_start(self, dataloader, dataloader_idx, dl_max_batches, num_dataloaders, return_predictions=False) -> None:
        self.dl_max_batches = dl_max_batches
        self.num_dataloaders = num_dataloaders
        self.return_predictions = return_predictions

    def advance(self, dataloader_iter, dataloader_idx, dl_max_batches, *args, **kwargs) -> None:
        batch_idx, batch = next(dataloader_iter)
        if batch is None:
            raise StopIteration

        # TODO: needed?
        # stop short when running on limited batches
        if batch_idx >= dl_max_batches:
            raise StopIteration

        # lightning module methods
        with self.trainer.profiler.profile("predict_step"):
            self.predict_step(batch, batch_idx, dataloader_idx)

    def on_run_end(self) -> Any:
        return self.predictions, self.all_batch_indices

# ------------------------------------------------------------------------------------------------------------
# HELPER --- TO BE CLEANED UP
# ------------------------------------------------------------------------------------------------------------

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        # extract batch_indices and store them
        self._store_batch_indices(dataloader_idx)

        model_ref = self.trainer.lightning_module

        self.trainer.call_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

        model_ref._current_fx_name = "predict_step"
        predictions = self.trainer.accelerator.predict_step(step_kwargs)

        if predictions is None:
            self.warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.trainer.call_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        if self.should_store_predictions:
            self.predictions.append(predictions)

    def _build_kwargs(self, batch, batch_idx, dataloader_idx):
        step_kwargs = OrderedDict([('batch', batch), ('batch_idx', batch_idx)])
        if self.num_dataloaders:
            step_kwargs['dataloader_idx'] = dataloader_idx
        return step_kwargs

    def _store_batch_indices(self, dataloader_idx: int) -> None:
        batch_sampler = self.trainer.predict_dataloaders[dataloader_idx].batch_sampler
        if isinstance(batch_sampler, IndexBatchSamplerWrapper):
            self.current_batch_indices = batch_sampler.batch_indices
            if self.should_store_predictions:
                self.all_batch_indices.append(batch_sampler.batch_indices)

