from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Union

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.connectors.logger_connector.result import Result
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.utilities.types import STEP_OUTPUT


class EvaluationLoop(Loop):

    def __init__(self):
        super().__init__()
        self.batch_idx: Optional[int] = None
        self.predictions: Optional[PredictionCollection] = None
        self.dataloader: Optional[Iterator] = None
        self.dl_max_batches: Optional[int] = None
        self.dataloader_idx: Optional[int] = None
        self.num_dataloaders: Optional[int] = None
        self.outputs = []

    def connect(self, trainer, *args, **kwargs):
        super().connect(trainer, *args, **kwargs)

    @property
    def done(self) -> bool:
        return self.batch_idx is not None and self.batch_idx >= self.dl_max_batches

    def reset(self) -> None:
        self.iteration_count = 0
        self.batch_idx = None
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)
        self.dl_max_batches = None
        self.dataloader_idx = None
        self.num_dataloaders = None
        self.outputs = []

    def on_run_start(self, dataloader, dataloader_idx, dl_max_batches, num_dataloaders) -> None:
        self.dl_max_batches = dl_max_batches
        self.dataloader_idx = dataloader_idx
        self.num_dataloaders = num_dataloaders
        self.dataloader = enumerate(dataloader)

    def advance(self, dataloader, dataloader_idx, dl_max_batches, num_dataloaders) -> None:
        batch_idx, batch = next(self.dataloader)

        # TODO: is self.batch_idx needed or can it be set to iteration_count?
        self.batch_idx = batch_idx

        if batch is None:
            raise StopIteration

        if batch_idx >= dl_max_batches:
            raise StopIteration

        # hook
        self.on_evaluation_batch_start(batch, batch_idx, dataloader_idx)

        # lightning module methods
        with self.trainer.profiler.profile("evaluation_step_and_end"):
            output = self.evaluation_step(batch, batch_idx, dataloader_idx)
            output = self.evaluation_step_end(output)

        # hook + store predictions
        self.on_evaluation_batch_end(output, batch, batch_idx, dataloader_idx)

        # log batch metrics
        self.trainer.logger_connector.log_evaluation_step_metrics()

        # track epoch level outputs
        self.outputs = self.trainer._track_output_for_epoch_end(self.outputs, output)

    def on_run_end(self) -> Any:
        return self.outputs

    def __getstate__(self):
        # avoid pickling errors "cannot pickle generator object"
        self.dataloader = None
        return self.__dict__


# ------------------------------------------------------------------------------------------------------------
# HELPER --- TO BE CLEANED UP
# ------------------------------------------------------------------------------------------------------------

    def evaluation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Optional[STEP_OUTPUT]:
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        model_ref = self.trainer.lightning_module
        model_ref._results = Result()

        if self.trainer.testing:
            model_ref._current_fx_name = "test_step"
            with self.trainer.profiler.profile("test_step"):
                output = self.trainer.accelerator.test_step(step_kwargs)
        else:
            model_ref._current_fx_name = "validation_step"
            with self.trainer.profiler.profile("validation_step"):
                output = self.trainer.accelerator.validation_step(step_kwargs)

        # capture any logged information
        self.trainer.logger_connector.cache_logged_metrics()
        # track batch size for weighted average
        if isinstance(output, Result):
            output.track_batch_size(batch)

        return output

    def evaluation_step_end(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        if self.trainer.testing:
            output = self.trainer.call_hook('test_step_end', *args, **kwargs)
        else:
            output = self.trainer.call_hook('validation_step_end', *args, **kwargs)
        return output

    def on_evaluation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # set dataloader_idx to model and track batch_size
        self.trainer.logger_connector.on_evaluation_batch_start(batch, dataloader_idx, self.num_dataloaders)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_start', batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_start', batch, batch_idx, dataloader_idx)

    def on_evaluation_batch_end(
        self,
        output: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_end', output, batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_end', output, batch, batch_idx, dataloader_idx)

        # store predicitons if do_write_predictions and track eval loss history
        self.store_predictions(output, batch_idx, dataloader_idx)

    def store_predictions(self, output: Optional[STEP_OUTPUT], batch_idx: int, dataloader_idx: int) -> None:
        # Add step predictions to prediction collection to write later
        if output is not None and self.predictions is not None:
            if isinstance(output, Result) and self.trainer.testing:
                self.predictions.add(output.pop('predictions', None))

        # track debug metrics
        self.trainer.dev_debugger.track_eval_loss_history(batch_idx, dataloader_idx, output)

    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Union[Any, int]]:
        # make dataloader_idx arg in validation_step optional
        step_kwargs = OrderedDict([('batch', batch), ('batch_idx', batch_idx)])

        multiple_val_loaders = (not self.trainer.testing and self.num_dataloaders > 1)
        multiple_test_loaders = (self.trainer.testing and self.num_dataloaders > 1)

        if multiple_test_loaders or multiple_val_loaders:
            step_kwargs['dataloader_idx'] = dataloader_idx

        return step_kwargs
