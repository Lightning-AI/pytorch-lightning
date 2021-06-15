from typing import Any, List, Optional, Sequence, Tuple, Union

from deprecate.utils import void
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loops.dataloader.dataloader_loop import DataLoaderLoop
from pytorch_lightning.loops.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class EvaluationDataLoaderLoop(DataLoaderLoop):
    """Loops over all dataloaders for evaluation"""

    def __init__(self):
        super().__init__()
        self._dataloaders: Optional[Union[DataLoader, Sequence[DataLoader]]] = None
        self._max_batches: Optional[Union[int, Sequence[int]]] = None
        self.outputs = []
        self.evaluation_loop = EvaluationEpochLoop()

        self._val_results = ResultCollection(training=False)
        self._test_results = ResultCollection(training=False)

    @property
    def num_dataloaders(self) -> int:
        """Returns the total number of dataloaders"""
        return self._get_num_dataloaders(self.dataloaders)

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns all dataloaders"""
        return self._dataloaders

    @property
    def results(self) -> Optional[ResultCollection]:
        """Returns the Results of obtained from all dataloaders"""
        if self.trainer.validating or self.trainer.sanity_checking:
            return self._val_results
        elif self.trainer.testing:
            return self._test_results
        return None

    @property
    def predictions(self):
        """Returns the predictions from all dataloaders"""
        # TODO: fixme
        return self.evaluation_loop.predictions

    def connect(self, trainer: 'pl.Trainer', *args: Any, **kwargs: Any) -> None:
        """Connects the loop to everything necessary (like trainer and accelerators)"""
        super().connect(trainer, *args, **kwargs)

        # TODO: why do we only forward *args and **kwargs here?
        # TODO; Can we make the trainer a weakref/proxy?
        self.evaluation_loop.connect(trainer, *args, **kwargs)

    @property
    def done(self) -> bool:
        """Returns whether all dataloaders are processed or evaluation should be skipped altogether"""
        return (self.current_dataloader_idx >= len(self.dataloaders)) or self.should_skip_evaluation(self._max_batches)

    def reset(self) -> None:
        """Resets the internal state of the loop"""
        self.iteration_count = 0

        # prepare dataloaders
        self._dataloaders, self._max_batches = self.get_eval_dataloaders(), self.get_max_batches()
        # bookkeeping
        self.outputs = []

        if isinstance(self._max_batches, int):
            self._max_batches = [self._max_batches] * len(self._dataloaders)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader"""
        void(*args, **kwargs)
        dataloader = self.trainer.accelerator.process_dataloader(self.current_dataloader)
        dataloader_iter = enumerate(dataloader)
        dl_max_batches = self._max_batches[self.current_dataloader_idx]

        dl_outputs = self.evaluation_loop.run(
            dataloader_iter, self.current_dataloader_idx, dl_max_batches, self.num_dataloaders
        )

        # store batch level output per dataloader
        if self.should_track_batch_outputs_for_epoch_end:
            self.outputs.append(dl_outputs)

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs the ``on_evaluation_start`` and ``on_evaluation_epoch_start`` hooks"""
        void(*args, **kwargs)
        # hook
        self.on_evaluation_start()
        self.on_evaluation_epoch_start()

    def on_run_end(self) -> Any:
        """Runs the ``on_evaluation_epoch_end`` hook"""
        outputs = self.outputs

        # free memory
        self.outputs = []

        # with a single dataloader don't pass a 2D list
        if len(outputs) > 0 and self.num_dataloaders == 1:
            outputs = outputs[0]

        # lightning module method
        self.evaluation_epoch_end(outputs)

        # hook
        self.on_evaluation_epoch_end()

        # log epoch metrics
        eval_loop_results = self.trainer.logger_connector.update_eval_epoch_metrics()

        # hook
        self.on_evaluation_end()

        return eval_loop_results

    def get_max_batches(self) -> List[Union[int, float]]:
        """Returns the max number of batches for each dataloader"""
        # select dataloaders
        if self.trainer.testing:
            max_batches = self.trainer.num_test_batches
        else:
            if self.trainer.sanity_checking:
                self.trainer.num_sanity_val_batches = [
                    min(self.trainer.num_sanity_val_steps, val_batches) for val_batches in self.trainer.num_val_batches
                ]
                max_batches = self.trainer.num_sanity_val_batches
            else:
                max_batches = self.trainer.num_val_batches
        return max_batches

    def get_eval_dataloaders(self) -> List[DataLoader]:
        """Returns the validation or test dataloaders"""
        if self.trainer.testing:
            return self.trainer.test_dataloaders
        return self.trainer.val_dataloaders

    # TODO: remove this method, got split into two above
    def get_evaluation_dataloaders(self) -> Tuple[Optional[List[DataLoader]], List[Union[int, float]]]:
        """Returns all validation/testing dataloaders together with their max_batches.

        Returns:
            a list of validation dataloaders and a list of corresponding max_batches

        """
        model = self.trainer.lightning_module

        # select dataloaders
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)

            dataloaders = self.trainer.test_dataloaders
            max_batches = self.trainer.num_test_batches
        else:
            # val
            if self.trainer.val_dataloaders is None or self.trainer.reload_dataloaders_every_epoch:
                self.trainer.reset_val_dataloader(model)
            if self.trainer.sanity_checking:
                self.trainer.num_sanity_val_batches = [
                    min(self.trainer.num_sanity_val_steps, val_batches) for val_batches in self.trainer.num_val_batches
                ]
                max_batches = self.trainer.num_sanity_val_batches
            else:
                max_batches = self.trainer.num_val_batches
            dataloaders = self.trainer.val_dataloaders
        return dataloaders, max_batches

    # TODO: this is currently also used in the new and old TrainingLoop
    def should_skip_evaluation(self, max_batches: List[Union[int, float]]) -> bool:
        """Whether the evaluation should be skipped"""
        return sum(max_batches) == 0

    def on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_start`` hooks"""
        self.should_track_batch_outputs_for_epoch_end: bool = self._should_track_batch_outputs_for_epoch_end()

        assert self.results is not None
        self.results.to(device=self.trainer.lightning_module.device)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)

    def on_evaluation_model_eval(self) -> None:
        """ Sets model to eval mode"""
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_eval()
        else:
            model_ref.on_validation_model_eval()

    def on_evaluation_model_train(self) -> None:
        """Sets model to train mode"""
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_train()
        else:
            model_ref.on_validation_model_train()

    def on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_{validation/test}_end`` hook"""
        if self.trainer.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

        if self.trainer.state.fn != TrainerFn.FITTING:
            # summarize profile results
            self.trainer.profiler.describe()

        # reset any `torchmetrics.Metric` and the logger connector state
        self.trainer.logger_connector.reset(metrics=True)

    def reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders"""
        model = self.trainer.lightning_module
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)
        else:
            self.trainer.reset_val_dataloader(model)

    def on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_epoch_start`` and ``on_{validation/test}_epoch_start`` hooks"""
        self.trainer.logger_connector.on_epoch_start()
        self.trainer.call_hook('on_epoch_start', *args, **kwargs)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)

    def _get_num_dataloaders(self, dataloaders: Optional[List[DataLoader]]) -> int:
        """Returns the number of dataloaders"""
        # case where user does:
        # return dl1, dl2
        if dataloaders is not None:
            length = len(dataloaders)
            if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
                length = len(dataloaders[0])
            return length
        else:
            return 0

    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage"""
        model = self.trainer.lightning_module
        if self.trainer.testing:
            return is_overridden('test_epoch_end', model)
        else:
            return is_overridden('validation_epoch_end', model)

    def evaluation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Runs ``{validation/test}_epoch_end``"""
        # inform logger the batch loop has finished
        self.trainer.logger_connector.epoch_end_reached()

        # call the model epoch end
        model = self.trainer.lightning_module

        # unset dataloader_idx in model
        model._current_dataloader_idx = None

        if self.trainer.testing:
            if is_overridden('test_epoch_end', model):
                model._current_fx_name = 'test_epoch_end'
                model.test_epoch_end(outputs)

        else:
            if is_overridden('validation_epoch_end', model):
                model._current_fx_name = 'validation_epoch_end'
                model.validation_epoch_end(outputs)

    def store_predictions(self, output: Optional[STEP_OUTPUT], batch_idx: int, dataloader_idx: int) -> None:
        """Stores predictions for later usage (e.g. writing to disk)"""
        # Add step predictions to prediction collection to write later
        if output is not None and self.predictions is not None:
            if isinstance(output, ResultCollection) and self.trainer.testing:
                self.predictions.add(output.pop('predictions', None))

        # track debug metrics
        self.trainer.dev_debugger.track_eval_loss_history(batch_idx, dataloader_idx, output)

    def on_evaluation_epoch_end(self) -> None:
        """Runs ``on_{validation/test}_epoch_end`` hook"""
        hook_name = "on_test_epoch_end" if self.trainer.testing else "on_validation_epoch_end"
        self.trainer.call_hook(hook_name)
        self.trainer.call_hook('on_epoch_end')
        self.trainer.logger_connector.on_epoch_end()
