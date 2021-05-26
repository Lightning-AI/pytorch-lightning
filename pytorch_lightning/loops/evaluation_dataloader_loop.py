from weakref import proxy

from pytorch_lightning.loops.base import Loop
from typing import Any, Optional, Sequence, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, DataLoader


class EvaluationDataLoaderLoop(Loop):

    def __init__(self):
        super().__init__()
        self._dataloaders: Optional[Union[DataLoader, Sequence[DataLoader]]] = None
        self._max_batches: Optional[Union[int, Sequence[int]]] = None

    def reset(self):
        self.iteration_count = 0

        # prepare dataloaders
        self._dataloaders, self._max_batches = self.get_evaluation_dataloaders()
        self._dataloaders = iter(enumerate(self._dataloaders))
        self._current_loader, self._current_loader_idx = None, None

    def done(self):
        try:
            self._current_loader_idx, self._current_loader = next(self._dataloaders)
        except StopIteration:
            return True
        return False

    def advance(self, *args: Any, **kwargs: Any) -> None:
        dataloader = self.trainer.accelerator.process_dataloader(self._current_loader)
        dl_max_batches = self.max_batches[self._current_loader_idx]

        self.evaluation_loop(dataloader, self._current_loader_idx, dl_max_batches)

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        # hook
        self.on_evaluation_start()

    def temp_run(self):
        # check if we want to skip this evaluation
        if self.should_skip_evaluation(self._max_batches):
            return [], []



        # set up the eval loop
        self.setup(self._max_batches, self._dataloaders)

        # hook
        self.on_evaluation_epoch_start()

        # run validation/testing
        for dataloader_idx, dataloader in enumerate(self._dataloaders):
            dataloader = self.trainer.accelerator.process_dataloader(dataloader)
            dl_max_batches = self.max_batches[dataloader_idx]

            self.evaluation_loop(dataloader, dataloader_idx, dl_max_batches)

        outputs = self.outputs

        # reset outputs
        self.outputs = []

        # with a single dataloader don't pass a 2D list
        if len(outputs) > 0 and self.num_dataloaders == 1:
            outputs = outputs[0]

        # lightning module method
        self.evaluation_epoch_end(outputs)

        # hook
        self.on_evaluation_epoch_end()

        # log epoch metrics
        eval_loop_results = self.trainer.logger_connector.get_evaluate_epoch_results()

        # hook
        self.on_evaluation_end()

        return eval_loop_results

    # TODO: Move this to separate loop
    def evaluation_loop(self, dataloader, dataloader_idx, dl_max_batches):
        dl_outputs = []
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            # stop short when running on limited batches
            if batch_idx >= dl_max_batches:
                break

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
            dl_outputs = self.trainer._track_output_for_epoch_end(dl_outputs, output)

        # store batch level output per dataloader
        if self.should_track_batch_outputs_for_epoch_end:
            self.outputs.append(dl_outputs)



# HELPERS

    def on_trainer_init(self) -> None:
        self.trainer.num_sanity_val_batches = []
        self.trainer.num_test_batches = []
        self.trainer.num_val_batches = []
        self.trainer.test_dataloaders = None
        self.trainer.val_dataloaders = None

        # .validate() and .test() set this when they load a checkpoint
        self.trainer.validated_ckpt_path = None
        self.trainer.tested_ckpt_path = None

        # when true, print evaluation results in .validate() and .test()
        self.trainer.verbose_evaluate = True


    def get_evaluation_dataloaders(self) -> Tuple[Optional[List[DataLoader]], List[Union[int, float]]]:
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


    def should_skip_evaluation(self, max_batches: List[Union[int, float]]) -> bool:
        return sum(max_batches) == 0


    def on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        self.should_track_batch_outputs_for_epoch_end: bool = self._should_track_batch_outputs_for_epoch_end()
        if self.trainer.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)


    def on_evaluation_model_eval(self) -> None:
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_eval()
        else:
            model_ref.on_validation_model_eval()


    def on_evaluation_model_train(self) -> None:
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_train()
        else:
            model_ref.on_validation_model_train()


    def on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        if self.trainer.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

        if self.trainer.state.fn != TrainerFn.FITTING:
            # summarize profile results
            self.trainer.profiler.describe()


    def reload_evaluation_dataloaders(self) -> None:
        model = self.trainer.lightning_module
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)
        else:
            self.trainer.reset_val_dataloader(model)


    def setup(self, max_batches: List[Union[int, float]], dataloaders: List[DataLoader]) -> None:
        # bookkeeping
        self.outputs = []
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches
        self.num_dataloaders = self._get_num_dataloaders(dataloaders)


    def on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.call_hook('on_epoch_start', *args, **kwargs)

        if self.trainer.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)


    def _build_kwargs(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Union[Any, int]]:
        # make dataloader_idx arg in validation_step optional
        step_kwargs = OrderedDict([('batch', batch), ('batch_idx', batch_idx)])

        multiple_val_loaders = (
                not self.trainer.testing and self._get_num_dataloaders(self.trainer.val_dataloaders) > 1
        )
        multiple_test_loaders = (self.trainer.testing and self._get_num_dataloaders(self.trainer.test_dataloaders) > 1)

        if multiple_test_loaders or multiple_val_loaders:
            step_kwargs['dataloader_idx'] = dataloader_idx

        return step_kwargs


    def _get_num_dataloaders(self, dataloaders: Optional[List[DataLoader]]) -> int:
        # case where user does:
        # return dl1, dl2
        if dataloaders is not None:
            length = len(dataloaders)
            if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
                length = len(dataloaders[0])
            return length
        else:
            return 0


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


    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        model = self.trainer.lightning_module
        if self.trainer.testing:
            return is_overridden('test_epoch_end', model=model)
        else:
            return is_overridden('validation_epoch_end', model=model)


    def evaluation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # unset dataloder_idx in model
        self.trainer.logger_connector.evaluation_epoch_end()

        # call the model epoch end
        model = self.trainer.lightning_module

        if self.trainer.testing:
            if is_overridden('test_epoch_end', model=model):
                model._current_fx_name = 'test_epoch_end'
                model.test_epoch_end(outputs)

        else:
            if is_overridden('validation_epoch_end', model=model):
                model._current_fx_name = 'validation_epoch_end'
                model.validation_epoch_end(outputs)

        # capture logging
        self.trainer.logger_connector.cache_logged_metrics()


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


    def on_evaluation_epoch_end(self) -> None:
        hook_name = "on_test_epoch_end" if self.trainer.testing else "on_validation_epoch_end"
        self.trainer.call_hook(hook_name)
        self.trainer.call_hook('on_epoch_end')
