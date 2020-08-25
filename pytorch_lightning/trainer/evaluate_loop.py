import torch
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.core.step_result import Result, EvalResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import flatten_dict


class EvaluationLoop(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.testing = False
        self.outputs = []
        self.predictions = None
        self.max_batches = None

    def get_evaluation_dataloaders(self):
        # select dataloaders
        model = self.trainer.get_model()

        if self.testing:
            self.trainer.reset_test_dataloader(model)
            dataloaders = self.trainer.test_dataloaders
            max_batches = self.trainer.num_test_batches
        else:
            if self.trainer.val_dataloaders is None:
                self.trainer.reset_val_dataloader(model)

            dataloaders = self.trainer.val_dataloaders
            max_batches = self.trainer.num_val_batches

        return dataloaders, max_batches

    def reload_evaluation_dataloaders(self):
        model = self.trainer.get_model()
        if self.testing:
            self.trainer.reset_test_dataloader(model)
        else:
            self.trainer.reset_val_dataloader(model)

    def is_using_eval_results(self):
        outputs = self.outputs
        using_eval_result = len(outputs) > 0 and len(outputs[0]) > 0 and isinstance(outputs[0][0], EvalResult)
        return using_eval_result

    def setup(self, model, max_batches, dataloaders):
        # copy properties for forward overrides
        self.trainer.copy_trainer_model_properties(model)

        # bookkeeping
        self.outputs = []
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches

    def on_evaluation_epoch_start(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)

    def build_args(self, test_mode, batch, batch_idx, dataloader_idx):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        multiple_val_loaders = (not test_mode and len(self.trainer.val_dataloaders) > 1)
        multiple_test_loaders = (test_mode and len(self.trainer.test_dataloaders) > 1)

        if multiple_test_loaders or multiple_val_loaders:
            args.append(dataloader_idx)

        return args

    def evaluation_step(self, test_mode, batch, batch_idx, dataloader_idx):
        # configure args
        args = self.build_args(test_mode, batch, batch_idx, dataloader_idx)

        # run actual test step
        if self.testing:
            output = self.trainer.accelerator_backend.test_step(args)
        else:
            output = self.trainer.accelerator_backend.validation_step(args)

        # track batch size for weighted average
        is_result_obj = isinstance(output, Result)
        if is_result_obj:
            output.track_batch_size(len(batch))

        # allow only EvalResult when using structured results (from val_step)
        if is_result_obj and not isinstance(output, EvalResult):
            m = 'only EvalResults or dicts are allowed from validation_step'
            raise MisconfigurationException(m)

        return output

    def evaluation_step_end(self, *args, **kwargs):
        if self.testing:
            output = self.trainer.call_hook('test_step_end', *args, **kwargs)
        else:
            output = self.trainer.call_hook('validation_step_end', *args, **kwargs)
        return output

    def evaluation_epoch_end(self, num_dataloaders):
        using_eval_result = self.is_using_eval_results()

        # call the model epoch end
        eval_results = self.__run_eval_epoch_end(num_dataloaders, using_eval_result)
        return eval_results

    def log_epoch_metrics(self, eval_results):
        using_eval_result = self.is_using_eval_results()
        if using_eval_result:
            if isinstance(eval_results, list):
                for eval_result in eval_results:
                    self.trainer.callback_metrics = eval_result.callback_metrics
            else:
                self.trainer.callback_metrics = eval_results.callback_metrics
        else:
            if isinstance(eval_results, list):
                for eval_result in eval_results:
                    # with a scalar return, auto set it to "val_loss" for callbacks
                    if isinstance(eval_result, torch.Tensor):
                        flat = {'val_loss': eval_result}
                    else:
                        flat = flatten_dict(eval_result)
                    self.trainer.callback_metrics.update(flat)
            else:
                # with a scalar return, auto set it to "val_loss" for callbacks
                if isinstance(eval_results, torch.Tensor):
                    flat = {'val_loss': eval_results}
                else:
                    flat = flatten_dict(eval_results)
                self.trainer.callback_metrics.update(flat)

    def __run_eval_epoch_end(self, num_dataloaders, using_eval_result):
        model = self.trainer.get_model()

        # with a single dataloader don't pass an array
        outputs = self.outputs
        eval_results = outputs
        if num_dataloaders == 1:
            eval_results = outputs[0]

        user_reduced = False

        if self.testing:
            if self.trainer.is_overridden('test_epoch_end', model=model):
                if using_eval_result:
                    eval_results = self.__gather_epoch_end_eval_results(outputs)

                eval_results = model.test_epoch_end(eval_results)
                user_reduced = True

        else:
            if self.trainer.is_overridden('validation_epoch_end', model=model):
                if using_eval_result:
                    eval_results = self.__gather_epoch_end_eval_results(outputs)

                eval_results = model.validation_epoch_end(eval_results)
                user_reduced = True

        if using_eval_result and not user_reduced:
            eval_results = self.__auto_reduce_result_objs(outputs)

        if not isinstance(eval_results, list):
            eval_results = [eval_results]

        return eval_results

    def __gather_epoch_end_eval_results(self, outputs):
        eval_results = []
        for epoch_output in outputs:
            result = epoch_output[0].__class__.gather(epoch_output)
            if 'checkpoint_on' in result:
                result.checkpoint_on = result.checkpoint_on.mean()
            if 'early_stop_on' in result:
                result.early_stop_on = result.early_stop_on.mean()

            eval_results.append(result)

        # with 1 dataloader don't pass in a list
        if len(eval_results) == 1:
            eval_results = eval_results[0]
        return eval_results

    def __auto_reduce_result_objs(self, outputs):
        # outputs has a list of results per dataloader
        eval_results = []
        for dl_output in outputs:
            result = dl_output[0]
            result = result.__class__.reduce_on_epoch_end(dl_output)
            if 'checkpoint_on' in result:
                result.checkpoint_on = result.checkpoint_on.mean()
            if 'early_stop_on' in result:
                result.early_stop_on = result.early_stop_on.mean()
            eval_results.append(result)

        return eval_results

    def on_evaluation_start(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)

    def on_evaluation_end(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

    def on_evaluation_batch_start(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_batch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_batch_start', *args, **kwargs)

    def on_evaluation_batch_end(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_batch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_batch_end', *args, **kwargs)

    def evaluation_batch_end_cleanup(self, output, batch_idx, dataloader_idx):
        # Add step predictions to prediction collection to write later
        if output is not None:
            do_write_predictions = isinstance(output, Result) and self.testing
            if do_write_predictions:
                self.predictions.add(output.pop('predictions', None))

        # track debug metrics
        self.trainer.dev_debugger.track_eval_loss_history(self.testing, batch_idx, dataloader_idx, output)

    def on_evaluation_epoch_end(self, eval_results, *args, **kwargs):
        # log epoch level metrics
        self.log_epoch_metrics(eval_results)

        # Write predictions to disk if they're available
        self.predictions.to_disk()

        # call the callback hook
        if self.testing:
            self.trainer.call_hook('on_test_epoch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_end', *args, **kwargs)

    def log_step_metrics(self, output, batch_idx):
        if self.trainer.running_sanity_check:
            return

        if isinstance(output, EvalResult):
            step_log_metrics = output.batch_log_metrics
            step_pbar_metrics = output.batch_pbar_metrics

            if len(step_log_metrics) > 0:
                # make the metrics appear as a different line in the same graph
                metrics_by_epoch = {}
                for k, v in step_log_metrics.items():
                    metrics_by_epoch[f'{k}/epoch_{self.trainer.current_epoch}'] = v

                self.trainer.log_metrics(metrics_by_epoch, {}, step=batch_idx)

            if len(step_pbar_metrics) > 0:
                self.trainer.add_progress_bar_metrics(step_pbar_metrics)

    def should_skip_evaluation(self, dataloaders, max_batches):
        # skip when dataloaders aren't defined
        if dataloaders is None:
            return True

        # enable disabling validation step with limit_val_batches = 0
        should_skip = sum(max_batches) == 0
        if should_skip:
            return True

        return False