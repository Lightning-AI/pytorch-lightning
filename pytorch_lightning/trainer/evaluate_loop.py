import torch
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.core.step_result import Result, EvalResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EvaluationLoop(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.testing = False
        self.outputs = []
        self.predictions = None
        self.max_batches = None

    def is_using_eval_results(self):
        outputs = self.outputs
        using_eval_result = len(outputs) > 0 and len(outputs[0]) > 0 and isinstance(outputs[0][0], EvalResult)
        return using_eval_result

    def setup(self, model, max_batches, dataloaders):
        # enable eval mode
        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.trainer.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

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

        if (test_mode and len(self.trainer.test_dataloaders) > 1) or (not test_mode and len(self.val_dataloaders) > 1):
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

    def on_evaluation_epoch_end(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_epoch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_end', *args, **kwargs)

    def log_metrics(self, output, batch_idx):
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
