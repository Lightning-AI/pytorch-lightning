# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import WarningCache


class EvaluationLoop(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.outputs = []
        self.step_metrics = []
        self.predictions = None
        self.max_batches = None
        self.warning_cache = WarningCache()
        self.num_dataloaders = None

    def on_trainer_init(self):
        self.trainer.num_val_batches = []
        self.trainer.num_sanity_val_batches = []
        self.trainer.num_test_batches = []
        self.trainer.test_dataloaders = None
        self.trainer.val_dataloaders = None
        self.trainer.running_sanity_check = False

        # when .test() is called, it sets this
        self.trainer.tested_ckpt_path = None

        # when true, prints test results
        self.trainer.verbose_test = True

    def get_evaluation_dataloaders(self, max_batches):
        # select dataloaders
        model = self.trainer.lightning_module

        # select dataloaders
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)

            dataloaders = self.trainer.test_dataloaders
            new_max_batches = self.trainer.num_test_batches
        else:
            # val
            in_sanity_check = self.trainer.running_sanity_check
            should_reload_every_epoch = self.trainer.reload_dataloaders_every_epoch
            if (self.trainer.val_dataloaders is None or should_reload_every_epoch) and not in_sanity_check:
                self.trainer.reset_val_dataloader(model)

            dataloaders = self.trainer.val_dataloaders
            new_max_batches = self.trainer.num_val_batches

        if max_batches is None:
            max_batches = new_max_batches

        return dataloaders, max_batches

    def should_skip_evaluation(self, max_batches):
        return sum(max_batches) == 0

    def on_evaluation_start(self, *args, **kwargs):
        if self.trainer.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)

    def on_evaluation_model_eval(self, *_, **__):
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_eval()
        else:
            model_ref.on_validation_model_eval()

    def on_evaluation_model_train(self, *_, **__):
        model_ref = self.trainer.lightning_module
        if self.trainer.testing:
            model_ref.on_test_model_train()
        else:
            model_ref.on_validation_model_train()

    def on_evaluation_end(self, *args, **kwargs):
        if self.trainer.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

    def reload_evaluation_dataloaders(self):
        model = self.trainer.lightning_module
        if self.trainer.testing:
            self.trainer.reset_test_dataloader(model)
        else:
            self.trainer.reset_val_dataloader(model)

    def setup(self, model, max_batches, dataloaders):
        # bookkeeping
        self.outputs = []
        self.predictions = PredictionCollection(self.trainer.global_rank, self.trainer.world_size)

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        self.max_batches = max_batches
        self.num_dataloaders = self._get_num_dataloaders(dataloaders)
        self._predictions = [[] for _ in range(self.num_dataloaders)]

    def on_evaluation_epoch_start(self, *args, **kwargs):
        if self.trainer.testing:
            self.trainer.call_hook('on_test_epoch_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_start', *args, **kwargs)

    def _build_args(self, batch, batch_idx, dataloader_idx):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        multiple_val_loaders = (
            not self.trainer.testing and self._get_num_dataloaders(self.trainer.val_dataloaders) > 1
        )
        multiple_test_loaders = (self.trainer.testing and self._get_num_dataloaders(self.trainer.test_dataloaders) > 1)

        if multiple_test_loaders or multiple_val_loaders:
            args.append(dataloader_idx)

        return args

    def _get_num_dataloaders(self, dataloaders):
        # case where user does:
        # return dl1, dl2
        length = len(dataloaders)
        if len(dataloaders) > 0 and isinstance(dataloaders[0], (list, tuple)):
            length = len(dataloaders[0])
        return length

    def evaluation_step(self, batch, batch_idx, dataloader_idx):
        # configure args
        args = self._build_args(batch, batch_idx, dataloader_idx)

        model_ref = self.trainer.lightning_module
        model_ref._results = Result()

        if self.testing:
            model_ref._current_fx_name = "test_step"
            with self.trainer.profiler.profile("test_step"):
                output = self.trainer.accelerator.test_step(args)
        else:
            model_ref._current_fx_name = "validation_step"
            with self.trainer.profiler.profile("validation_step"):
                output = self.trainer.accelerator.validation_step(args)

        # capture any logged information
        self.trainer.logger_connector.cache_logged_metrics()
        # track batch size for weighted average
        is_result_obj = isinstance(output, Result)
        if is_result_obj:
            output.track_batch_size(batch)

        return output

    def evaluation_step_end(self, *args, **kwargs):
        if self.trainer.testing:
            output = self.trainer.call_hook('test_step_end', *args, **kwargs)
        else:
            output = self.trainer.call_hook('validation_step_end', *args, **kwargs)
        return output

    def evaluation_epoch_end(self):
        # unset dataloder_idx in model
        self.trainer.logger_connector.evaluation_epoch_end(self.trainer.testing)

        # call the model epoch end
        deprecated_results = self.__run_eval_epoch_end(self.num_dataloaders)

        # enable returning anything
        for i, r in enumerate(deprecated_results):
            if not isinstance(r, (dict, Result, torch.Tensor)):
                deprecated_results[i] = []

        return deprecated_results

    def log_epoch_metrics_on_evaluation_end(self):
        # get the final loop results
        eval_loop_results = self.trainer.logger_connector.get_evaluate_epoch_results()
        return eval_loop_results

    def __run_eval_epoch_end(self, num_dataloaders):
        model = self.trainer.lightning_module

        # with a single dataloader don't pass an array
        outputs = self.outputs
        eval_results = outputs
        if num_dataloaders == 1:
            eval_results = outputs[0]

        user_reduced = False

        if self.trainer.testing:
            if is_overridden('test_epoch_end', model=model):
                model._current_fx_name = 'test_epoch_end'
                eval_results = model.test_epoch_end(eval_results)
                user_reduced = True

        else:
            if is_overridden('validation_epoch_end', model=model):
                model._current_fx_name = 'validation_epoch_end'
                eval_results = model.validation_epoch_end(eval_results)
                user_reduced = True

        # capture logging
        self.trainer.logger_connector.cache_logged_metrics()
        # depre warning
        if eval_results is not None and user_reduced:
            step = 'testing_epoch_end' if self.trainer.testing else 'validation_epoch_end'
            self.warning_cache.warn(
                f'The {step} should not return anything as of 9.1.'
                ' To log, use self.log(...) or self.write(...) directly in the LightningModule'
            )

        if not isinstance(eval_results, list):
            eval_results = [eval_results]

        # track depreceated metrics
        self.trainer.logger_connector.track_metrics_deprecated(eval_results)

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

    def on_predict_epoch_end(self):
        self.trainer._progress_bar_callback.on_test_end(self.trainer, self.trainer.lightning_module)

        results = self._predictions

        def _convert_to_numpy(v):
            return v.cpu().numpy()

        results = apply_to_collection(results, torch.Tensor, _convert_to_numpy)

        return results, None

    def on_evaluation_batch_start(self, batch, batch_idx, dataloader_idx):
        # set dataloader_idx to model and track batch_size
        self.trainer.logger_connector.on_evaluation_batch_start(
            self.trainer.testing, batch, dataloader_idx, self.num_dataloaders
        )

        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_start', batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_start', batch, batch_idx, dataloader_idx)

    def on_evaluation_batch_end(self, output, batch, batch_idx, dataloader_idx):
        if self.trainer.testing:
            self.trainer.call_hook('on_test_batch_end', output, batch, batch_idx, dataloader_idx)
        else:
            self.trainer.call_hook('on_validation_batch_end', output, batch, batch_idx, dataloader_idx)

        # store predicitons if do_write_predictions and track eval loss history
        self.store_predictions(output, batch_idx, dataloader_idx)

    def store_predictions(self, output, batch_idx, dataloader_idx):
        # Add step predictions to prediction collection to write later
        if output is not None:
            do_write_predictions = isinstance(output, Result) and self.trainer.testing
            if do_write_predictions:
                self.predictions.add(output.pop('predictions', None))

        # track debug metrics
        self.trainer.dev_debugger.track_eval_loss_history(batch_idx, dataloader_idx, output)

    def on_evaluation_epoch_end(self, *args, **kwargs):
        # call the callback hook
        if self.trainer.testing:
            self.trainer.call_hook('on_test_epoch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_end', *args, **kwargs)

        self.trainer.call_hook('on_epoch_end')

    def log_evaluation_step_metrics(self, output, batch_idx):
        if self.trainer.running_sanity_check:
            return

        step_log_metrics = {}
        step_pbar_metrics = {}

        self.__log_result_step_metrics(step_log_metrics, step_pbar_metrics, batch_idx)

    def __log_result_step_metrics(self, step_log_metrics, step_pbar_metrics, batch_idx):
        cached_results = self.trainer.logger_connector.cached_results
        cached_batch_pbar_metrics, cached_batch_log_metrics = cached_results.update_logger_connector()

        step_log_metrics.update(cached_batch_log_metrics)
        step_pbar_metrics.update(cached_batch_pbar_metrics)

        if len(step_log_metrics) > 0:
            # make the metrics appear as a different line in the same graph
            metrics_by_epoch = {}
            for k, v in step_log_metrics.items():
                metrics_by_epoch[f'{k}/epoch_{self.trainer.current_epoch}'] = v

            self.trainer.logger_connector.log_metrics(metrics_by_epoch, {}, step=batch_idx)

        if len(step_pbar_metrics) > 0:
            self.trainer.logger_connector.add_progress_bar_metrics(step_pbar_metrics)
