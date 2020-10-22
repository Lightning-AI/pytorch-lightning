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
from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.core.step_result import Result, EvalResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.warning_utils import WarningCache


class EvaluationLoop(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.testing = False
        self.outputs = []
        self.step_metrics = []
        self.predictions = None
        self.max_batches = None
        self.warning_cache = WarningCache()

    def on_trainer_init(self):
        self.trainer.num_val_batches = []
        self.trainer.num_sanity_val_batches = []
        self.trainer.num_test_batches = []
        self.trainer.test_dataloaders = None
        self.trainer.val_dataloaders = None
        self.trainer.running_sanity_check = False
        self.trainer.testing = False

        # when .test() is called, it sets this
        self.trainer.tested_ckpt_path = None

        # when true, prints test results
        self.trainer.verbose_test = True

    def get_evaluation_dataloaders(self, max_batches):
        # select dataloaders
        model = self.trainer.get_model()

        # select dataloaders
        if self.testing:
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

    def should_skip_evaluation(self, dataloaders, max_batches):
        # skip when dataloaders aren't defined
        if dataloaders is None:
            return True

        # enable disabling validation step with limit_val_batches = 0
        should_skip = sum(max_batches) == 0
        if should_skip:
            return True

        return False

    def on_evaluation_start(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_start', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_start', *args, **kwargs)

    def on_evaluation_model_eval(self, *args, **kwargs):
        model_ref = self.trainer.get_model()
        if self.testing:
            model_ref.on_test_model_eval()
        else:
            model_ref.on_validation_model_eval()

    def on_evaluation_model_train(self, *args, **kwargs):
        model_ref = self.trainer.get_model()
        if self.testing:
            model_ref.on_test_model_train()
        else:
            model_ref.on_validation_model_train()

    def on_evaluation_end(self, *args, **kwargs):
        if self.testing:
            self.trainer.call_hook('on_test_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_end', *args, **kwargs)

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
        self.trainer.model_connector.copy_trainer_model_properties(model)

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

        multiple_val_loaders = (not test_mode and self._get_num_dataloaders(self.trainer.val_dataloaders) > 1)
        multiple_test_loaders = (test_mode and self._get_num_dataloaders(self.trainer.test_dataloaders) > 1)

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
            output.track_batch_size(batch)

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
        deprecated_results = self.__run_eval_epoch_end(num_dataloaders, using_eval_result)

        # 1.0
        epoch_logs = self.trainer.get_model()._results

        # enable returning anything
        for i, r in enumerate(deprecated_results):
            if not isinstance(r, (dict, Result, torch.Tensor)):
                deprecated_results[i] = []

        return deprecated_results, epoch_logs

    def log_epoch_metrics(self, deprecated_eval_results, epoch_logs, test_mode):
        using_eval_result = self.is_using_eval_results()
        eval_loop_results = self.trainer.logger_connector.on_evaluation_epoch_end(
            deprecated_eval_results,
            epoch_logs,
            using_eval_result,
            test_mode
        )
        return eval_loop_results

    def __run_eval_epoch_end(self, num_dataloaders, using_eval_result):
        model = self.trainer.get_model()

        # reset results
        model._results = Result()

        # with a single dataloader don't pass an array
        outputs = self.outputs
        eval_results = outputs
        if num_dataloaders == 1:
            eval_results = outputs[0]

        user_reduced = False

        if self.testing:
            if is_overridden('test_epoch_end', model=model):
                model._current_fx_name = 'test_epoch_end'
                if using_eval_result:
                    eval_results = self.__gather_epoch_end_eval_results(outputs)

                eval_results = model.test_epoch_end(eval_results)
                user_reduced = True

        else:
            if is_overridden('validation_epoch_end', model=model):
                model._current_fx_name = 'validation_epoch_end'
                if using_eval_result:
                    eval_results = self.__gather_epoch_end_eval_results(outputs)

                eval_results = model.validation_epoch_end(eval_results)
                user_reduced = True

        # depre warning
        if eval_results is not None and user_reduced:
            step = 'testing_epoch_end' if self.testing else 'validation_epoch_end'
            m = f'The {step} should not return anything as of 9.1.' \
                f'to log, use self.log(...) or self.write(...) directly in the LightningModule'
            self.warning_cache.warn(m)

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

    def on_evaluation_batch_start(self, *args, **kwargs):
        # reset the result of the PL module
        model = self.trainer.get_model()
        model._results = Result()
        model._current_fx_name = 'evaluation_step'

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
        # call the callback hook
        if self.testing:
            self.trainer.call_hook('on_test_epoch_end', *args, **kwargs)
        else:
            self.trainer.call_hook('on_validation_epoch_end', *args, **kwargs)

    def log_evaluation_step_metrics(self, batch, batch_idx):
        results = self.trainer.get_model()._results
        if len(results) == 1:
            return None

        results.track_batch_size(batch)
        self.__log_result_step_metrics(results, batch_idx)

        return results

    # TODO: deprecate at 1.0
    def log_evaluation_step_metrics_legacy(self, output, batch_idx):
        if self.trainer.running_sanity_check:
            return

        if isinstance(output, EvalResult):
            self.__log_result_step_metrics(output, batch_idx)

    def __log_result_step_metrics(self, output, batch_idx):
        step_log_metrics = output.get_batch_log_metrics(include_forked_originals=False)
        step_pbar_metrics = output.get_batch_pbar_metrics(include_forked_originals=False)

        if len(step_log_metrics) > 0:
            # make the metrics appear as a different line in the same graph
            metrics_by_epoch = {}
            for k, v in step_log_metrics.items():
                metrics_by_epoch[f'{k}/epoch_{self.trainer.current_epoch}'] = v

            self.trainer.logger_connector.log_metrics(metrics_by_epoch, {}, step=batch_idx)

        if len(step_pbar_metrics) > 0:
            self.trainer.logger_connector.add_progress_bar_metrics(step_pbar_metrics)
