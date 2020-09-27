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

from pytorch_lightning.trainer.supporters import PredictionCollection
from pytorch_lightning.core.step_result import Result, EvalResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_utils import is_overridden


class EvaluationLoop(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.testing = False
        self.outputs = []
        self.predictions = None
        self.max_batches = None

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

    def log_epoch_metrics(self, eval_results, test_mode):
        using_eval_result = self.is_using_eval_results()
        eval_loop_results = self.trainer.logger_connector.on_evaluation_epoch_end(
            eval_results,
            using_eval_result,
            test_mode
        )
        return eval_loop_results

    def __run_eval_epoch_end(self, num_dataloaders, using_eval_result):
        model = self.trainer.get_model()

        eval_results = self.outputs
        if using_eval_result:
            eval_results = self.__gather_epoch_end_eval_results(eval_results)

        # with a single dataloader don't pass an array
        if num_dataloaders == 1:
            eval_results = eval_results[0]

        if self.testing:
            if is_overridden('test_epoch_end', model=model):
                eval_results = model.test_epoch_end(eval_results)

        else:
            if is_overridden('validation_epoch_end', model=model):
                eval_results = model.validation_epoch_end(eval_results)

        if not isinstance(eval_results, list):
            eval_results = [eval_results]

        is_result_obj = len(eval_results) > 0  and isinstance(eval_results[0], Result)
        if is_result_obj:
            eval_results = self.__auto_reduce_result_objs(eval_results)

        return eval_results

    def __gather_epoch_end_eval_results(self, outputs):
        eval_results = []
        for epoch_output in outputs:
            eval_results.append(EvalResult.gather(epoch_output))

        return eval_results

    def __auto_reduce_result_objs(self, outputs):
        # outputs has a list of results per dataloader
        eval_results = []
        for result in outputs:
            eval_results.append(EvalResult.reduce_on_epoch_end(result))

        return eval_results

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

                self.trainer.logger_connector.log_metrics(metrics_by_epoch, {}, step=batch_idx)

            if len(step_pbar_metrics) > 0:
                self.trainer.logger_connector.add_progress_bar_metrics(step_pbar_metrics)
