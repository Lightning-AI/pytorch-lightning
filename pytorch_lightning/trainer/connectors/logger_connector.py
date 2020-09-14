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
from pytorch_lightning.core import memory
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from pytorch_lightning.utilities import flatten_dict
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.core.step_result import EvalResult, Result
from pprint import pprint
from typing import Iterable


class LoggerConnector:

    def __init__(self, trainer):
        self.trainer = trainer
        self.callback_metrics = {}
        self.logged_metrics = {}
        self.progress_bar_metrics = {}

    def on_trainer_init(self, logger, log_save_interval, row_log_interval):
        # logging
        self.configure_logger(logger)
        self.trainer.log_save_interval = log_save_interval
        self.trainer.row_log_interval = row_log_interval

    def configure_logger(self, logger):
        if logger is True:
            # default logger
            self.trainer.logger = TensorBoardLogger(
                save_dir=self.trainer.default_root_dir,
                version=self.trainer.slurm_job_id,
                name='lightning_logs'
            )
        elif logger is False:
            self.trainer.logger = None
        else:
            if isinstance(logger, Iterable):
                self.trainer.logger = LoggerCollection(logger)
            else:
                self.trainer.logger = logger

    def log_metrics(self, metrics, grad_norm_dic, step=None):
        """Logs the metric dict passed in.
        If `step` parameter is None and `step` key is presented is metrics,
        uses metrics["step"] as a step

        Args:
            metrics (dict): Metric values
            grad_norm_dic (dict): Gradient norms
            step (int): Step for which metrics should be logged. Default value corresponds to `self.global_step`
        """
        # add gpu memory
        if self.trainer.on_gpu and self.trainer.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.trainer.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.trainer.metrics_to_scalars(metrics)

        if "step" in scalar_metrics and step is None:
            step = scalar_metrics.pop("step")

        elif step is None:
            # added metrics by Lightning for convenience
            scalar_metrics['epoch'] = self.trainer.current_epoch
            step = step if step is not None else self.trainer.global_step

        # log actual metrics
        if self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.agg_and_log_metrics(scalar_metrics, step=step)
            self.trainer.logger.save()

            # track the logged metrics
            self.logged_metrics = scalar_metrics
            self.trainer.dev_debugger.track_logged_metrics_history(scalar_metrics)

    def add_progress_bar_metrics(self, metrics):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.progress_bar_metrics[k] = v

        self.trainer.dev_debugger.track_pbar_metrics_history(metrics)

    def on_evaluation_epoch_end(self, eval_results, using_eval_result, test_mode):
        # TODO: merge both functions?
        self._log_on_evaluation_epoch_end_metrics(eval_results, using_eval_result)
        return self.__log_evaluation_epoch_metrics_2(eval_results, test_mode)

    def _log_on_evaluation_epoch_end_metrics(self, eval_results, using_eval_result):
        if using_eval_result:
            if isinstance(eval_results, list):
                for eval_result in eval_results:
                    self.trainer.logger_connector.callback_metrics = eval_result.callback_metrics
            else:
                self.trainer.logger_connector.callback_metrics = eval_results.callback_metrics
        else:
            if isinstance(eval_results, list):
                for eval_result in eval_results:
                    # with a scalar return, auto set it to "val_loss" for callbacks
                    if isinstance(eval_result, torch.Tensor):
                        flat = {'val_loss': eval_result}
                    else:
                        flat = flatten_dict(eval_result)
                    self.trainer.logger_connector.callback_metrics.update(flat)
            else:
                # with a scalar return, auto set it to "val_loss" for callbacks
                if isinstance(eval_results, torch.Tensor):
                    flat = {'val_loss': eval_results}
                else:
                    flat = flatten_dict(eval_results)
                self.trainer.logger_connector.callback_metrics.update(flat)

    def __log_evaluation_epoch_metrics_2(self, eval_results, test_mode):
        if self.trainer.running_sanity_check:
            return

        eval_loop_results = []
        if eval_results is not None and len(eval_results) > 0:

            # in eval, the user may return something at every validation step without final reduction
            if not isinstance(eval_results, list):
                eval_results = [eval_results]

            for result_idx, result in enumerate(eval_results):
                if isinstance(result, EvalResult):
                    prog_bar_metrics = result.epoch_pbar_metrics
                    log_metrics = result.epoch_log_metrics
                    callback_metrics = result.callback_metrics

                    # in testing we don't need the callback metrics
                    if test_mode:
                        callback_metrics = {}
                else:
                    _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.trainer.process_output(result)

                # eval loop returns all metrics
                dataloader_result_metrics = {**prog_bar_metrics, **log_metrics, **callback_metrics}

                # add metrics to prog bar
                self.trainer.logger_connector.add_progress_bar_metrics(prog_bar_metrics)

                # log metrics
                self.trainer.logger_connector.log_metrics(log_metrics, {})

                # track metrics for callbacks
                self.trainer.logger_connector.callback_metrics.update(callback_metrics)

                if len(dataloader_result_metrics) > 0:
                    eval_loop_results.append(dataloader_result_metrics)

        # log results of test
        if test_mode and self.trainer.is_global_zero and self.trainer.verbose_test:
            print('-' * 80)
            for result_idx, results in enumerate(eval_loop_results):
                print(f'DATALOADER:{result_idx} TEST RESULTS')
                pprint(results)
                print('-' * 80)

        return eval_loop_results

    def on_train_epoch_end(self, epoch_output, checkpoint_accumulator, early_stopping_accumulator, num_optimizers):
        self.log_train_epoch_end_metrics(epoch_output, checkpoint_accumulator,
                                         early_stopping_accumulator, num_optimizers)

    def log_train_epoch_end_metrics(self,
                                    epoch_output,
                                    checkpoint_accumulator,
                                    early_stopping_accumulator,
                                    num_optimizers):
        # epoch output is a list. Each item in that list has all the outputs per optimizer
        # epoch_output[optimizer_idx][training_step_idx][tbptt_index]
        # remember that not using truncated backprop is equivalent with truncated back prop of len(1)

        model = self.trainer.get_model()

        epoch_log_metrics = {}
        epoch_callback_metrics = {}
        epoch_progress_bar_metrics = {}

        # -----------------------
        # Calculate epoch callback values if given
        # -----------------------
        if checkpoint_accumulator.num_values > 0:
            epoch_callback_metrics['checkpoint_on'] = checkpoint_accumulator.mean()

        if early_stopping_accumulator.num_values > 0:
            epoch_callback_metrics['early_stop_on'] = early_stopping_accumulator.mean()

        # ------------------------
        # determine if using a result obj
        # ------------------------
        # [optimizer_idx][training_step_idx][tbptt_index]
        opt_idx_outputs = epoch_output[0]

        try:
            sample_obj = opt_idx_outputs[0][0] if isinstance(opt_idx_outputs[0], list) else opt_idx_outputs[0]
            is_result_obj = len(epoch_output) > 0 and isinstance(sample_obj, Result)
        except IndexError as e:
            is_result_obj = False

        # --------------------------
        # EPOCH END STEP IF DEFINED
        # --------------------------
        if is_overridden('training_epoch_end', model=model):
            self.trainer.global_step += 1

            if is_result_obj:
                # with result object gather across time and training steps so each opt idx has a single result obj
                epoch_output = self.__gather_result_across_time_and_optimizers(epoch_output)

            if num_optimizers == 1:
                epoch_output = epoch_output[0]

            # run training_epoch_end
            # a list with a result per optimizer index
            epoch_output = model.training_epoch_end(epoch_output)

            if isinstance(epoch_output, Result):
                epoch_log_metrics = epoch_output.epoch_log_metrics
                epoch_progress_bar_metrics = epoch_output.epoch_pbar_metrics
            else:
                _processed_outputs = self.trainer.process_output(epoch_output)
                epoch_progress_bar_metrics = _processed_outputs[1]
                epoch_log_metrics = _processed_outputs[2]
                epoch_callback_metrics = _processed_outputs[3]

        # --------------------------
        # Structured Result (auto epoch end)
        # --------------------------
        elif is_result_obj:
            epoch_log_metrics, epoch_progress_bar_metrics = self.__auto_reduce_results_on_epoch_end(epoch_output)

        # --------------------------
        # track results
        # --------------------------
        # add the metrics to the loggers
        if epoch_log_metrics and len(epoch_log_metrics) > 0:
            self.log_metrics(epoch_log_metrics, {})

        # add metrics to callbacks
        self.callback_metrics.update(epoch_callback_metrics)

        # add metrics to progress_bar
        if len(epoch_progress_bar_metrics) > 0:
            self.add_progress_bar_metrics(epoch_progress_bar_metrics)

    def __auto_reduce_results_on_epoch_end(self, epoch_output):
        epoch_log_metrics = {}
        epoch_progress_bar_metrics = {}
        for opt_outputs in epoch_output:
            # reduce across time first
            time_reduced_outputs = []
            for train_step_idx in range(len(opt_outputs)):
                tbptt_outs = opt_outputs[train_step_idx]
                tbptt_outs = tbptt_outs[0].__class__.reduce_across_time(tbptt_outs)
                time_reduced_outputs.append(tbptt_outs)

            # reduce across training steps
            opt_outputs = time_reduced_outputs[0].__class__.reduce_on_epoch_end(time_reduced_outputs)
            opt_outputs.minimize = opt_outputs.minimize.mean()
            epoch_log_metrics.update(opt_outputs.epoch_log_metrics)
            epoch_progress_bar_metrics.update(opt_outputs.epoch_pbar_metrics)

        return epoch_log_metrics, epoch_progress_bar_metrics

    def __gather_result_across_time_and_optimizers(self, epoch_output):
        """
        Gather results into a single padded tensor per metric where each tensor is gathered across
        time and across time steps.

        Returns:
            a list where each element is a Result with the tensors gathered
        """
        gathered_epoch_outputs = []
        for opt_outputs in epoch_output:
            # gather across time first
            time_gathered_outputs = []
            for train_step_idx in range(len(opt_outputs)):
                tbptt_outs = opt_outputs[train_step_idx]
                tbptt_outs = tbptt_outs[0].__class__.gather(tbptt_outs)
                time_gathered_outputs.append(tbptt_outs)

            # gather across training steps
            # each metric has dimensions (training_steps, seq_len) (seq_len=1 when no tbptt is used)
            gathered_opt_output = time_gathered_outputs[0].__class__.padded_gather(time_gathered_outputs)
            gathered_epoch_outputs.append(gathered_opt_output)

        return gathered_epoch_outputs

    def save_train_loop_metrics_to_loggers(self, batch_idx, batch_output):
        # when metrics should be logged
        should_log_metrics = (batch_idx + 1) % self.trainer.row_log_interval == 0 or self.trainer.should_stop
        if should_log_metrics or self.trainer.fast_dev_run:
            # logs user requested information to logger
            metrics = batch_output.batch_log_metrics
            grad_norm_dic = batch_output.grad_norm_dic
            if len(metrics) > 0 or len(grad_norm_dic) > 0:
                self.log_metrics(metrics, grad_norm_dic)
