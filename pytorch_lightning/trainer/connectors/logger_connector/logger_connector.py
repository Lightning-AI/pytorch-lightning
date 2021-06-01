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
import os
from copy import deepcopy
from pprint import pprint
from typing import Dict, Iterable, List, Optional, Union

import torch

from pytorch_lightning.core import memory
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector.epoch_result_store import EpochResultStore
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import FxValidator
from pytorch_lightning.trainer.connectors.logger_connector.metrics_holder import MetricsHolder
from pytorch_lightning.trainer.connectors.logger_connector.result import Result
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities import DeviceType
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT


class LoggerConnector:

    def __init__(self, trainer, log_gpu_memory: Optional[str] = None):
        self.trainer = trainer
        self.log_gpu_memory = log_gpu_memory
        self._callback_metrics = MetricsHolder()
        self._evaluation_callback_metrics = MetricsHolder(to_float=True)
        self._logged_metrics = MetricsHolder()
        self._progress_bar_metrics = MetricsHolder(to_float=True)
        self.eval_loop_results = []
        self._cached_results = {stage: EpochResultStore(trainer) for stage in RunningStage}
        self._cached_results[None] = EpochResultStore(trainer)
        self._fx_validator = FxValidator()
        self._val_log_step: int = 0
        self._test_log_step: int = 0

    @property
    def callback_metrics(self) -> Dict:
        return self.get_metrics("callback_metrics")

    @callback_metrics.setter
    def callback_metrics(self, callback_metrics: Dict) -> None:
        self.set_metrics("callback_metrics", callback_metrics)

    @property
    def evaluation_callback_metrics(self) -> Dict:
        return self.get_metrics("evaluation_callback_metrics")

    @evaluation_callback_metrics.setter
    def evaluation_callback_metrics(self, evaluation_callback_metrics: Dict) -> None:
        self.set_metrics("evaluation_callback_metrics", evaluation_callback_metrics)

    @property
    def logged_metrics(self) -> Dict:
        return self.get_metrics("logged_metrics")

    @logged_metrics.setter
    def logged_metrics(self, logged_metrics: Dict) -> None:
        self.set_metrics("logged_metrics", logged_metrics)

    @property
    def progress_bar_metrics(self) -> Dict:
        return self.get_metrics("progress_bar_metrics")

    @progress_bar_metrics.setter
    def progress_bar_metrics(self, progress_bar_metrics: Dict) -> None:
        self.set_metrics("progress_bar_metrics", progress_bar_metrics)

    @property
    def cached_results(self) -> Union[EpochResultStore, None]:
        return self._cached_results.get(self.trainer.state.stage)

    def get_metrics(self, key: str) -> Dict:
        metrics_holder: MetricsHolder = getattr(self, f"_{key}")
        model = self.trainer.lightning_module
        metrics_holder.convert(model.device if model is not None else None)
        return metrics_holder.metrics

    def set_metrics(self, key: str, val: Dict) -> None:
        metrics_holder: MetricsHolder = getattr(self, f"_{key}")
        metrics_holder.reset(val)

    def reset(self) -> None:
        self.cached_results.reset()

    def check_logging(self, fx_name: str, on_step: bool, on_epoch: bool) -> None:
        self._fx_validator.check_logging(fx_name=fx_name, on_step=on_step, on_epoch=on_epoch)

    def on_evaluation_batch_start(self, batch, dataloader_idx, num_dataloaders):
        model = self.trainer.lightning_module
        # set dataloader_idx only if multiple ones
        model._current_dataloader_idx = dataloader_idx if num_dataloaders > 1 else None
        # track batch_size
        self.cached_results._batch_size = Result.extract_batch_size(batch)

    def on_train_split_start(self, split_idx: int, opt_idx: int, split_batch) -> None:
        self.cached_results._split_idx = split_idx
        self.cached_results._opt_idx = opt_idx
        self.cached_results._batch_size = Result.extract_batch_size(split_batch)

    def on_train_batch_end(self) -> None:
        self.cached_results._split_idx = None
        self.cached_results._opt_idx = None
        self.cached_results._batch_size = None

    def cache_logged_metrics(self):
        self._cached_results[self.trainer.state.stage].cache_result()

    def on_trainer_init(self, logger, flush_logs_every_n_steps: int, log_every_n_steps: int, move_metrics_to_cpu: bool):
        # logging
        self.configure_logger(logger)
        self.trainer.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.trainer.log_every_n_steps = log_every_n_steps
        self.trainer.move_metrics_to_cpu = move_metrics_to_cpu

    @property
    def should_flush_logs(self):
        should_flush = (self.trainer.global_step + 1) % self.trainer.flush_logs_every_n_steps == 0
        return should_flush or self.trainer.should_stop

    @property
    def should_update_logs(self):
        should_log_every_n_steps = (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
        return should_log_every_n_steps or self.trainer.should_stop

    def configure_logger(self, logger):
        if logger is True:
            version = os.environ.get('PL_EXP_VERSION', self.trainer.slurm_job_id)

            # default logger
            self.trainer.logger = TensorBoardLogger(
                save_dir=self.trainer.default_root_dir, version=version, name='lightning_logs'
            )
        elif logger is False:
            self.trainer.logger = None
        else:
            if isinstance(logger, Iterable):
                self.trainer.logger = LoggerCollection(logger)
            else:
                self.trainer.logger = logger

    def cache_training_step_metrics(self, opt_closure_result):
        """
        This function is responsible to update
        logger_connector internals metrics holder based for depreceated logging
        """
        using_results_obj = isinstance(opt_closure_result.training_step_output, Result)

        # temporary dict to collect metrics
        logged_metrics_tmp = {}
        pbar_metrics_tmp = {}
        callback_metrics_tmp = {}

        if using_results_obj:
            batch_log_metrics = opt_closure_result.training_step_output.get_batch_log_metrics(
                include_forked_originals=False
            )
            logged_metrics_tmp.update(batch_log_metrics)

            batch_pbar_metrics = opt_closure_result.training_step_output.get_batch_pbar_metrics(
                include_forked_originals=False
            )
            pbar_metrics_tmp.update(batch_pbar_metrics)

            forked_metrics = opt_closure_result.training_step_output.get_forked_metrics()
            callback_metrics_tmp.update(forked_metrics)
            callback_metrics_tmp.update(logged_metrics_tmp)

        else:
            batch_log_metrics = opt_closure_result.training_step_output.log_metrics
            logged_metrics_tmp.update(batch_log_metrics)

            batch_pbar_metrics = opt_closure_result.training_step_output.pbar_on_batch_end
            pbar_metrics_tmp.update(batch_pbar_metrics)

        # track progress bar metrics
        if len(pbar_metrics_tmp) > 0:
            self.add_progress_bar_metrics(pbar_metrics_tmp)

        self._callback_metrics.update(callback_metrics_tmp)
        self._logged_metrics.update(logged_metrics_tmp)

    def log_metrics(self, metrics, grad_norm_dict, step=None):
        """Logs the metric dict passed in.
        If `step` parameter is None and `step` key is presented is metrics,
        uses metrics["step"] as a step

        Args:
            metrics (dict): Metric values
            grad_norm_dict (dict): Gradient norms
            step (int): Step for which metrics should be logged. Default value is `self.global_step` during training or
                the total validation / test log step count during validation and testing.
        """
        # add gpu memory
        if self.trainer._device_type == DeviceType.GPU and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dict)

        # turn all tensors to scalars
        scalar_metrics = metrics_to_scalars(metrics)

        if "step" in scalar_metrics and step is None:
            step = scalar_metrics.pop("step")

        elif step is None:
            # added metrics by Lightning for convenience
            scalar_metrics['epoch'] = self.trainer.current_epoch
            step = self.trainer.global_step

        # log actual metrics
        if self.trainer.logger is not None:
            if self.trainer.is_global_zero:
                self.trainer.logger.agg_and_log_metrics(scalar_metrics, step=step)
                self.trainer.logger.save()

            # track the logged metrics
            self.logged_metrics.update(scalar_metrics)

    def add_progress_bar_metrics(self, metrics):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            self._progress_bar_metrics.metrics[k] = v

    def evaluation_epoch_end(self):
        # reset dataloader idx
        model_ref = self.trainer.lightning_module
        model_ref._current_dataloader_idx = None

        # setting `has_batch_loop_finished` to True
        # will perform Results reduction accross entire epoch.
        self.cached_results.has_batch_loop_finished = True

    def add_to_eval_loop_results(self, dl_idx, has_been_initialized):
        callback_metrics = deepcopy(self.evaluation_callback_metrics)
        for key in list(callback_metrics.keys()):
            if "dataloader_idx" in key:
                if f"dataloader_idx_{dl_idx}" not in key:
                    # remove dl_idx from self.callback_metrics not belonging to this dataset.
                    del callback_metrics[key]
        if has_been_initialized:
            self.eval_loop_results[dl_idx].update(callback_metrics)
        else:
            self.eval_loop_results.append(callback_metrics)

    def prepare_eval_loop_results(self):
        num_dataloaders = self.trainer.evaluation_loop.num_dataloaders
        has_been_initialized = len(self.eval_loop_results) == num_dataloaders
        for dl_idx in range(self.trainer.evaluation_loop.num_dataloaders):
            self.add_to_eval_loop_results(dl_idx, has_been_initialized)

    def get_evaluate_epoch_results(self) -> _EVALUATE_OUTPUT:
        if not self.trainer.sanity_checking:
            # log all the metrics as a single dict
            metrics_to_log = self.cached_results.get_epoch_log_metrics()
            if len(metrics_to_log) > 0:
                self.log_metrics(metrics_to_log, {})

        self.prepare_eval_loop_results()

        # log results of evaluation
        if (
            self.trainer.state.fn != TrainerFn.FITTING and self.trainer.evaluating and self.trainer.is_global_zero
            and self.trainer.verbose_evaluate
        ):
            print('-' * 80)
            for result_idx, results in enumerate(self.eval_loop_results):
                print(f'DATALOADER:{result_idx} {self.trainer.state.stage.upper()} RESULTS')
                pprint({
                    k: (v.item() if v.numel() == 1 else v.tolist()) if isinstance(v, torch.Tensor) else v
                    for k, v in results.items()
                })
                print('-' * 80)

        results = self.eval_loop_results

        # clear mem
        self.eval_loop_results = []
        return results

    def on_train_epoch_end(self):
        # inform cached logger connector epoch finished
        self.cached_results.has_batch_loop_finished = True

    def log_train_epoch_end_metrics(self, epoch_output: List[List[List[Result]]]) -> None:
        # epoch output is a list. Each item in that list has all the outputs per optimizer
        # epoch_output[optimizer_idx][training_step_idx][tbptt_index]
        # remember that not using truncated backprop is equivalent with truncated back prop of len(1)

        # log/aggregate metrics automatically
        epoch_log_metrics, epoch_progress_bar_metrics = self.__auto_reduce_results_on_epoch_end(epoch_output)

        # it will perform reduction over epoch and return log metrics
        cached_epoch_log_metrics = self.cached_results.get_epoch_log_metrics()
        cached_epoch_pbar_metrics = self.cached_results.get_epoch_pbar_metrics()

        # update
        epoch_log_metrics.update(cached_epoch_log_metrics)
        epoch_progress_bar_metrics.update(cached_epoch_pbar_metrics)

        # --------------------------
        # track results
        # --------------------------
        # add the metrics to the loggers and callbacks
        if epoch_log_metrics and len(epoch_log_metrics) > 0:
            self.log_metrics(epoch_log_metrics, {})
            self._callback_metrics.update(epoch_log_metrics)

        # add metrics to progress_bar and callbacks
        if len(epoch_progress_bar_metrics) > 0:
            self.add_progress_bar_metrics(epoch_progress_bar_metrics)
            self._callback_metrics.update(epoch_progress_bar_metrics)

        # reset epoch loop result for next epoch
        self.cached_results.reset()

    def __auto_reduce_results_on_epoch_end(self, epoch_output):
        epoch_log_metrics = {}
        epoch_progress_bar_metrics = {}
        for opt_outputs in epoch_output:
            # reduce across time first
            time_reduced_outputs = []
            for tbptt_outs in opt_outputs:
                tbptt_outs = tbptt_outs[0].__class__.reduce_across_time(tbptt_outs)
                if len(tbptt_outs) > 1:
                    time_reduced_outputs.append(tbptt_outs)

            if len(time_reduced_outputs) == 0:
                continue

            # reduce across training steps
            opt_outputs = time_reduced_outputs[0].__class__.reduce_on_epoch_end(time_reduced_outputs)

            # with manual opt need 1 + metrics because meta is always there
            if opt_outputs.minimize is not None:
                opt_outputs.minimize = opt_outputs.minimize.mean()
            epoch_log_metrics.update(opt_outputs.epoch_log_metrics)
            epoch_progress_bar_metrics.update(opt_outputs.epoch_pbar_metrics)

        return epoch_log_metrics, epoch_progress_bar_metrics

    def log_train_step_metrics(self, batch_output):
        if self.trainer.train_loop.should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return
        _, batch_log_metrics = self.cached_results.update_logger_connector()
        # when metrics should be logged
        if self.should_update_logs or self.trainer.fast_dev_run is True:
            # logs user requested information to logger
            grad_norm_dict = batch_output.grad_norm_dict
            if grad_norm_dict is None:
                grad_norm_dict = {}
            if len(batch_log_metrics) > 0 or len(grad_norm_dict) > 0:
                self.log_metrics(batch_log_metrics, grad_norm_dict)
                self._callback_metrics.update(batch_log_metrics)

    @property
    def evaluation_log_step(self) -> Optional[int]:
        if self.trainer.state.stage is RunningStage.VALIDATING:
            return self._val_log_step
        elif self.trainer.state.stage is RunningStage.TESTING:
            return self._test_log_step
        else:
            return None

    def increment_evaluation_log_step(self) -> None:
        if self.trainer.state.stage is RunningStage.VALIDATING:
            self._val_log_step += 1
        elif self.trainer.state.stage is RunningStage.TESTING:
            self._test_log_step += 1

    def log_evaluation_step_metrics(self) -> None:
        if self.trainer.sanity_checking:
            return
        _, batch_log_metrics = self.cached_results.update_logger_connector()

        # logs user requested information to logger
        if len(batch_log_metrics) > 0:
            kwargs = dict() if "step" in batch_log_metrics else dict(step=self.evaluation_log_step)
            self.log_metrics(batch_log_metrics, {}, **kwargs)

        # increment the step even if nothing was logged
        self.increment_evaluation_log_step()
