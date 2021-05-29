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
from typing import Any, Dict, Iterable, Optional

import torch

from pytorch_lightning.core import memory
from pytorch_lightning.core.result import DefaultMetricsKeys
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import FxValidator
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities import DeviceType
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT


class LoggerConnector:

    def __init__(self, trainer, log_gpu_memory: Optional[str] = None):
        self.trainer = trainer
        self.log_gpu_memory = log_gpu_memory
        self.eval_loop_results = []
        self._fx_validator = FxValidator()
        self._val_log_step: int = 0
        self._test_log_step: int = 0
        self._progress_bar_metrics: Dict[str, float] = {}
        self._logged_metrics: Dict[str, float] = {}
        self._callback_metrics: Dict[str, float] = {}

    def on_trainer_init(self, logger, flush_logs_every_n_steps: int, log_every_n_steps: int, move_metrics_to_cpu: bool):
        # logging
        self.configure_logger(logger)
        self.trainer.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.trainer.log_every_n_steps = log_every_n_steps
        self.trainer.move_metrics_to_cpu = move_metrics_to_cpu

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

    @property
    def should_flush_logs(self):
        should_flush = (self.trainer.global_step + 1) % self.trainer.flush_logs_every_n_steps == 0
        return should_flush or self.trainer.should_stop

    @property
    def should_update_logs(self):
        should_log_every_n_steps = (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
        return should_log_every_n_steps or self.trainer.should_stop

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

            self.add_logged_metrics(scalar_metrics)

    """
    Evaluation metric updates
    """

    def evaluation_epoch_end(self):
        # reset dataloader idx
        model_ref = self.trainer.lightning_module
        model_ref._current_dataloader_idx = None
        self.trainer.result_collections.on_epoch_end_reached = True

    def add_to_eval_loop_results(self, dl_idx, has_been_initialized):
        if self.trainer.sanity_checking:
            return

        callback_metrics = self.trainer.result_collections.metrics[DefaultMetricsKeys.CALLBACK]
        if os.getenv("PL_DEV_DEBUG", '0') == '1':
            callback_metrics["debug_epoch"] = self.trainer.current_epoch
        callback_metrics = deepcopy(callback_metrics)
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
        metrics = self.trainer.result_collections.metrics

        # update metrics
        self.add_progress_bar_metrics(metrics[DefaultMetricsKeys.PBAR])
        self.add_callback_metrics(metrics[DefaultMetricsKeys.CALLBACK])

        if not self.trainer.sanity_checking:

            # log all the metrics as a single dict
            metrics_to_log = metrics[DefaultMetricsKeys.LOG]
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

    def on_evaluation_start(self):
        root_device = self.trainer.lightning_module.device
        self.trainer.result_collections.root_device = root_device

    def on_evaluation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int, num_dataloaders: int) -> None:
        model = self.trainer.lightning_module
        # set dataloader_idx only if multiple ones
        model._current_dataloader_idx = dataloader_idx if num_dataloaders > 1 else None

        # track batch_size
        self.trainer.result_collections.extract_batch_size(batch)
        self.trainer.result_collections.batch_idx = batch_idx

    def update_evaluation_step_metrics(self) -> None:
        metrics = self.trainer.result_collections.metrics

        # update metrics
        self.add_progress_bar_metrics(metrics[DefaultMetricsKeys.PBAR])
        self.add_callback_metrics(metrics[DefaultMetricsKeys.CALLBACK])

        if self.trainer.sanity_checking:
            return

        batch_log_metrics = metrics[DefaultMetricsKeys.LOG]

        # logs user requested information to logger
        if len(batch_log_metrics) > 0:
            kwargs = dict() if "step" in batch_log_metrics else dict(step=self.evaluation_log_step)
            self.log_metrics(batch_log_metrics, {}, **kwargs)

        # increment the step even if nothing was logged
        self.increment_evaluation_log_step()

    """
    Train metric updates
    """

    def on_train_start(self):
        root_device = self.trainer.lightning_module.device
        self.trainer.result_collections.root_device = root_device

    def on_train_split_start(self, batch_idx: int, split_batch: Any) -> None:
        self.trainer.result_collections.extract_batch_size(split_batch)
        self.trainer.result_collections.batch_idx = batch_idx

    def on_train_batch_end(self) -> None:
        self.trainer.result_collections.batch_size = 1

    def update_train_step_metrics(self, batch_output):
        metrics = self.trainer.result_collections.metrics

        # update metrics
        self.add_progress_bar_metrics(metrics[DefaultMetricsKeys.PBAR])
        self.add_callback_metrics(metrics[DefaultMetricsKeys.CALLBACK])

        if self.trainer.train_loop.should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return

        batch_log_metrics = metrics[DefaultMetricsKeys.LOG]

        # when metrics should be logged
        if self.should_update_logs or self.trainer.fast_dev_run is True:
            # logs user requested information to logger
            grad_norm_dict = batch_output.grad_norm_dict
            if grad_norm_dict is None:
                grad_norm_dict = {}
            if len(batch_log_metrics) > 0 or len(grad_norm_dict) > 0:
                self.log_metrics(batch_log_metrics, grad_norm_dict)

    def on_train_epoch_end(self):
        # inform cached logger connector epoch finished
        self.trainer.result_collections.on_epoch_end_reached = True

    def update_train_epoch_metrics(self) -> None:

        metrics = self.trainer.result_collections.metrics

        # update metrics
        self.add_progress_bar_metrics(metrics[DefaultMetricsKeys.PBAR])

        callback_metrics = metrics[DefaultMetricsKeys.CALLBACK]

        self._callback_metrics.update(callback_metrics)

        epoch_log_metrics = metrics[DefaultMetricsKeys.LOG]

        # add the metrics to the loggers
        if epoch_log_metrics and len(epoch_log_metrics) > 0:
            epoch_log_metrics["epoch"] = self.trainer.current_epoch
            self._logged_metrics.update(epoch_log_metrics)
            self.log_metrics(epoch_log_metrics, {})

        # reset result collection for next epoch
        self.trainer.result_collections.reset_metrics()

    """
    Utilities and properties
    """

    @property
    def callback_metrics(self) -> Dict[str, float]:
        if self.trainer.result_collections:
            metrics = self.trainer.result_collections.metrics[DefaultMetricsKeys.CALLBACK]
            self._callback_metrics.update(metrics)
            if os.getenv("PL_DEV_DEBUG", '0') == '1':
                self._callback_metrics["debug_epoch"] = self.trainer.current_epoch
        return self._callback_metrics

    @property
    def logged_metrics(self) -> Dict[str, float]:
        if self.trainer.result_collections:
            metrics = self.trainer.result_collections.metrics[DefaultMetricsKeys.LOG]
            self._logged_metrics.update(metrics)
        return self._logged_metrics

    @property
    def progress_bar_metrics(self) -> Dict[str, float]:
        if self.trainer.result_collections:
            metrics = self.trainer.result_collections.metrics[DefaultMetricsKeys.PBAR]
            self._progress_bar_metrics.update(metrics)
        return self._progress_bar_metrics

    def add_progress_bar_metrics(self, metrics: Dict[str, float]) -> None:
        self._progress_bar_metrics.update(metrics)
        self.trainer.dev_debugger.track_pbar_metrics_history(metrics)

    def add_logged_metrics(self, metrics: Dict[str, float]) -> None:
        self._logged_metrics.update(metrics)
        self.trainer.dev_debugger.track_logged_metrics_history(metrics)

    def add_callback_metrics(self, metrics: Dict[str, float]) -> None:
        self._callback_metrics.update(metrics)

    def check_logging(self, fx_name: str, on_step: bool, on_epoch: bool) -> None:
        self._fx_validator.check_logging(fx_name=fx_name, on_step=on_step, on_epoch=on_epoch)
