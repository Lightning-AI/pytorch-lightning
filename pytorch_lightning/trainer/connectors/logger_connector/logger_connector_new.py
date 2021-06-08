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
from pprint import pprint
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

import pytorch_lightning as pl
from pytorch_lightning.core import memory
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection, TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector.result_new import _METRIC, MetricSource
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities import DeviceType
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT


class LoggerConnector:

    def __init__(self, trainer: 'pl.Trainer', log_gpu_memory: Optional[str] = None) -> None:
        self.trainer = trainer
        self.log_gpu_memory = log_gpu_memory
        self.eval_loop_results = []
        self._val_log_step: int = 0
        self._test_log_step: int = 0
        self._progress_bar_metrics: Dict[str, float] = {}
        self._logged_metrics: Dict[str, _METRIC] = {}
        self._callback_metrics: Dict[str, _METRIC] = {}
        self._epoch_end_reached = False
        self._current_fx: Optional[str] = None
        self._batch_idx: Optional[int] = None
        self._split_idx: Optional[int] = None

    def on_trainer_init(
        self, logger: LightningLoggerBase, flush_logs_every_n_steps: int, log_every_n_steps: int,
        move_metrics_to_cpu: bool
    ) -> None:
        self.configure_logger(logger)
        self.trainer.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.trainer.log_every_n_steps = log_every_n_steps
        self.trainer.move_metrics_to_cpu = move_metrics_to_cpu

    @property
    def should_flush_logs(self) -> bool:
        should_flush = (self.trainer.global_step + 1) % self.trainer.flush_logs_every_n_steps == 0
        return should_flush or self.trainer.should_stop

    @property
    def should_update_logs(self) -> bool:
        should_log_every_n_steps = (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
        return should_log_every_n_steps or self.trainer.should_stop

    def configure_logger(self, logger: LightningLoggerBase) -> None:
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

    def log_metrics(self, metrics: Dict[str, _METRIC], step: Optional[int] = None) -> None:
        """Logs the metric dict passed in.
        If `step` parameter is None and `step` key is presented is metrics,
        uses metrics["step"] as a step

        Args:
            metrics: Metric values
            step: Step for which metrics should be logged. Default value is `self.global_step` during training or
                the total validation / test log step count during validation and testing.
        """
        # add gpu memory
        if self.trainer._device_type == DeviceType.GPU and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            metrics.update(mem_map)

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

            self._logged_metrics.update(scalar_metrics)

    """
    Evaluation metric updates
    """

    def prepare_eval_loop_results(self, metrics: Mapping[str, _METRIC]) -> None:
        if self.trainer.sanity_checking:
            return

        num_dataloaders = self.trainer.evaluation_loop.num_dataloaders
        has_been_initialized = len(self.eval_loop_results) == num_dataloaders
        for dl_idx in range(self.trainer.evaluation_loop.num_dataloaders):
            # remove callback metrics that don't belong to this dataloader
            callback_metrics = {
                k: v
                for k, v in metrics.items() if "dataloader_idx" not in k or f"dataloader_idx_{dl_idx}" in k
            }
            if has_been_initialized:
                self.eval_loop_results[dl_idx].update(callback_metrics)
            else:
                self.eval_loop_results.append(callback_metrics)

    def get_evaluate_epoch_results(self) -> _EVALUATE_OUTPUT:
        assert self._epoch_end_reached
        metrics = self.metrics

        if not self.trainer.sanity_checking:
            # log all the metrics as a single dict
            log_metrics = metrics[MetricSource.LOG]
            if log_metrics:
                self.log_metrics(log_metrics)

        self.prepare_eval_loop_results(metrics[MetricSource.CALLBACK])

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

    def on_evaluation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int, num_dataloaders: int) -> None:
        model = self.trainer.lightning_module
        # set dataloader_idx only if multiple ones
        model._current_dataloader_idx = dataloader_idx if num_dataloaders > 1 else None

        # track batch_size
        self.trainer.result_collection.extract_batch_size(batch)
        self._batch_idx = batch_idx

    def update_evaluation_step_metrics(self) -> None:
        if self.trainer.sanity_checking:
            return

        # logs user requested information to logger
        assert not self._epoch_end_reached
        metrics = self.metrics[MetricSource.LOG]
        if metrics:
            self.log_metrics(metrics, step=self.evaluation_log_step)

        # increment the step even if nothing was logged
        self.increment_evaluation_log_step()

    """
    Train metric updates
    """

    def on_train_split_start(self, batch_idx: int, split_idx: int, split_batch: Any) -> None:
        self.trainer.result_collection.extract_batch_size(split_batch)
        self._batch_idx = batch_idx
        self._split_idx = split_idx

    def update_train_step_metrics(self) -> None:
        if self.trainer.train_loop.should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return

        # when metrics should be logged
        assert not self._epoch_end_reached
        metrics = self.metrics[MetricSource.LOG]
        if self.should_update_logs or self.trainer.fast_dev_run is True and metrics:
            self.log_metrics(metrics)

    def update_train_epoch_metrics(self) -> None:
        # add the metrics to the loggers
        assert self._epoch_end_reached
        metrics = self.metrics[MetricSource.LOG]
        if metrics:
            self.log_metrics(metrics)

        # reset result collection for next epoch
        self.trainer.result_collection.reset(metrics=True)

    def teardown(self):
        self.trainer.train_loop.train_results.cpu()
        self.trainer.evaluation_loop.validation_results.cpu()
        self.trainer.evaluation_loop.test_results.cpu()

    """
    Utilities and properties
    """

    def on_epoch_start(self) -> None:
        self._epoch_end_reached = False

    def on_batch_start(self) -> None:
        self._epoch_end_reached = False

    def epoch_end_reached(self):
        self.trainer.logger_connector._epoch_end_reached = True
        self.trainer.logger_connector._batch_idx = None
        self.trainer.logger_connector._split_idx = None

    def on_epoch_end(self) -> None:
        assert self._epoch_end_reached
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics[MetricSource.PBAR])
        self._callback_metrics.update(metrics[MetricSource.CALLBACK])
        self._logged_metrics.update(metrics[MetricSource.LOG])
        self._current_fx = None

    def on_batch_end(self) -> None:
        assert not self._epoch_end_reached
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics[MetricSource.PBAR])
        self._callback_metrics.update(metrics[MetricSource.CALLBACK])
        self._logged_metrics.update(metrics[MetricSource.LOG])

    def should_reset_tensors(self, fx: str) -> bool:
        is_different_fx = self._current_fx != fx
        if self._split_idx is None:
            is_first_batch = self._batch_idx in (None, 0)
        else:
            is_first_batch = self._batch_idx + self._split_idx == 0
        return is_different_fx and is_first_batch

    def reset(self, metrics: Optional[bool] = None) -> None:
        self.trainer.result_collection.reset(metrics=metrics)
        self._batch_idx = None
        self._split_idx = None
        self._current_fx = None

    @property
    def metrics(self) -> Dict[MetricSource, Dict[str, _METRIC]]:
        """This function returns either batch or epoch metrics depending on ``_epoch_end_reached``."""
        on_step = not self._epoch_end_reached
        return self.trainer.result_collection.metrics(on_step)

    @property
    def callback_metrics(self) -> Dict[str, _METRIC]:
        if self.trainer.result_collection:
            metrics = self.metrics[MetricSource.CALLBACK]
            self._callback_metrics.update(metrics)
        return self._callback_metrics

    @property
    def logged_metrics(self) -> Dict[str, _METRIC]:
        if self.trainer.result_collection:
            metrics = self.metrics[MetricSource.LOG]
            self._logged_metrics.update(metrics)
        return self._logged_metrics

    @property
    def progress_bar_metrics(self) -> Dict[str, float]:
        if self.trainer.result_collection:
            metrics = self.metrics[MetricSource.PBAR]
            self._progress_bar_metrics.update(metrics)
        return self._progress_bar_metrics
