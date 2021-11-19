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
from pprint import pprint
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection, TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector.result import _METRICS, _OUT_DICT, _PBAR_DICT
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities import DeviceType, memory
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from pytorch_lightning.utilities.warnings import rank_zero_deprecation


class LoggerConnector:
    def __init__(self, trainer: "pl.Trainer", log_gpu_memory: Optional[str] = None) -> None:
        self.trainer = trainer
        if log_gpu_memory is not None:
            rank_zero_deprecation(
                "Setting `log_gpu_memory` with the trainer flag is deprecated in v1.5 and will be removed in v1.7. "
                "Please monitor GPU stats with the `DeviceStatsMonitor` callback directly instead."
            )
        self.log_gpu_memory = log_gpu_memory
        self.eval_loop_results: List[_OUT_DICT] = []
        self._val_log_step: int = 0
        self._test_log_step: int = 0
        self._progress_bar_metrics: _PBAR_DICT = {}
        self._logged_metrics: _OUT_DICT = {}
        self._callback_metrics: _OUT_DICT = {}
        self._gpus_metrics: Dict[str, float] = {}
        self._epoch_end_reached = False
        self._current_fx: Optional[str] = None
        self._batch_idx: Optional[int] = None
        self._split_idx: Optional[int] = None

    def on_trainer_init(
        self,
        logger: Union[bool, LightningLoggerBase, Iterable[LightningLoggerBase]],
        flush_logs_every_n_steps: Optional[int],
        log_every_n_steps: int,
        move_metrics_to_cpu: bool,
    ) -> None:
        self.configure_logger(logger)
        if flush_logs_every_n_steps is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(flush_logs_every_n_steps={flush_logs_every_n_steps})` is deprecated in v1.5 "
                "and will be removed in v1.7. Please configure flushing in the logger instead."
            )
        else:
            flush_logs_every_n_steps = 100  # original default parameter
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

    def configure_logger(self, logger: Union[bool, LightningLoggerBase, Iterable[LightningLoggerBase]]) -> None:
        if isinstance(logger, bool):
            # default logger
            self.trainer.logger = (
                TensorBoardLogger(
                    save_dir=self.trainer.default_root_dir, version=self.trainer.slurm_job_id, name="lightning_logs"
                )
                if logger
                else None
            )
        elif isinstance(logger, Iterable):
            self.trainer.logger = LoggerCollection(logger)
        else:
            self.trainer.logger = logger

    def log_metrics(self, metrics: _OUT_DICT, step: Optional[int] = None) -> None:
        """Logs the metric dict passed in. If `step` parameter is None and `step` key is presented is metrics, uses
        metrics["step"] as a step.

        Args:
            metrics: Metric values
            step: Step for which metrics should be logged. Default value is `self.global_step` during training or
                the total validation / test log step count during validation and testing.
        """
        if self.trainer.logger is None or not metrics:
            return

        # turn all tensors to scalars
        scalar_metrics = metrics_to_scalars(metrics)

        if step is None:
            step = scalar_metrics.pop("step", None)

        self._logged_metrics.update(scalar_metrics)

        if step is None:
            # added metrics for convenience
            scalar_metrics.setdefault("epoch", self.trainer.current_epoch)
            step = self.trainer.global_step

        # log actual metrics
        self.trainer.logger.agg_and_log_metrics(scalar_metrics, step=step)
        self.trainer.logger.save()

    """
    Evaluation metric updates
    """

    @property
    def _eval_log_step(self) -> Optional[int]:
        if self.trainer.state.stage is RunningStage.VALIDATING:
            return self._val_log_step
        if self.trainer.state.stage is RunningStage.TESTING:
            return self._test_log_step
        return None

    def _increment_eval_log_step(self) -> None:
        if self.trainer.state.stage is RunningStage.VALIDATING:
            self._val_log_step += 1
        elif self.trainer.state.stage is RunningStage.TESTING:
            self._test_log_step += 1

    def on_evaluation_batch_start(self, dataloader_idx: int, num_dataloaders: int) -> None:
        model = self.trainer.lightning_module
        # set dataloader_idx only if multiple ones
        model._current_dataloader_idx = dataloader_idx if num_dataloaders > 1 else None

    def update_eval_step_metrics(self) -> None:
        if self.trainer.sanity_checking:
            return

        # logs user requested information to logger
        assert not self._epoch_end_reached
        self.log_metrics(self.metrics["log"], step=self._eval_log_step)

        # increment the step even if nothing was logged
        self._increment_eval_log_step()

    @staticmethod
    def _filter_metrics_for_dataloader(
        dl_idx: int, metrics: _OUT_DICT, metric_prefix: str = "dataloader_idx"
    ) -> _OUT_DICT:
        return {k: v for k, v in metrics.items() if metric_prefix not in k or k.endswith(f"{metric_prefix}_{dl_idx}")}

    def _prepare_eval_loop_results(self, metrics: _OUT_DICT) -> None:
        if self.trainer.sanity_checking:
            return

        num_dataloaders = self.trainer._evaluation_loop.num_dataloaders
        has_been_initialized = len(self.eval_loop_results) == num_dataloaders
        for dl_idx in range(self.trainer._evaluation_loop.num_dataloaders):
            # remove callback metrics that don't belong to this dataloader
            callback_metrics = self._filter_metrics_for_dataloader(dl_idx, metrics)
            if has_been_initialized:
                self.eval_loop_results[dl_idx].update(callback_metrics)
            else:
                self.eval_loop_results.append(callback_metrics)

    def update_eval_epoch_metrics(self) -> List[_OUT_DICT]:
        assert self._epoch_end_reached
        metrics = self.metrics

        if not self.trainer.sanity_checking:
            # log all the metrics as a single dict
            self.log_metrics(metrics["log"])

        self._prepare_eval_loop_results(metrics["callback"])

        # log results of evaluation
        if (
            self.trainer.state.fn not in (TrainerFn.FITTING, TrainerFn.TUNING)
            and self.trainer.evaluating
            and self.trainer.is_global_zero
            and self.trainer.verbose_evaluate
        ):
            print("-" * 80)
            assert self.trainer.state.stage is not None
            for i, metrics_dict in enumerate(self.eval_loop_results):
                print(f"DATALOADER:{i} {self.trainer.state.stage.upper()} RESULTS")
                pprint(
                    {
                        k: (v.item() if v.numel() == 1 else v.tolist()) if isinstance(v, torch.Tensor) else v
                        for k, v in metrics_dict.items()
                    }
                )
                print("-" * 80)

        results = self.eval_loop_results
        # clear mem
        self.eval_loop_results = []
        return results

    """
    Train metric updates
    """

    def on_train_split_start(self, split_idx: int, split_batch: Any) -> None:
        self._split_idx = split_idx

    def update_train_step_metrics(self) -> None:
        if self.trainer.fit_loop._should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return

        # TODO: remove this call in v1.7
        self._log_gpus_metrics()

        # when metrics should be logged
        assert not self._epoch_end_reached
        if self.should_update_logs or self.trainer.fast_dev_run:
            self.log_metrics(self.metrics["log"])

    def update_train_epoch_metrics(self) -> None:
        # add the metrics to the loggers
        assert self._epoch_end_reached
        self.log_metrics(self.metrics["log"])

        # reset result collection for next epoch
        assert self.trainer._results is not None
        self.trainer._results.reset(metrics=True)

    def _log_gpus_metrics(self) -> None:
        """
        .. deprecated:: v1.5
            This function was deprecated in v1.5 in favor of
            `pytorch_lightning.accelerators.gpu._get_nvidia_gpu_stats` and will be removed in v1.7.
        """
        for key, mem in self.gpus_metrics.items():
            if self.log_gpu_memory == "min_max":
                self.trainer.lightning_module.log(key, mem, prog_bar=False, logger=True)
            else:
                gpu_id = int(key.split("/")[0].split(":")[1])
                if gpu_id in self.trainer._accelerator_connector.parallel_device_ids:
                    self.trainer.lightning_module.log(
                        key, mem, prog_bar=False, logger=True, on_step=True, on_epoch=False
                    )

    """
    Utilities and properties
    """

    def on_epoch_start(self) -> None:
        self._epoch_end_reached = False

    def on_batch_start(self, batch_idx: int, batch: Any) -> None:
        self._batch_idx = batch_idx
        self._epoch_end_reached = False

        assert self.trainer._results is not None
        # attach reference to the new batch and remove the cached batch_size
        self.trainer._results.batch = batch
        self.trainer._results.batch_size = None

    def epoch_end_reached(self) -> None:
        self._epoch_end_reached = True
        self._batch_idx = None
        self._split_idx = None
        assert self.trainer._results is not None

    def on_epoch_end(self) -> None:
        assert self._epoch_end_reached
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics["pbar"])
        self._callback_metrics.update(metrics["callback"])
        self._logged_metrics.update(metrics["log"])
        self._current_fx = None

    def on_batch_end(self) -> None:
        assert not self._epoch_end_reached
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics["pbar"])
        self._callback_metrics.update(metrics["callback"])
        self._logged_metrics.update(metrics["log"])

        assert self.trainer._results is not None
        # drop the reference to current batch and batch_size
        self.trainer._results.batch = None
        self.trainer._results.batch_size = None

    def should_reset_tensors(self, fx: str) -> bool:
        is_different_fx = self._current_fx != fx
        if self._split_idx is None:
            is_first_batch = self._batch_idx in (None, 0)
        else:
            is_first_batch = bool(self._batch_idx) + self._split_idx == 0
        return is_different_fx and is_first_batch

    def reset_metrics(self) -> None:
        self._progress_bar_metrics = {}
        self._logged_metrics = {}
        self._callback_metrics = {}

    def reset_results(self) -> None:
        if self.trainer._results is not None:
            self.trainer._results.reset()

        self._batch_idx = None
        self._split_idx = None
        self._current_fx = None

    @property
    def metrics(self) -> _METRICS:
        """This function returns either batch or epoch metrics depending on ``_epoch_end_reached``."""
        on_step = not self._epoch_end_reached
        assert self.trainer._results is not None
        return self.trainer._results.metrics(on_step)

    @property
    def gpus_metrics(self) -> Dict[str, float]:
        """
        .. deprecated:: v1.5
            Will be removed in v1.7.
        """
        if self.trainer._device_type == DeviceType.GPU and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            self._gpus_metrics.update(mem_map)
        return self._gpus_metrics

    @property
    def callback_metrics(self) -> _OUT_DICT:
        if self.trainer._results:
            metrics = self.metrics["callback"]
            self._callback_metrics.update(metrics)
        return self._callback_metrics

    @property
    def logged_metrics(self) -> _OUT_DICT:
        if self.trainer._results:
            metrics = self.metrics["log"]
            self._logged_metrics.update(metrics)
        return self._logged_metrics

    @property
    def progress_bar_metrics(self) -> _PBAR_DICT:
        if self.trainer._results:
            metrics = self.metrics["pbar"]
            self._progress_bar_metrics.update(metrics)
        return self._progress_bar_metrics

    def teardown(self) -> None:
        args = (torch.Tensor, move_data_to_device, "cpu")
        self._logged_metrics = apply_to_collection(self._logged_metrics, *args)
        self._progress_bar_metrics = apply_to_collection(self._progress_bar_metrics, *args)
        self._callback_metrics = apply_to_collection(self._callback_metrics, *args)
