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
from typing import Any, Iterable, Optional, Union

from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment
from pytorch_lightning.trainer.connectors.logger_connector.result import _METRICS, _OUT_DICT, _PBAR_DICT
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class LoggerConnector:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.trainer = trainer
        self._progress_bar_metrics: _PBAR_DICT = {}
        self._logged_metrics: _OUT_DICT = {}
        self._callback_metrics: _OUT_DICT = {}
        self._epoch_end_reached = False
        self._current_fx: Optional[str] = None
        self._batch_idx: Optional[int] = None
        self._split_idx: Optional[int] = None
        self._override_agg_and_log_metrics: bool = False

    def on_trainer_init(
        self,
        logger: Union[bool, Logger, Iterable[Logger]],
        log_every_n_steps: int,
        move_metrics_to_cpu: bool,
    ) -> None:
        self.configure_logger(logger)
        self.trainer.log_every_n_steps = log_every_n_steps
        self.trainer.move_metrics_to_cpu = move_metrics_to_cpu
        for logger in self.trainer.loggers:
            if is_overridden("agg_and_log_metrics", logger, Logger):
                self._override_agg_and_log_metrics = True
                rank_zero_deprecation(
                    "`Logger.agg_and_log_metrics` is deprecated in v1.6 and will be removed"
                    " in v1.8. `Trainer` will directly call `Logger.log_metrics` so custom"
                    " loggers should not implement `Logger.agg_and_log_metrics`."
                )
                break

    @property
    def should_update_logs(self) -> bool:
        # `+ 1` because it can be checked before a step is executed, for example, in `on_train_batch_start`
        should_log = (self.trainer.fit_loop.epoch_loop._batches_that_stepped + 1) % self.trainer.log_every_n_steps == 0
        return should_log or self.trainer.should_stop

    def configure_logger(self, logger: Union[bool, Logger, Iterable[Logger]]) -> None:
        if not logger:
            # logger is None or logger is False
            self.trainer.loggers = []
        elif logger is True:
            # default logger
            self.trainer.loggers = [
                TensorBoardLogger(save_dir=self.trainer.default_root_dir, version=SLURMEnvironment.job_id())
            ]
        elif isinstance(logger, Iterable):
            self.trainer.loggers = list(logger)
        else:
            self.trainer.loggers = [logger]

    def log_metrics(self, metrics: _OUT_DICT, step: Optional[int] = None) -> None:
        """Logs the metric dict passed in. If `step` parameter is None and `step` key is presented is metrics, uses
        metrics["step"] as a step.

        Args:
            metrics: Metric values
            step: Step for which metrics should be logged. Default value is `self.global_step` during training or
                the total validation / test log step count during validation and testing.
        """
        if not self.trainer.loggers or not metrics:
            return

        self._logged_metrics.update(metrics)

        # turn all tensors to scalars
        scalar_metrics = metrics_to_scalars(metrics)

        if step is None:
            step = scalar_metrics.pop("step", None)

        if step is None:
            # added metrics for convenience
            scalar_metrics.setdefault("epoch", self.trainer.current_epoch)
            step = self.trainer.fit_loop.epoch_loop._batches_that_stepped

        # log actual metrics
        for logger in self.trainer.loggers:
            if self._override_agg_and_log_metrics:
                logger.agg_and_log_metrics(metrics=scalar_metrics, step=step)
            else:
                logger.log_metrics(metrics=scalar_metrics, step=step)
            logger.save()

    """
    Evaluation metric updates
    """

    def _evaluation_epoch_end(self) -> None:
        results = self.trainer._results
        assert results is not None
        results.dataloader_idx = None

    def update_eval_step_metrics(self, step: int) -> None:
        assert not self._epoch_end_reached
        # logs user requested information to logger
        self.log_metrics(self.metrics["log"], step=step)

    def update_eval_epoch_metrics(self) -> _OUT_DICT:
        assert self._epoch_end_reached
        if self.trainer.sanity_checking:
            return {}
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics["pbar"])
        self._callback_metrics.update(metrics["callback"])
        self._logged_metrics.update(metrics["log"])
        return metrics["log"]

    def log_eval_end_metrics(self, metrics: _OUT_DICT) -> None:
        assert self._epoch_end_reached
        if self.trainer.sanity_checking:
            return

        # log all the metrics as a single dict
        self.log_metrics(metrics)

    """
    Train metric updates
    """

    def on_train_split_start(self, split_idx: int) -> None:
        self._split_idx = split_idx

    def update_train_step_metrics(self) -> None:
        if self.trainer.fit_loop._should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return

        # when metrics should be logged
        assert not self._epoch_end_reached
        if self.should_update_logs or self.trainer.fast_dev_run:
            self.log_metrics(self.metrics["log"])

    def update_train_epoch_metrics(self) -> None:
        # add the metrics to the loggers
        assert self._epoch_end_reached
        self.log_metrics(self.metrics["log"])

        # reset result collection for next epoch
        self.reset_results()

    """
    Utilities and properties
    """

    def on_epoch_start(self) -> None:
        self._epoch_end_reached = False

    def on_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> None:
        self._batch_idx = batch_idx
        self._epoch_end_reached = False

        results = self.trainer._results
        assert results is not None
        # attach reference to the new batch and remove the cached batch_size
        results.batch = batch
        results.batch_size = None
        results.dataloader_idx = dataloader_idx

    def epoch_end_reached(self) -> None:
        self._epoch_end_reached = True
        self._batch_idx = None
        self._split_idx = None

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
        results = self.trainer._results
        if results is not None:
            results.reset()

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
        args = (Tensor, move_data_to_device, "cpu")
        self._logged_metrics = apply_to_collection(self._logged_metrics, *args)
        self._progress_bar_metrics = apply_to_collection(self._progress_bar_metrics, *args)
        self._callback_metrics = apply_to_collection(self._callback_metrics, *args)
