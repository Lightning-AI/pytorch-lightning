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
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

log = logging.getLogger(__name__)


class ExperimentWriter:
    r"""
    Experiment writer for CSVLogger.

    Currently supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_HPARAMS_FILE = "hparams.yaml"
    NAME_METRICS_FILE = "metrics.csv"

    def __init__(self, log_dir: str) -> None:
        self.hparams: Dict[str, Any] = {}
        self.metrics: List[Dict[str, float]] = []

        self.log_dir = log_dir
        if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.metrics:
            return

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(self.metrics)


class CSVLogger(Logger):
    r"""
    Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: str,
        name: str = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._experiment: Optional[ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version"
        """
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> str:
        """The current directory where logs are saved.

        Returns:
            The path to current directory where logs are saved.
        """
        return self._save_dir

    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.module.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics, step)
        if step is not None and (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.
        """
        return self._name

    @property
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.
        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self) -> int:
        root_dir = self.root_dir

        if not os.path.isdir(root_dir):
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
