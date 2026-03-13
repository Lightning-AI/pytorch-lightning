# Copyright The Lightning AI team.
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
VisualDL Logger
---------------
"""

import os
from argparse import Namespace
from typing import Any, Optional, Union

from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.loggers.visualdl import _VISUALDL_AVAILABLE
from lightning.fabric.loggers.visualdl import VisualDLLogger as FabricVisualDLLogger
from lightning.fabric.utilities.cloud_io import _is_dir
from lightning.fabric.utilities.logger import _convert_params
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn


class VisualDLLogger(Logger, FabricVisualDLLogger):
    r"""Log to local or remote file system in `VisualDL <https://www.paddlepaddle.org.cn/paddle/visualdl>`_ format.

    Implemented using :class:`visualdl.LogWriter`. Logs are saved to
    ``os.path.join(save_dir, name, version)``. This logger supports various visualization functions
    including scalar metrics, images, audio, text, histograms, PR curves, ROC curves, and high-dimensional data.

    This logger supports logging to remote filesystems via ``fsspec``.

    Example:

    .. testcode::
        :skipif: not _VISUALDL_AVAILABLE

        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import VisualDLLogger

        logger = VisualDLLogger("visualdl_logs", name="my_model")
        trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'lightning_logs'``. If it is the empty string then no per-experiment
            subdirectory is used.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph: Adds the computational graph to VisualDL. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        prefix: A string to put at the beginning of metric keys.
        sub_dir: Sub-directory to group VisualDL logs. If a sub_dir argument is passed
            then logs are saved in ``/save_dir/name/version/sub_dir/``. Defaults to ``None`` in which
            logs are saved in ``/save_dir/name/version/``.
        display_name: This parameter is displayed in the location of `Select Data Stream` in the panel.
            If not set, the default name is `logdir`.
        file_name: Set the name of the log file. If the file_name already exists, new records will be added
            to the same log file. Note that the name should include 'vdlrecords'.
        max_queue: The maximum capacity of the data generated before recording in a log file. Default value is 10.
            If the capacity is reached, the data are immediately written into the log file.
        flush_secs: The maximum cache time of the data generated before recording in a log file. Default value is 120.
            When this time is reached, the data are immediately written to the log file.
        filename_suffix: Add a suffix to the default log file name.
    """

    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(
        self,
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[_PATH] = None,
        display_name: Optional[str] = None,
        file_name: Optional[str] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        super().__init__(
            root_dir=save_dir,
            name=name,
            version=version,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            sub_dir=sub_dir,
            display_name=display_name,
            file_name=file_name,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        if log_graph and not _VISUALDL_AVAILABLE:
            rank_zero_warn(
                f"You set `VisualDLLogger(log_graph=True)` but `visualdl` is not available.\n{str(_VISUALDL_AVAILABLE)}"
            )
        self._log_graph = log_graph and _VISUALDL_AVAILABLE
        self.hparams: Union[dict[str, Any], Namespace] = {}

    @property
    @override
    def root_dir(self) -> str:
        """Parent directory for all VisualDL checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version"

        """
        return os.path.join(super().root_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        """The directory for this run's VisualDL checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    @override
    def save_dir(self) -> str:
        """Gets the save directory where the VisualDL experiments are saved.

        Returns:
            The local path to the save directory where the VisualDL experiments are saved.

        """
        return self._root_dir

    @override
    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        metrics: Optional[dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Record hyperparameters.

        Args:
            params: A dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Optional global step number for the logged metrics

        Note:
            VisualDL handles hyperparameters differently than TensorBoard. This implementation
            logs hyperparameters as text in a structured format for visualization in the
            hyper_parameters component.

        """
        if _OMEGACONF_AVAILABLE:
            from omegaconf import Container, OmegaConf

        params = _convert_params(params)

        # store params to output
        if _OMEGACONF_AVAILABLE and isinstance(params, Container):
            self.hparams = OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        return super().log_hyperparams(params=params, metrics=metrics, step=step)

    @override
    @rank_zero_only
    def log_graph(self, model: "pl.LightningModule", input_array: Optional[Tensor] = None) -> None:
        """Log the model graph to VisualDL.

        Note:
            VisualDL graph logging requires manual export of the model. You can use the VisualDL Graph component
            separately by launching visualdl with the --model parameter pointing to your saved model file.

        """
        if not self._log_graph:
            return

        rank_zero_warn(
            "VisualDL graph logging requires manual export of the model. "
            "You can use the VisualDL Graph component separately by launching "
            "visualdl with the --model parameter pointing to your saved model file."
        )

    @override
    @rank_zero_only
    def save(self) -> None:
        """Save hyperparameters to YAML file."""
        super().save()
        dir_path = self.log_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if _is_dir(self._fs, dir_path) and not self._fs.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize the logger."""
        super().finalize(status)
        if status == "success":
            # saving hparams happens independent of experiment manager
            self.save()

    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance

        """
        pass

    @override
    def _get_next_version(self) -> int:
        """Get the next available version number."""
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if _is_dir(self._fs, d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
