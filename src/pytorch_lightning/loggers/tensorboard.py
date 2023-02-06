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
TensorBoard Logger
------------------
"""

import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

from torch import Tensor

import pytorch_lightning as pl
from lightning_fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning_fabric.loggers.tensorboard import TensorBoardLogger as FabricTensorBoardLogger
from lightning_fabric.utilities.logger import _convert_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

log = logging.getLogger(__name__)

if _OMEGACONF_AVAILABLE:
    from omegaconf import Container, OmegaConf

# Skip doctests if requirements aren't available
if not (_TENSORBOARD_AVAILABLE or _TENSORBOARDX_AVAILABLE):
    __doctest_skip__ = ["TensorBoardLogger", "TensorBoardLogger.*"]


class TensorBoardLogger(Logger, FabricTensorBoardLogger):
    r"""
    Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format.

    Implemented using :class:`~tensorboardX.SummaryWriter`. Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    Example:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger("tb_logs", name="my_model")
        trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``. If it is the empty string then no per-experiment
            subdirectory is used.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        prefix: A string to put at the beginning of metric keys.
        sub_dir: Sub-directory to group TensorBoard logs. If a sub_dir argument is passed
            then logs are saved in ``/save_dir/name/version/sub_dir/``. Defaults to ``None`` in which
            logs are saved in ``/save_dir/name/version/``.
        \**kwargs: Additional arguments used by :class:`tensorboardX.SummaryWriter` can be passed as keyword
            arguments in this logger. To automatically flush to disk, `max_queue` sets the size
            of the queue for pending logs before flushing. `flush_secs` determines how many seconds
            elapses before flushing.

    Example:
        >>> import shutil, tempfile
        >>> tmp = tempfile.mkdtemp()
        >>> tbl = TensorBoardLogger(tmp)
        >>> tbl.log_hyperparams({"epochs": 5, "optimizer": "Adam"})
        >>> tbl.log_metrics({"acc": 0.75})
        >>> tbl.log_metrics({"acc": 0.9})
        >>> tbl.finalize("success")
        >>> shutil.rmtree(tmp)
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
        **kwargs: Any,
    ):
        super().__init__(
            root_dir=save_dir,
            name=name,
            version=version,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            sub_dir=sub_dir,
            **kwargs,
        )
        if log_graph and not _TENSORBOARD_AVAILABLE:
            rank_zero_warn("You set `TensorBoardLogger(log_graph=True)` but `tensorboard` is not available.")
        self._log_graph = log_graph and _TENSORBOARD_AVAILABLE
        self.hparams: Union[Dict[str, Any], Namespace] = {}

    @property
    def root_dir(self) -> str:
        """Parent directory for all tensorboard checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version"
        """
        return os.path.join(super().root_dir, self.name)

    @property
    def log_dir(self) -> str:
        """The directory for this run's tensorboard checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def save_dir(self) -> str:
        """Gets the save directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the save directory where the TensorBoard experiments are saved.
        """
        return self._root_dir

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record hyperparameters. TensorBoard logs with and without saved hyperparameters are incompatible, the
        hyperparameters are then not displayed in the TensorBoard. Please delete or move the previously saved logs
        to display the new ones with hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
        """
        params = _convert_params(params)

        # store params to output
        if _OMEGACONF_AVAILABLE and isinstance(params, Container):
            self.hparams = OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        return super().log_hyperparams(params=params, metrics=metrics)

    @rank_zero_only
    def log_graph(self, model: "pl.LightningModule", input_array: Optional[Tensor] = None) -> None:
        if not self._log_graph:
            return

        input_array = model.example_input_array if input_array is None else input_array

        if input_array is None:
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `model.example_input_array` attribute"
                " is not set or `input_array` was not given."
            )
        elif not isinstance(input_array, (Tensor, tuple)):
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `input_array` or `model.example_input_array`"
                f" has type {type(input_array)} which can't be traced by TensorBoard. Make the input array a tuple"
                f" representing the positional arguments to the model's `forward()` implementation."
            )
        else:
            input_array = model._on_before_batch_transfer(input_array)
            input_array = model._apply_batch_transfer_handler(input_array)
            with pl.core.module._jit_is_scripting():
                self.experiment.add_graph(model, input_array)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        dir_path = self.log_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if self._fs.isdir(dir_path) and not self._fs.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if status == "success":
            # saving hparams happens independent of experiment manager
            self.save()

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        pass

    def _get_next_version(self) -> int:
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
