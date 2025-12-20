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
"""Fabric/PyTorch Lightning logger that enables remote experiment tracking, logging, and artifact management on
lightning.ai."""

import logging
import os
import warnings
from argparse import Namespace
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Union

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.utilities import _scan_checkpoints

if TYPE_CHECKING:
    from litlogger import Experiment

log = logging.getLogger(__name__)

_LITLOGGER_AVAILABLE = RequirementCache("litlogger>=0.1.0")


def _create_experiment_name() -> str:
    """Create a random experiment name using litlogger's generator."""
    from litlogger.generator import _create_name

    return _create_name()


class LitLogger(Logger):
    """Logger that enables remote experiment tracking, logging, and artifact management on lightning.ai."""

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: Optional[_PATH] = None,
        name: Optional[str] = None,
        teamspace: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        store_step: bool = True,
        log_model: bool = False,
        save_logs: bool = True,
        checkpoint_name: Optional[str] = None,
    ) -> None:
        """Initialize the LightningLogger.

        Args:
            root_dir: Folder where logs and metadata are stored (default: ./lightning_logs).
            name: Name of your experiment (defaults to a generated name).
            teamspace: Teamspace name where charts and artifacts will appear.
            metadata: Extra metadata to associate with the experiment as tags.
            log_model: If True, automatically log model checkpoints as artifacts.
            save_logs: If True, capture and upload terminal logs.
            checkpoint_name: Override the base name for logged checkpoints.

        Example::

            from lightning.pytorch import Trainer
            from lightning.pytorch.demos.boring_classes import BoringModel, BoringDataModule
            from lightning.pytorch.loggers.litlogger import LitLogger

            class LoggingModel(BoringModel):
                def training_step(self, batch, batch_idx: int):
                    loss = self.step(batch)
                    # logging the computed loss
                    self.log("train_loss", loss)
                    return {"loss": loss}

            trainer = Trainer(
                max_epochs=10,
                enable_model_summary=False,
                logger=LitLogger("./lightning_logs", name="boring_model")
            )
            model = BoringModel()
            data_module = BoringDataModule()
            trainer.fit(model, data_module)
            trainer.test(model, data_module)

        """
        self._root_dir = os.fspath(root_dir or "./lightning_logs")
        self._name = name or _create_experiment_name()
        self._version: Optional[str] = None
        self._teamspace = teamspace
        self._sub_dir = None
        self._prefix = ""
        self._fs = get_filesystem(self._root_dir)
        self._step = -1
        self._metadata = metadata or {}
        self._is_ready = False
        self._log_model = log_model
        self._save_logs = save_logs
        self._checkpoint_callback: Optional[ModelCheckpoint] = None
        self._logged_model_time: dict[str, float] = {}
        self._checkpoint_name = checkpoint_name

    # ──────────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────────

    @property
    @override
    def name(self) -> str:
        """Gets the name of the experiment."""
        return self._name

    @property
    @override
    def version(self) -> Optional[str]:
        """Get the experiment version - its time of creation."""
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Gets the save directory where the litlogger experiments are saved."""
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        """The directory for this run's tensorboard checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        log_dir = os.path.join(self.root_dir, self.name)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        return os.path.expanduser(log_dir)

    @property
    def save_dir(self) -> str:
        return self.log_dir

    @property
    def sub_dir(self) -> Optional[str]:
        """Gets the sub directory where the TensorBoard experiments are saved."""
        return self._sub_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> Optional["Experiment"]:
        """Returns the underlying litlogger Experiment object."""
        import litlogger

        if litlogger.experiment is not None:
            return litlogger.experiment

        if not self._is_ready:
            self._is_ready = True

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)

        if self.version is None:
            # Generate version as proper RFC 3339 timestamp with Z suffix (required by protobuf)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            self._version = timestamp.replace("+00:00", "Z")

        litlogger.init(
            name=self._name,
            root_dir=self._root_dir,
            teamspace=self._teamspace,
            metadata={k: str(v) for k, v in self._metadata.items()},
            store_step=True,
            store_created_at=True,
            save_logs=self._save_logs,
        )

        return litlogger.experiment

    @property
    @rank_zero_only
    def url(self) -> str:
        return self.experiment.url

    # ──────────────────────────────────────────────────────────────────────────────
    # Override methods from Logger
    # ──────────────────────────────────────────────────────────────────────────────

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        import litlogger

        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # Ensure experiment is initialized
        _ = self.experiment

        self._step = self._step + 1 if step is None else step

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        metrics = {k: v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}
        litlogger.log_metrics(metrics, step=self._step)

    @override
    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log hyperparams."""
        if isinstance(params, Namespace):
            params = params.__dict__
        params.update(self._metadata or {})
        self._metadata = params

    @override
    @rank_zero_only
    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        warnings.warn("LitLogger does not support `log_graph`", UserWarning, stacklevel=2)

    @override
    @rank_zero_only
    def save(self) -> None:
        pass

    @override
    @rank_zero_only
    def finalize(self, status: Optional[str] = None) -> None:
        import litlogger

        if litlogger.experiment is not None:
            # log checkpoints as artifacts before finalizing
            if self._checkpoint_callback:
                self._scan_and_log_checkpoints(self._checkpoint_callback)
            litlogger.finalize(status)

    # ──────────────────────────────────────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────────────────────────────────────

    @rank_zero_only
    def log_metadata(
        self,
        params: Union[dict[str, Any], Namespace],
    ) -> None:
        """Log hyperparams."""
        if isinstance(params, Namespace):
            params = params.__dict__
        params.update(self._metadata or {})
        self._metadata = params

    @rank_zero_only
    def log_model(
        self,
        model: Any,
        staging_dir: Optional[str] = None,
        verbose: bool = False,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save and upload a model object to cloud storage.

        Args:
            model: The model object to save and upload (e.g., torch.nn.Module).
            staging_dir: Optional local directory for staging the model before upload.
            verbose: Whether to show progress bar during upload.
            version: Optional version string for the model.
            metadata: Optional metadata dictionary to store with the model.

        """
        import litlogger

        _ = self.experiment
        litlogger.log_model(model, staging_dir, verbose, version, metadata)

    @rank_zero_only
    def log_model_artifact(
        self,
        path: str,
        verbose: bool = False,
        version: Optional[str] = None,
    ) -> None:
        """Upload a model file or directory to cloud storage using litmodels.

        Args:
            path: Path to the local model file or directory to upload.
            verbose: Whether to show progress bar during upload. Defaults to False.
            version: Optional version string for the model. Defaults to the experiment version.

        """
        import litlogger

        _ = self.experiment
        litlogger.log_model_artifact(path, verbose, version)

    @rank_zero_only
    def get_file(self, path: str, verbose: bool = True) -> str:
        """Download a file artifact from the cloud for this experiment.

        Args:
            path: Path where the file should be saved locally.
            verbose: Whether to print a confirmation message after download. Defaults to True.

        Returns:
            str: The local path where the file was saved.

        """
        import litlogger

        _ = self.experiment
        return litlogger.get_file(path, verbose=verbose)

    @rank_zero_only
    def get_model(self, staging_dir: Optional[str] = None, verbose: bool = False, version: Optional[str] = None) -> Any:
        """Download and load a model object using litmodels.

        Args:
            staging_dir: Optional directory where the model will be downloaded.
            verbose: Whether to show progress bar.
            version: Optional version string for the model.

        Returns:
            The loaded model object.

        """
        import litlogger

        _ = self.experiment
        return litlogger.get_model(staging_dir, verbose, version)

    @rank_zero_only
    def get_model_artifact(self, path: str, verbose: bool = False, version: Optional[str] = None) -> str:
        """Download a model artifact file or directory from cloud storage using litmodels.

        Args:
            path: Path where the model should be saved locally.
            verbose: Whether to show progress bar during download.
            version: Optional version string for the model.

        Returns:
            str: The local path where the model was saved.

        """
        import litlogger

        _ = self.experiment
        return litlogger.get_model_artifact(path, verbose, version)

    @rank_zero_only
    def log_file(self, path: str) -> None:
        """Log a file as an artifact to the Lightning platform.

        The file will be logged in the Teamspace drive,
        under a folder identified by the experiment name.

        Args:
            path: Path to the file to log.

        Example::
            logger = LitLogger(...)
            logger.log_file('config.yaml')

        """
        import litlogger

        _ = self.experiment
        litlogger.log_file(path)

    # ──────────────────────────────────────────────────────────────────────────────
    # Callback methods
    # ──────────────────────────────────────────────────────────────────────────────

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Called after a checkpoint is saved.

        Logs checkpoints as artifacts if enabled.

        """
        if self._log_model is False:
            return
        if checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        else:
            self._checkpoint_callback = checkpoint_callback

    # ──────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ──────────────────────────────────────────────────────────────────────────────

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Find new checkpoints from the callback and log them as model artifacts."""
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        for timestamp, path_ckpt, _score, tag in checkpoints:
            if not self._checkpoint_name:
                self._checkpoint_name = self.experiment.name
            # Ensure the version tag is unique by appending a timestamp
            unique_tag = f"{tag}-{int(datetime.now(timezone.utc).timestamp())}"
            self.log_model_artifact(path_ckpt, verbose=True, version=unique_tag)
            # remember logged models - timestamp needed in case filename didn't change
            self._logged_model_time[path_ckpt] = timestamp
