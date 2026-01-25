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
SwanLab Logger
--------------
"""

import os
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override

from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_json_serializable,
    _convert_params,
    _sanitize_callable_params,
)
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    pass

_SWANLAB_AVAILABLE = RequirementCache("swanlab>=0.3.0")


class SwanLabLogger(Logger):
    r"""Log using `SwanLab <https://swanlab.cn>`_.

    **Installation and set-up**

    Install with pip:

    .. code-block:: bash

        pip install swanlab

    Create a `SwanLabLogger` instance:

    .. code-block:: python

        from lightning.pytorch.loggers import SwanLabLogger

        swanlab_logger = SwanLabLogger(project="my_project")

    Pass the logger instance to the `Trainer`:

    .. code-block:: python

        trainer = Trainer(logger=swanlab_logger)

    A new SwanLab run will be created when training starts if you have not created one manually before with `swanlab.init()`.

    **Log metrics**

    Log from :class:`~lightning.pytorch.core.LightningModule`:

    .. code-block:: python

        class LitModule(LightningModule):
            def training_step(self, batch, batch_idx):
                self.log("train/loss", loss)

    Use directly swanlab module:

    .. code-block:: python

        import swanlab
        swanlab.log({"train/loss": loss})

    **Log hyper-parameters**

    Save :class:`~lightning.pytorch.core.LightningModule` parameters:

    .. code-block:: python

        class LitModule(LightningModule):
            def __init__(self, *args, **kwarg):
                self.save_hyperparameters()

    Add other config parameters:

    .. code-block:: python

        # add one parameter
        swanlab_logger.experiment.config["key"] = value

        # add multiple parameters
        swanlab_logger.experiment.config.update({key1: val1, key2: val2})

        # use directly swanlab module
        import swanlab
        swanlab.config["key"] = value
        swanlab.config.update()

    **Log media**

    Log images with:

    .. code-block:: python

        # using tensors, numpy arrays or PIL images
        swanlab_logger.log_image(key="samples", images=[img1, img2])

        # adding captions
        swanlab_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

    Log text with:

    .. code-block:: python

        # using columns and data
        columns = ["input", "label", "prediction"]
        data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]
        swanlab_logger.log_text(key="samples", columns=columns, data=data)

    **Log model checkpoints**

    Log model checkpoints at the end of training:

    .. code-block:: python

        swanlab_logger = SwanLabLogger(log_model=True)

    Log model checkpoints as they get created during training:

    .. code-block:: python

        swanlab_logger = SwanLabLogger(log_model="all")

    See Also:
        - `SwanLab Documentation <https://docs.swanlab.cn>`__

    Args:
        project: The project name for SwanLab. If not set, defaults to ``'lightning_logs'``.
        experiment_name: The experiment name. If not set, SwanLab will generate one automatically.
        description: A description of the experiment.
        logdir: Directory to save logs locally. If not set, SwanLab will use its default location.
        mode: Mode for SwanLab ('cloud', 'local', 'offline', 'disabled'). Default is 'cloud'.
        save_dir: Same as logdir.
        version: Sets the version, mainly used to resume a previous run.
        offline: Run offline (data can be synced later to SwanLab servers).
        prefix: A string to put at the beginning of metric keys.
        experiment: SwanLab experiment object. Automatically set when creating a run.
        config: Dictionary of hyperparameters to log.
        log_model: Log checkpoints created by :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~lightning.pytorch.callbacks.ModelCheckpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        \**kwargs: Arguments passed to :func:`swanlab.init`.

    Raises:
        ModuleNotFoundError:
            If required SwanLab package is not installed on the device.

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        project: Optional[str] = None,
        experiment_name: Optional[str] = None,
        description: Optional[str] = None,
        logdir: Optional[_PATH] = None,
        mode: str = "cloud",
        save_dir: Optional[_PATH] = None,
        version: Optional[str] = None,
        offline: bool = False,
        prefix: str = "",
        experiment: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        log_model: Union[Literal["all"], bool] = False,
        **kwargs: Any,
    ) -> None:
        if not _SWANLAB_AVAILABLE:
            raise ModuleNotFoundError(str(_SWANLAB_AVAILABLE))

        super().__init__()
        self._project = project
        self._experiment_name = experiment_name
        self._description = description
        self._logdir = logdir or save_dir
        self._mode = mode
        self._prefix = prefix
        self._experiment = experiment
        self._config = config or {}
        self._offline = offline
        self._kwargs = kwargs
        self._log_model = log_model
        self._logged_model_time: dict[str, float] = {}
        self._checkpoint_callbacks: dict[int, ModelCheckpoint] = {}

        # Process paths
        if self._logdir is not None:
            self._logdir = os.fspath(self._logdir)

        # Set default project name
        if self._project is None:
            self._project = os.environ.get("SWANLAB_PROJECT", "lightning_logs")

        # Set mode based on offline flag
        if self._offline:
            self._mode = "offline"

        # Set version
        self._id = version

    def __getstate__(self) -> dict[str, Any]:
        import swanlab

        # Hack: If the 'spawn' launch method is used, the logger will get pickled and this `__getstate__` gets called.
        # We create an experiment here in the main process, and attach to it in the worker process.
        _ = self.experiment

        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "_run_id", None)
            state["_name"] = self._experiment.name

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Any:
        r"""Actual swanlab object. To use swanlab features in your :class:`~lightning.pytorch.core.LightningModule` do the
        following.

        Example::

        .. code-block:: python

            self.logger.experiment.some_swanlab_function()

        """
        import swanlab

        if self._experiment is None:
            # Check if there's a valid swanlab run already in progress
            existing_run = swanlab.run.get_run() if hasattr(swanlab.run, 'get_run') else None
            if existing_run is not None:
                # swanlab process already created in this instance
                rank_zero_warn(
                    "There is a swanlab run already in progress and newly created instances of `SwanLabLogger` will reuse"
                    " this run. If this is not desired, call `swanlab.finish()` before instantiating `SwanLabLogger`."
                )
                self._experiment = existing_run
            else:
                # create new swanlab process
                init_kwargs = {
                    "project": self._project,
                    "experiment_name": self._experiment_name,
                    "description": self._description,
                    "logdir": self._logdir,
                    "mode": self._mode,
                    "config": self._config,
                }
                init_kwargs.update(self._kwargs)

                if self._id is not None:
                    init_kwargs["resume"] = self._id

                self._experiment = swanlab.init(**init_kwargs)

        return self._experiment

    def watch(
        self, model: nn.Module, log: Optional[str] = "gradients", log_freq: int = 100, log_graph: bool = True
    ) -> None:
        """Watch model gradients and parameters.

        Note:
            SwanLab does not currently support the watch() method. This call will be ignored.

        """
        rank_zero_warn("SwanLab does not support the watch() method. This call will be ignored.")

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        params = _convert_json_serializable(params)

        # Update config if experiment already exists
        if self._experiment is not None:
            if hasattr(self._experiment, "config"):
                self._experiment.config.update(params)

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        
        # Ensure experiment is initialized before logging
        _ = self.experiment
        import swanlab
        
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        # Convert torch.Tensor to scalar if needed
        log_dict = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            log_dict[key] = value

        if step is not None:
            swanlab.log(log_dict, step=step)
        else:
            swanlab.log(log_dict)

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: Optional[list[str]] = None,
        data: Optional[list[list[Any]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log a Table containing any object type (text, image, audio, video, etc).

        Can be defined either with `columns` and `data` or with `dataframe`.

        """
        import swanlab

        # Convert table to text representation
        if dataframe is not None:
            text_content = dataframe.to_string()
        elif columns and data:
            header = " | ".join(columns)
            rows = [" | ".join(str(cell) for cell in row) for row in data]
            text_content = header + "\n" + "-" * len(header) + "\n" + "\n".join(rows)
        else:
            text_content = ""

        metrics = {key: swanlab.Text(text_content)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_text(
        self,
        key: str,
        columns: Optional[list[str]] = None,
        data: Optional[list[list[str]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log text as a Table.

        Can be defined either with `columns` and `data` or with `dataframe`.

        """
        self.log_table(key, columns, data, dataframe, step)

    @rank_zero_only
    def log_image(self, key: str, images: list[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        """Log images (tensors, numpy arrays, PIL Images or file paths).

        Optional kwargs are lists passed to each image (ex: caption).

        """
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")

        import swanlab

        captions = kwargs.get("caption", [None] * n)
        if not isinstance(captions, list):
            captions = [captions] * n

        swanlab_images = [swanlab.Image(img, caption=cap) for img, cap in zip(images, captions)]
        metrics = {key: swanlab_images}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_audio(self, key: str, audios: list[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        r"""Log audios (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the audio files
            audios: The list of audio file paths, or numpy arrays to be logged
            step: The step number to be used for logging the audio files
            \**kwargs: Optional kwargs are lists passed to each ``swanlab.Audio`` instance (ex: caption, sample_rate).

        """
        if not isinstance(audios, list):
            raise TypeError(f'Expected a list as "audios", found {type(audios)}')
        n = len(audios)
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")

        import swanlab

        captions = kwargs.get("caption", [None] * n)
        sample_rates = kwargs.get("sample_rate", [None] * n)
        if not isinstance(captions, list):
            captions = [captions] * n
        if not isinstance(sample_rates, list):
            sample_rates = [sample_rates] * n

        swanlab_audios = []
        for audio, cap, sr in zip(audios, captions, sample_rates):
            audio_kwargs = {"caption": cap}
            if sr is not None:
                audio_kwargs["sample_rate"] = sr
            swanlab_audios.append(swanlab.Audio(audio, **audio_kwargs))

        metrics = {key: swanlab_audios}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_video(self, key: str, videos: list[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        """Log videos (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the video files
            videos: The list of video file paths, or numpy arrays to be logged
            step: The step number to be used for logging the video files
            **kwargs: Optional kwargs are lists passed to each swanlab.Video instance (ex: caption, fps, format).

        """
        if not isinstance(videos, list):
            raise TypeError(f'Expected a list as "videos", found {type(videos)}')
        n = len(videos)
        for k, v in kwargs.items():
            if isinstance(v, list) and len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")

        import swanlab

        captions = kwargs.get("caption", [None] * n)
        if not isinstance(captions, list):
            captions = [captions] * n

        swanlab_videos = [swanlab.Video(video, caption=cap) for video, cap in zip(videos, captions)]
        metrics = {key: swanlab_videos}
        self.log_metrics(metrics, step)

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        return self._logdir

    @property
    @override
    def name(self) -> Optional[str]:
        """The project name of this experiment.

        Returns:
            The name of the project the current experiment belongs to.

        """
        return self._project

    @property
    @override
    def version(self) -> Optional[str]:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if the experiment exists else the id given to the constructor.

        """
        # don't create an experiment if we don't have one
        if self._experiment is not None:
            return getattr(self._experiment, "_run_id", None)
        return self._id

    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Log checkpoints after they are saved."""
        if self._log_model == "all" or (self._log_model is True and checkpoint_callback.save_top_k == -1):
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callbacks[id(checkpoint_callback)] = checkpoint_callback

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Scan and log checkpoints."""
        import swanlab

        # Get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # Log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
            }
            # Log checkpoint info as metrics (SwanLab doesn't support artifacts like WandB)
            if self._experiment is not None:
                swanlab.log({
                    "checkpoint/path": str(p),
                    "checkpoint/score": metadata["score"] if metadata["score"] is not None else 0.0,
                    "checkpoint/filename": metadata["original_filename"],
                })
            # Remember logged models - timestamp needed in case filename didn't change
            self._logged_model_time[p] = t

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize the SwanLab experiment.

        Args:
            status: Status string (e.g., 'success', 'failed').

        """
        # Log remaining checkpoints on success
        if status == "success" and self._experiment is not None:
            for checkpoint_callback in self._checkpoint_callbacks.values():
                self._scan_and_log_checkpoints(checkpoint_callback)

        if self._experiment is not None:
            try:
                import swanlab
                swanlab.finish()
            except RuntimeError:
                # Ignore error if swanlab.init() was never called
                pass
            except Exception:
                # Ignore other errors during cleanup
                pass
