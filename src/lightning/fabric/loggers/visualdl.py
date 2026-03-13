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

import os
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from lightning.fabric.utilities.logger import _sanitize_params as _utils_sanitize_params
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH
from lightning.fabric.wrappers import _unwrap_objects

_VISUALDL_AVAILABLE = RequirementCache("visualdl")

if TYPE_CHECKING:
    if _VISUALDL_AVAILABLE:
        from visualdl import LogWriter


class VisualDLLogger(Logger):
    r"""Log to local file system in `VisualDL <https://www.paddlepaddle.org.cn/paddle/visualdl>`_ format.

    Implemented using :class:`visualdl.LogWriter`. Logs are saved to
    ``os.path.join(root_dir, name, version)``. This logger supports various visualization functions
    including scalar metrics, images, audio, text, histograms, PR curves, ROC curves, and high-dimensional data.

    Args:
        root_dir: The root directory in which all your experiments with different names and versions will be stored.
        name: Experiment name. Defaults to ``'lightning_logs'``. If it is the empty string then no per-experiment
            subdirectory is used.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to ``log_hyperparams`` without a metric are ignored).
        prefix: A string to put at the beginning of all metric keys.
        sub_dir: Sub-directory to group VisualDL logs. If a ``sub_dir`` argument is passed
            then logs are saved in ``/root_dir/name/version/sub_dir/``. Defaults to ``None`` in which case
            logs are saved in ``/root_dir/name/version/``.
        display_name: This parameter is displayed in the location of `Select Data Stream` in the panel.
            If not set, the default name is `logdir`.
        file_name: Set the name of the log file. If the file_name already exists, new records will be added
            to the same log file. Note that the name should include 'vdlrecords'.
        max_queue: The maximum capacity of the data generated before recording in a log file. Default value is 10.
            If the capacity is reached, the data are immediately written into the log file.
        flush_secs: The maximum cache time of the data generated before recording in a log file. Default value is 120.
            When this time is reached, the data are immediately written to the log file.
        filename_suffix: Add a suffix to the default log file name.

    Example::

        from lightning.fabric.loggers import VisualDLLogger

        logger = VisualDLLogger("path/to/logs/root", name="my_model")
        logger.log_hyperparams({"epochs": 5, "optimizer": "Adam"})
        logger.log_metrics({"acc": 0.75})
        logger.finalize("success")

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[_PATH] = None,
        display_name: Optional[str] = None,
        file_name: Optional[str] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        if not _VISUALDL_AVAILABLE:
            raise ModuleNotFoundError(
                f"`visualdl` is not available. Try `pip install visualdl` to install it.\n{str(_VISUALDL_AVAILABLE)}"
            )
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)

        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)

        self._display_name = display_name
        self._file_name = file_name
        self._max_queue = max_queue
        self._flush_secs = flush_secs
        self._filename_suffix = filename_suffix

        self._experiment: Optional[LogWriter] = None

    @property
    @override
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Gets the save directory where the VisualDL experiments are saved.

        Returns:
            The local path to the save directory where the VisualDL experiments are saved.

        """
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        """The directory for this run's visualdl checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, self.name, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def sub_dir(self) -> Optional[str]:
        """Gets the sub directory where the VisualDL experiments are saved.

        Returns:
            The local path to the sub directory where the VisualDL experiments are saved.

        """
        return self._sub_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> "LogWriter":
        """Actual visualdl object. To use VisualDL features anywhere in your code, do the following.

        Example::

            logger.experiment.add_scalar(tag="acc", step=1, value=0.5678)

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)

        from visualdl import LogWriter

        self._experiment = LogWriter(
            logdir=self.log_dir,
            max_queue=self._max_queue,
            flush_secs=self._flush_secs,
            filename_suffix=self._filename_suffix,
            file_name=self._file_name,
            display_name=self._display_name,
        )
        return self._experiment

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()

            if isinstance(v, dict):
                # VisualDL doesn't have direct add_scalars equivalent, so log each separately
                for sub_k, sub_v in v.items():
                    try:
                        self.experiment.add_scalar(tag=f"{k}/{sub_k}", step=step or 0, value=float(sub_v))
                    except Exception as ex:
                        raise ValueError(
                            f"\n you tried to log {sub_v} which is currently not supported. Try a scalar/tensor."
                        ) from ex
            else:
                try:
                    self.experiment.add_scalar(tag=k, step=step or 0, value=float(v))
                except Exception as ex:
                    raise ValueError(
                        f"\n you tried to log {v} which is currently not supported. Try a scalar/tensor."
                    ) from ex

    @override
    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        metrics: Optional[dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Record hyperparameters. VisualDL logs hyperparameters through the hyper_parameters component.

        Args:
            params: A dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Optional global step number for the logged metrics

        Note:
            VisualDL handles hyperparameters differently than TensorBoard. This implementation
            logs hyperparameters as text in a structured format for visualization in the
            hyper_parameters component.

        """
        params = _convert_params(params)

        # format params into a suitable structure
        params = _flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, step)

        # Log hyperparameters as a formatted text entry for the hyper_parameters component
        # This allows visualization in VisualDL's hyper parameters panel
        hp_text = "Hyperparameters:\n"
        for k, v in params.items():
            hp_text += f"{k}: {v}\n"

        if metrics:
            hp_text += "\nMetrics:\n"
            for k, v in metrics.items():
                hp_text += f"{k}: {v}\n"

        self.experiment.add_text(tag="hyperparameters", step=step or 0, text_string=hp_text)

    @override
    @rank_zero_only
    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        """Log the model graph. Note: VisualDL supports graph visualization through its Graph component.

        Args:
            model: The model to log
            input_array: Input tensor to trace the model graph

        Note:
            VisualDL supports both static and dynamic graph visualization. This method attempts
            to use the dynamic graph visualization if input_array is provided.

        """
        model_example_input = getattr(model, "example_input_array", None)
        input_array = model_example_input if input_array is None else input_array
        model = _unwrap_objects(model)

        if input_array is None:
            rank_zero_warn(
                "Could not log computational graph to VisualDL: The `model.example_input_array` attribute"
                " is not set or `input_array` was not given."
            )
            return

        # VisualDL doesn't have a direct add_graph method like TensorBoard
        # Instead, we can attempt to log the model structure as text or use a custom approach
        # For now, we'll log a message about graph logging limitations
        rank_zero_warn(
            "VisualDL graph logging requires manual export of the model. "
            "You can use the VisualDL Graph component separately by launching "
            "visualdl with the --model parameter pointing to your saved model file."
        )

    @override
    @rank_zero_only
    def save(self) -> None:
        """Flush the experiment to ensure all data is written to disk."""
        self.experiment.flush()

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Close the experiment."""
        if self._experiment is not None:
            self.experiment.flush()
            self.experiment.close()

    def _get_next_version(self) -> int:
        """Get the next available version number."""
        save_dir = os.path.join(self.root_dir, self.name)

        try:
            listdir_info = self._fs.listdir(save_dir)
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

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Sanitize parameters for logging."""
        params = _utils_sanitize_params(params)
        # Convert any multi-dimensional arrays to strings
        return {k: str(v) if hasattr(v, "ndim") and v.ndim > 1 else v for k, v in params.items()}

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling."""
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state

    @rank_zero_only
    def log_image(
        self,
        tag: str,
        image: Union[Tensor, np.ndarray, str],
        step: Optional[int] = None,
        dataformats: str = "HWC",
        walltime=None,
    ) -> None:
        """Log an image to VisualDL.

        Args:
            tag: Data identifier
            image: Image to log. Can be a tensor, numpy array, or file path
            step: Global step value
            dataformats: Format of image. Defaults to "HWC" (height, width, channels)
            walltime: Wall time of image. Defaults to None

        """

        # Convert input to numpy array
        if isinstance(image, str):
            # Handle file path using PIL (which is a VisualDL dependency)
            from PIL import Image as PILImage

            try:
                pil_img = PILImage.open(image)
                # Convert PIL image to numpy array
                img_array = np.array(pil_img)
            except Exception as e:
                raise ValueError(f"Failed to load image from path {image}: {e}")

        elif isinstance(image, Tensor):
            # Handle PyTorch tensor
            import torch

            if image.device != torch.device("cpu"):
                image = image.cpu()

            # Convert to numpy and handle different tensor shapes
            if image.dim() == 4:  # Batch of images
                # Take first image if batch provided
                image = image[0]

            # Assume (C, H, W) or (H, W, C)
            if (
                image.dim() == 3
                and image.size(0) in [1, 3]
                and image.size(0) < image.size(1)
                and image.size(0) < image.size(2)
            ):
                # Likely (C, H, W) format, need to transpose to HWC
                image = image.permute(1, 2, 0)

            if image.dtype == torch.float32 or image.dtype == torch.float64:
                # Assume float images are in [0,1] range, convert to uint8
                if image.max() <= 1.0:
                    img_array = (image.numpy() * 255).astype(np.uint8)
                else:
                    img_array = image.numpy().astype(np.uint8)
            else:
                img_array = image.numpy()

        else:
            # Assume it's already a numpy array
            img_array = image

        # Ensure the array is uint8
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)

        # Add image to VisualDL
        self.experiment.add_image(tag=tag, img=img_array, step=step or 0, dataformats=dataformats, walltime=walltime)

    @rank_zero_only
    def log_histogram(
        self,
        tag: str,
        values: Union[Tensor, np.ndarray],
        step: Optional[int] = None,
        walltime: Optional[int] = None,
        buckets: Optional[int] = None,
    ) -> None:
        """Log a histogram of values.

        Args:
            tag: Data identifier
            values: Values to create histogram from
            step: Global step value
            walltime: Wall time of histogram
            buckets: Number of buckets for the histogram

        """
        if isinstance(values, Tensor):
            values = values.cpu().numpy()

        self.experiment.add_histogram(tag=tag, step=step or 0, values=values, buckets=buckets, walltime=walltime)

    @rank_zero_only
    def log_text(self, tag: str, text: str, step: Optional[int] = None, walltime: Optional[int] = None) -> None:
        """Log text data.

        Args:
            tag: Data identifier
            text: Text content to log
            step: Global step value
            walltime: Wall time of text

        """
        self.experiment.add_text(tag=tag, step=step or 0, text_string=text, walltime=walltime)

    @rank_zero_only
    def log_audio(
        self,
        tag: str,
        audio: Union[Tensor, np.ndarray, str],
        step: Optional[int] = None,
        sample_rate: int = 16000,
        walltime: Optional[int] = None,
    ) -> None:
        """Log audio data.

        Args:
            tag: Data identifier
            audio: Audio to log. Can be a tensor, numpy array, or file path
            step: Global step value
            sample_rate: Sample rate of the audio
            walltime: Wall time of audio

        """

        # Convert input to numpy array
        if isinstance(audio, str):
            # Handle file path using soundfile
            try:
                import soundfile as sf

                audio_array, file_sr = sf.read(audio)

                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)

                # Resample if needed (optional, can be added later)
                if file_sr != sample_rate:
                    rank_zero_warn(
                        f"Audio file sample rate {file_sr} doesn't match requested sample rate {sample_rate}. "
                        f"Using original sample rate {file_sr}."
                    )
                    sample_rate = file_sr

            except ImportError:
                raise ModuleNotFoundError(
                    "Reading audio from file path requires `soundfile`. "
                    "Try `pip install soundfile` to install it.\n"
                    "Alternatively, pass audio as numpy array instead of file path."
                )
            except Exception as e:
                raise ValueError(f"Failed to load audio from path {audio}: {e}")

        elif isinstance(audio, Tensor):
            audio_array = audio.cpu().numpy()
        else:
            audio_array = audio

        # Ensure audio is float32 and in range [-1, 1]
        if audio_array.dtype != np.float32:
            if audio_array.dtype in [np.int16, np.int32]:
                audio_array = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
            else:
                audio_array = audio_array.astype(np.float32)

        # VisualDL's add_audio expects numpy array
        self.experiment.add_audio(
            tag=tag, audio_array=audio_array, step=step or 0, sample_rate=sample_rate, walltime=walltime
        )

    @rank_zero_only
    def log_embeddings(
        self,
        tag: str,
        mat: Optional[Union[Tensor, np.ndarray, list]] = None,
        metadata: Optional[Union[Tensor, np.ndarray, list]] = None,
        metadata_header: Optional[Union[Tensor, np.ndarray, list]] = None,
        walltime: Optional[int] = None,
    ) -> None:
        """Log high-dimensional embeddings for visualization.

        Args:
            tag: Data identifier
            mat: A matrix where each row represents a feature vector (numpy array, tensor, or list)
            metadata: Labels for each point in the embedding (1D or 2D matrix)
            metadata_header: Meta data headers for the metadata (required if metadata is 2D)
            walltime: Wall time of embeddings

        """
        import numpy as np

        # Convert inputs to lists (VisualDL expects lists)
        if mat is not None:
            if isinstance(mat, Tensor):
                mat = mat.cpu().numpy().tolist()
            elif isinstance(mat, np.ndarray):
                mat = mat.tolist()

        if metadata is not None:
            if isinstance(metadata, Tensor):
                metadata = metadata.cpu().numpy().tolist()
            elif isinstance(metadata, np.ndarray):
                metadata = metadata.tolist()

        if metadata_header is not None:
            if isinstance(metadata_header, Tensor):
                metadata_header = metadata_header.cpu().numpy().tolist()
            elif isinstance(metadata_header, np.ndarray):
                metadata_header = metadata_header.tolist()

        # Auto-generate metadata_header if metadata is 2D and header not provided
        if metadata is not None and metadata_header is None and metadata and isinstance(metadata[0], list):
            metadata_header = [f"label_{i}" for i in range(len(metadata[0]))]

        self.experiment.add_embeddings(
            tag=tag, mat=mat, metadata=metadata, metadata_header=metadata_header, walltime=walltime
        )

    @rank_zero_only
    def log_pr_curve(
        self,
        tag: str,
        labels: Union[Tensor, np.ndarray],
        predictions: Union[Tensor, np.ndarray],
        step: Optional[int] = None,
        num_thresholds: int = 100,
        weights: Optional[float] = None,
        walltime: Optional[int] = None,
    ) -> None:
        """Log PR curve data.

        Args:
            tag: Data identifier
            labels: Ground truth labels
            predictions: Predicted probabilities
            step: Global step value
            num_thresholds: Number of thresholds to use

        """
        if isinstance(labels, Tensor):
            labels = labels.cpu().numpy()
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()

        self.experiment.add_pr_curve(
            tag=tag,
            labels=labels,
            predictions=predictions,
            step=step or 0,
            num_thresholds=num_thresholds,
            weights=weights,
            walltime=walltime,
        )

    @rank_zero_only
    def log_roc_curve(
        self,
        tag: str,
        labels: Union[Tensor, np.ndarray],
        predictions: Union[Tensor, np.ndarray],
        step: Optional[int] = None,
        num_thresholds: int = 100,
        weights: Optional[float] = None,
        walltime: Optional[int] = None,
    ) -> None:
        """Log ROC curve data.

        Args:
            tag: Data identifier
            labels: Ground truth labels
            predictions: Predicted probabilities
            step: Global step value
            num_thresholds: Number of thresholds to use

        """
        if isinstance(labels, Tensor):
            labels = labels.cpu().numpy()
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()

        self.experiment.add_roc_curve(
            tag=tag,
            labels=labels,
            predictions=predictions,
            step=step or 0,
            num_thresholds=num_thresholds,
            weights=weights,
            walltime=walltime,
        )
