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
Comet Logger
------------
"""

import logging
import os
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from comet_ml import ExistingExperiment, Experiment, OfflineExperiment

log = logging.getLogger(__name__)
_COMET_AVAILABLE = RequirementCache("comet-ml>=3.31.0", module="comet_ml")


class CometLogger(Logger):
    r"""Track your parameters, metrics, source code and more using `Comet
    <https://www.comet.com/?utm_source=lightning.pytorch&utm_medium=referral>`_.

    Install it with pip:

    .. code-block:: bash

        pip install comet-ml

    Comet requires either an API Key (online mode) or a local directory path (offline mode).

    **ONLINE MODE**

    .. code-block:: python

        import os
        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import CometLogger

        # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
            save_dir=".",  # Optional
            project_name="default_project",  # Optional
            rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
            experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),  # Optional
            experiment_name="lightning_logs",  # Optional
        )
        trainer = Trainer(logger=comet_logger)

    **OFFLINE MODE**

    .. code-block:: python

        from lightning.pytorch.loggers import CometLogger

        # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        comet_logger = CometLogger(
            save_dir=".",
            workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
            project_name="default_project",  # Optional
            rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
            experiment_name="lightning_logs",  # Optional
        )
        trainer = Trainer(logger=comet_logger)

    **Log Hyperparameters:**

    Log parameters used to initialize a :class:`~lightning.pytorch.core.LightningModule`:

    .. code-block:: python

        class LitModule(LightningModule):
            def __init__(self, *args, **kwarg):
                self.save_hyperparameters()

    Log other Experiment Parameters

    .. code-block:: python

        # log a single parameter
        logger.log_hyperparams({"batch_size": 16})

        # log multiple parameters
        logger.log_hyperparams({"batch_size": 16, "learning_rate": 0.001})

    **Log Metrics:**

    .. code-block:: python

        # log a single metric
        logger.log_metrics({"train/loss": 0.001})

        # add multiple metrics
        logger.log_metrics({"train/loss": 0.001, "val/loss": 0.002})

    **Access the Comet Experiment object:**

    You can gain access to the underlying Comet
    `Experiment <https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/>`__ object
    and its methods through the :obj:`logger.experiment` property. This will let you use
    the additional logging features provided by the Comet SDK.

    Some examples of data you can log through the Experiment object:

    Log Image data:

    .. code-block:: python

        img = PIL.Image.open("<path to image>")
        logger.experiment.log_image(img, file_name="my_image.png")

    Log Text data:

    .. code-block:: python

        text = "Lightning is awesome!"
        logger.experiment.log_text(text)

    Log Audio data:

    .. code-block:: python

        audio = "<path to audio data>"
        logger.experiment.log_audio(audio, file_name="my_audio.wav")

    Log arbitrary data assets:

    You can log any type of data to Comet as an asset. These can be model
    checkpoints, datasets, debug logs, etc.

    .. code-block:: python

        logger.experiment.log_asset("<path to your asset>", file_name="my_data.pkl")

    Log Models to Comet's Model Registry:

    .. code-block:: python

        logger.experiment.log_model(name="my-model", "<path to your model>")

    See Also:
        - `Demo in Google Colab <https://tinyurl.com/22phzw5s>`__
        - `Comet Documentation <https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch-lightning/>`__

    Args:
        api_key: Required in online mode. API key, found on Comet.ml. If not given, this
            will be loaded from the environment variable COMET_API_KEY or ~/.comet.config
            if either exists.
        save_dir: Required in offline mode. The path for the directory to save local
            comet logs. If given, this also sets the directory for saving checkpoints.
        project_name: Optional. Send your experiment to a specific project.
            Otherwise will be sent to Uncategorized Experiments.
            If the project name does not already exist, Comet.ml will create a new project.
        rest_api_key: Optional. Rest API key found in Comet.ml settings.
            This is used to determine version number
        experiment_name: Optional. String representing the name for this particular experiment on Comet.ml.
        experiment_key: Optional. If set, restores from existing experiment.
        offline: If api_key and save_dir are both given, this determines whether
            the experiment will be in online or offline mode. This is useful if you use
            save_dir to control the checkpoints directory and have a ~/.comet.config
            file but still want to run offline experiments.
        prefix: A string to put at the beginning of metric keys.
        \**kwargs: Additional arguments like `workspace`, `log_code`, etc. used by
            :class:`CometExperiment` can be passed as keyword arguments in this logger.

    Raises:
        ModuleNotFoundError:
            If required Comet package is not installed on the device.
        MisconfigurationException:
            If neither ``api_key`` nor ``save_dir`` are passed as arguments.

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        api_key: Optional[str] = None,
        save_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        rest_api_key: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_key: Optional[str] = None,
        offline: bool = False,
        prefix: str = "",
        **kwargs: Any,
    ):
        if not _COMET_AVAILABLE:
            raise ModuleNotFoundError(str(_COMET_AVAILABLE))
        super().__init__()
        self._experiment = None
        self._save_dir: Optional[str]
        self.rest_api_key: Optional[str]

        # needs to be set before the first `comet_ml` import
        os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

        import comet_ml

        # Determine online or offline mode based on which arguments were passed to CometLogger
        api_key = api_key or comet_ml.config.get_api_key(None, comet_ml.config.get_config())

        if api_key is not None and save_dir is not None:
            self.mode = "offline" if offline else "online"
            self.api_key = api_key
            self._save_dir = save_dir
        elif api_key is not None:
            self.mode = "online"
            self.api_key = api_key
            self._save_dir = None
        elif save_dir is not None:
            self.mode = "offline"
            self._save_dir = save_dir
        else:
            # If neither api_key nor save_dir are passed as arguments, raise an exception
            raise MisconfigurationException("CometLogger requires either api_key or save_dir during initialization.")

        log.info(f"CometLogger will be initialized in {self.mode} mode")

        self._project_name: Optional[str] = project_name
        self._experiment_key: Optional[str] = experiment_key
        self._experiment_name: Optional[str] = experiment_name
        self._prefix: str = prefix
        self._kwargs: Any = kwargs
        self._future_experiment_key: Optional[str] = None

        if rest_api_key is not None:
            from comet_ml.api import API

            # Comet.ml rest API, used to determine version number
            self.rest_api_key = rest_api_key
            self.comet_api = API(self.rest_api_key)
        else:
            self.rest_api_key = None
            self.comet_api = None

    @property
    @rank_zero_experiment
    def experiment(self) -> Union["Experiment", "ExistingExperiment", "OfflineExperiment"]:
        r"""Actual Comet object. To use Comet features in your :class:`~lightning.pytorch.core.LightningModule` do the
        following.

        Example::

            self.logger.experiment.some_comet_function()

        """
        if self._experiment is not None and self._experiment.alive:
            return self._experiment

        if self._future_experiment_key is not None:
            os.environ["COMET_EXPERIMENT_KEY"] = self._future_experiment_key

        from comet_ml import ExistingExperiment, Experiment, OfflineExperiment

        try:
            if self.mode == "online":
                if self._experiment_key is None:
                    self._experiment = Experiment(api_key=self.api_key, project_name=self._project_name, **self._kwargs)
                    self._experiment_key = self._experiment.get_key()
                else:
                    self._experiment = ExistingExperiment(
                        api_key=self.api_key,
                        project_name=self._project_name,
                        previous_experiment=self._experiment_key,
                        **self._kwargs,
                    )
            else:
                self._experiment = OfflineExperiment(
                    offline_directory=self.save_dir, project_name=self._project_name, **self._kwargs
                )
            self._experiment.log_other("Created from", "pytorch-lightning")
        finally:
            if self._future_experiment_key is not None:
                os.environ.pop("COMET_EXPERIMENT_KEY")
                self._future_experiment_key = None

        if self._experiment_name:
            self._experiment.set_name(self._experiment_name)

        return self._experiment

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        self.experiment.log_parameters(params)

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        # Comet.ml expects metrics to be a dictionary of detached tensors on CPU
        metrics_without_epoch = metrics.copy()
        for key, val in metrics_without_epoch.items():
            if isinstance(val, Tensor):
                metrics_without_epoch[key] = val.cpu().detach()

        epoch = metrics_without_epoch.pop("epoch", None)
        metrics_without_epoch = _add_prefix(metrics_without_epoch, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics_without_epoch, step=step, epoch=epoch)

    def reset_experiment(self) -> None:
        self._experiment = None

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        r"""When calling ``self.experiment.end()``, that experiment won't log any more data to Comet. That's why, if you
        need to log any more data, you need to create an ExistingCometExperiment. For example, to log data when testing
        your model after training, because when training is finalized :meth:`CometLogger.finalize` is called.

        This happens automatically in the :meth:`~CometLogger.experiment` property, when
        ``self._experiment`` is set to ``None``, i.e. ``self.reset_experiment()``.

        """
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.experiment.end()
        self.reset_experiment()

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        return self._save_dir

    @property
    @override
    def name(self) -> str:
        """Gets the project name.

        Returns:
            The project name if it is specified, else "comet-default".

        """
        # Don't create an experiment if we don't have one
        if self._experiment is not None and self._experiment.project_name is not None:
            return self._experiment.project_name

        if self._project_name is not None:
            return self._project_name

        return "comet-default"

    @property
    @override
    def version(self) -> str:
        """Gets the version.

        Returns:
            The first one of the following that is set in the following order

            1. experiment id.
            2. experiment key.
            3. "COMET_EXPERIMENT_KEY" environment variable.
            4. future experiment key.

            If none are present generates a new guid.

        """
        # Don't create an experiment if we don't have one
        if self._experiment is not None:
            return self._experiment.id

        if self._experiment_key is not None:
            return self._experiment_key

        if "COMET_EXPERIMENT_KEY" in os.environ:
            return os.environ["COMET_EXPERIMENT_KEY"]

        if self._future_experiment_key is not None:
            return self._future_experiment_key

        import comet_ml

        # Pre-generate an experiment key
        self._future_experiment_key = comet_ml.generate_guid()

        return self._future_experiment_key

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        # Save the experiment id in case an experiment object already exists,
        # this way we could create an ExistingExperiment pointing to the same
        # experiment
        state["_experiment_key"] = self._experiment.id if self._experiment is not None else None

        # Remove the experiment object as it contains hard to pickle objects
        # (like network connections), the experiment object will be recreated if
        # needed later
        state["_experiment"] = None
        return state

    @override
    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        if self._experiment is not None:
            self._experiment.set_model_graph(model)
