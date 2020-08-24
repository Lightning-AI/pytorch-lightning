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
Comet
-----
"""

from argparse import Namespace
from typing import Optional, Dict, Union, Any

try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml import ExistingExperiment as CometExistingExperiment
    from comet_ml import OfflineExperiment as CometOfflineExperiment
    from comet_ml import BaseExperiment as CometBaseExperiment
    try:
        from comet_ml.api import API
    except ImportError:  # pragma: no-cover
        # For more information, see: https://www.comet.ml/docs/python-sdk/releases/#release-300
        from comet_ml.papi import API  # pragma: no-cover
    from comet_ml.config import get_config, get_api_key
except ImportError:  # pragma: no-cover
    CometExperiment = None
    CometExistingExperiment = None
    CometOfflineExperiment = None
    CometBaseExperiment = None
    API = None
    _COMET_AVAILABLE = False
else:
    _COMET_AVAILABLE = True


import torch
from torch import is_tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_only


class CometLogger(LightningLoggerBase):
    r"""
    Log using `Comet.ml <https://www.comet.ml>`_. Install it with pip:

    .. code-block:: bash

        pip install comet-ml

    Comet requires either an API Key (online mode) or a local directory path (offline mode).

    **ONLINE MODE**

    Example:
        >>> import os
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CometLogger
        >>> # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        >>> comet_logger = CometLogger(
        ...     api_key=os.environ.get('COMET_API_KEY'),
        ...     workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        ...     save_dir='.',  # Optional
        ...     project_name='default_project',  # Optional
        ...     rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        ...     experiment_name='default'  # Optional
        ... )
        >>> trainer = Trainer(logger=comet_logger)

    **OFFLINE MODE**

    Example:
        >>> from pytorch_lightning.loggers import CometLogger
        >>> # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        >>> comet_logger = CometLogger(
        ...     save_dir='.',
        ...     workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        ...     project_name='default_project',  # Optional
        ...     rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        ...     experiment_name='default'  # Optional
        ... )
        >>> trainer = Trainer(logger=comet_logger)

    Args:
        api_key: Required in online mode. API key, found on Comet.ml. If not given, this
            will be loaded from the environment variable COMET_API_KEY or ~/.comet.config
            if either exists.
        save_dir: Required in offline mode. The path for the directory to save local
            comet logs. If given, this also sets the directory for saving checkpoints.
        workspace: Optional. Name of workspace for this user
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
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 workspace: Optional[str] = None,
                 project_name: Optional[str] = None,
                 rest_api_key: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 experiment_key: Optional[str] = None,
                 offline: bool = False,
                 **kwargs):

        if not _COMET_AVAILABLE:
            raise ImportError('You want to use `comet_ml` logger which is not installed yet,'
                              ' install it with `pip install comet-ml`.')
        super().__init__()
        self._experiment = None

        # Determine online or offline mode based on which arguments were passed to CometLogger
        api_key = api_key or get_api_key(None, get_config())

        if api_key is not None and save_dir is not None:
            self.mode = "offline" if offline else "online"
            self.api_key = api_key
            self._save_dir = save_dir
        elif api_key is not None:
            self.mode = "online"
            self.api_key = api_key
        elif save_dir is not None:
            self.mode = "offline"
            self._save_dir = save_dir
        else:
            # If neither api_key nor save_dir are passed as arguments, raise an exception
            raise MisconfigurationException(
                "CometLogger requires either api_key or save_dir during initialization."
            )

        log.info(f"CometLogger will be initialized in {self.mode} mode")

        self.workspace = workspace
        self.project_name = project_name
        self.experiment_key = experiment_key
        self._kwargs = kwargs

        if rest_api_key is not None:
            # Comet.ml rest API, used to determine version number
            self.rest_api_key = rest_api_key
            self.comet_api = API(self.rest_api_key)
        else:
            self.rest_api_key = None
            self.comet_api = None

        if experiment_name:
            self.experiment.set_name(experiment_name)
        self._kwargs = kwargs

    @property
    @rank_zero_experiment
    def experiment(self) -> CometBaseExperiment:
        r"""
        Actual Comet object. To use Comet features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_comet_function()

        """
        if self._experiment is not None:
            return self._experiment

        if self.mode == "online":
            if self.experiment_key is None:
                self._experiment = CometExperiment(
                    api_key=self.api_key,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    **self._kwargs
                )
                self.experiment_key = self._experiment.get_key()
            else:
                self._experiment = CometExistingExperiment(
                    api_key=self.api_key,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    previous_experiment=self.experiment_key,
                    **self._kwargs
                )
        else:
            self._experiment = CometOfflineExperiment(
                offline_directory=self.save_dir,
                workspace=self.workspace,
                project_name=self.project_name,
                **self._kwargs
            )

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        self.experiment.log_parameters(params)

    @rank_zero_only
    def log_metrics(
            self,
            metrics: Dict[str, Union[torch.Tensor, float]],
            step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        # Comet.ml expects metrics to be a dictionary of detached tensors on CPU
        for key, val in metrics.items():
            if is_tensor(val):
                metrics[key] = val.cpu().detach()

        self.experiment.log_metrics(metrics, step=step)

    def reset_experiment(self):
        self._experiment = None

    @rank_zero_only
    def finalize(self, status: str) -> None:
        r"""
        When calling ``self.experiment.end()``, that experiment won't log any more data to Comet.
        That's why, if you need to log any more data, you need to create an ExistingCometExperiment.
        For example, to log data when testing your model after training, because when training is
        finalized :meth:`CometLogger.finalize` is called.

        This happens automatically in the :meth:`~CometLogger.experiment` property, when
        ``self._experiment`` is set to ``None``, i.e. ``self.reset_experiment()``.
        """
        self.experiment.end()
        self.reset_experiment()

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> str:
        return str(self.experiment.project_name)

    @property
    def version(self) -> str:
        return self.experiment.id

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
