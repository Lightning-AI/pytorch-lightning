r"""

.. _comet:

CometLogger
-------------
"""

from logging import getLogger

try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml import ExistingExperiment as CometExistingExperiment
    from comet_ml import OfflineExperiment as CometOfflineExperiment
    try:
        from comet_ml.api import API
    except ImportError:
        # For more information, see: https://www.comet.ml/docs/python-sdk/releases/#release-300
        from comet_ml.papi import API
except ImportError:
    raise ImportError('Missing comet_ml package.')

from torch import is_tensor

from .base import LightningLoggerBase, rank_zero_only
from ..utilities.debugging import MisconfigurationException

logger = getLogger(__name__)


class CometLogger(LightningLoggerBase):
    r"""
    Log using `comet.ml <https://www.comet.ml>`_.
    """

    def __init__(self, api_key=None, save_dir=None, workspace=None,
                 rest_api_key=None, project_name=None, experiment_name=None,
                 experiment_key=None, **kwargs):
        r"""

        Requires either an API Key (online mode) or a local directory path (offline mode)

        .. code-block:: python

            # ONLINE MODE
            from pytorch_lightning.loggers import CometLogger
            # arguments made to CometLogger are passed on to the comet_ml.Experiment class
            comet_logger = CometLogger(
                api_key=os.environ["COMET_API_KEY"],
                workspace=os.environ["COMET_WORKSPACE"], # Optional
                project_name="default_project", # Optional
                rest_api_key=os.environ["COMET_REST_API_KEY"], # Optional
                experiment_name="default" # Optional
            )
            trainer = Trainer(logger=comet_logger)

        .. code-block:: python

            # OFFLINE MODE
            from pytorch_lightning.loggers import CometLogger
            # arguments made to CometLogger are passed on to the comet_ml.Experiment class
            comet_logger = CometLogger(
                save_dir=".",
                workspace=os.environ["COMET_WORKSPACE"], # Optional
                project_name="default_project", # Optional
                rest_api_key=os.environ["COMET_REST_API_KEY"], # Optional
                experiment_name="default" # Optional
            )
            trainer = Trainer(logger=comet_logger)

        Args:
            api_key (str): Required in online mode. API key, found on Comet.ml
            save_dir (str): Required in offline mode. The path for the directory to save local comet logs
            workspace (str): Optional. Name of workspace for this user
            project_name (str): Optional. Send your experiment to a specific project.
            Otherwise will be sent to Uncategorized Experiments.
            If project name does not already exists Comet.ml will create a new project.
            rest_api_key (str): Optional. Rest API key found in Comet.ml settings.
                This is used to determine version number
            experiment_name (str): Optional. String representing the name for this particular experiment on Comet.ml

        """
        super().__init__()
        self._experiment = None

        # Determine online or offline mode based on which arguments were passed to CometLogger
        if save_dir is not None and api_key is not None:
            # If arguments are passed for both save_dir and api_key, preference is given to online mode
            self.mode = "online"
            self.api_key = api_key
        elif api_key is not None:
            self.mode = "online"
            self.api_key = api_key
        elif save_dir is not None:
            self.mode = "offline"
            self.save_dir = save_dir
        else:
            # If neither api_key nor save_dir are passed as arguments, raise an exception
            raise MisconfigurationException("CometLogger requires either api_key or save_dir during initialization.")

        logger.info(f"CometLogger will be initialized in {self.mode} mode")

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
            try:
                self.name = experiment_name
            except TypeError as e:
                logger.exception("Failed to set experiment name for comet.ml logger")

    @property
    def experiment(self):
        r"""

        Actual comet object. To use comet features do the following.

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
    def log_hyperparams(self, params):
        self.experiment.log_parameters(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # Comet.ml expects metrics to be a dictionary of detached tensors on CPU
        for key, val in metrics.items():
            if is_tensor(val):
                metrics[key] = val.cpu().detach()

        self.experiment.log_metrics(metrics, step=step)

    def reset_experiment(self):
        self._experiment = None

    @rank_zero_only
    def finalize(self, status):
        r"""
        When calling self.experiment.end(), that experiment won't log any more data to Comet. That's why, if you need
        to log any more data you need to create an ExistingCometExperiment. For example, to log data when testing your
        model after training, because when training is finalized CometLogger.finalize is called.

        This happens automatically in the CometLogger.experiment property, when self._experiment is set to None
        i.e. self.reset_experiment().
        """
        self.experiment.end()
        self.reset_experiment()

    @property
    def name(self):
        return self.experiment.project_name

    @name.setter
    def name(self, value):
        self.experiment.set_name(value)

    @property
    def version(self):
        return self.experiment.id
