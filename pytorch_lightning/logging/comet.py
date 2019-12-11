"""
Log using `comet <https://www.comet.ml>`_

Comet logger can be used in either online or offline mode.
To log in online mode, CometLogger requries an API key:

.. code-block:: python

    from pytorch_lightning.logging import CometLogger
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        api_key=os.environ["COMET_KEY"],
        workspace=os.environ["COMET_WORKSPACE"], # Optional
        project_name="default_project", # Optional
        rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name="default" # Optional
    )
    trainer = Trainer(logger=comet_logger)

To log in offline mode, CometLogger requires a path to a local directory:

.. code-block:: python

    from pytorch_lightning.logging import CometLogger
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        save_dir=".",
        workspace=os.environ["COMET_WORKSPACE"], # Optional
        project_name="default_project", # Optional
        rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name="default" # Optional
    )
    trainer = Trainer(logger=comet_logger)


Use the logger anywhere in you LightningModule as follows:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.whatever_comet_ml_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.whatever_comet_ml_supports(...)


"""

from logging import getLogger

try:
    from comet_ml import Experiment as CometExperiment
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
    def __init__(self, api_key=None, save_dir=None, workspace=None,
                 rest_api_key=None, project_name=None, experiment_name=None, **kwargs):
        """Initialize a Comet.ml logger.
        Requires either an API Key (online mode) or a local directory path (offline mode)

        :param str api_key: Required in online mode. API key, found on Comet.ml
        :param str save_dir: Required in offline mode. The path for the directory to save local comet logs
        :param str workspace: Optional. Name of workspace for this user
        :param str project_name: Optional. Send your experiment to a specific project.
        Otherwise will be sent to Uncategorized Experiments.
        If project name does not already exists Comet.ml will create a new project.
        :param str rest_api_key: Optional. Rest API key found in Comet.ml settings.
        This is used to determine version number
        :param str experiment_name: Optional. String representing the name for this particular experiment on Comet.ml
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
        if self._experiment is not None:
            return self._experiment

        if self.mode == "online":
            self._experiment = CometExperiment(
                api_key=self.api_key,
                workspace=self.workspace,
                project_name=self.project_name,
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

    @rank_zero_only
    def finalize(self, status):
        self.experiment.end()

    @property
    def name(self):
        return self.experiment.project_name

    @name.setter
    def name(self, value):
        self.experiment.set_name(value)

    @property
    def version(self):
        return self.experiment.id
