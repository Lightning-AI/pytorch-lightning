from logging import getLogger

try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml.papi import API
except ImportError:
    raise ImportError('Missing comet_ml package.')

from torch import is_tensor

from .base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class CometLogger(LightningLoggerBase):
    def __init__(self, api_key, workspace, rest_api_key=None, project_name=None, experiment_name=None, **kwargs):
        """
        Initialize a Comet.ml logger

        :param str api_key: API key, found on Comet.ml
        :param str workspace: Name of workspace for this user
        :param str project_name: Optional. Send your experiment to a specific project.
        Otherwise will be sent to Uncategorized Experiments.
        If project name does not already exists Comet.ml will create a new project.
        :param str rest_api_key: Optional. Rest API key found in Comet.ml settings.
        This is used to determine version number
        :param str experiment_name: Optional. String representing the name for this particular experiment on Comet.ml
        """
        super().__init__()
        self._experiment = None

        self.api_key = api_key
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
                self._set_experiment_name(experiment_name)
            except TypeError as e:
                logger.exception("Failed to set experiment name for comet.ml logger")

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment

        self._experiment = CometExperiment(
            api_key=self.api_key,
            workspace=self.workspace,
            project_name=self.project_name,
            **self._kwargs
        )

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.log_parameters(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step_num=None):
        # Comet.ml expects metrics to be a dictionary of detached tensors on CPU
        for key, val in metrics.items():
            if is_tensor(val):
                metrics[key] = val.cpu().detach()

        self.experiment.log_metrics(metrics, step=step_num)

    @rank_zero_only
    def finalize(self, status):
        self.experiment.end()

    @rank_zero_only
    def _set_experiment_name(self, experiment_name):
        self.experiment.set_name(experiment_name)

    @property
    def name(self):
        return self.experiment.project_name

    @property
    def version(self):
        if self.project_name and self.rest_api_key:
            # Determines the number of experiments in this project, and returns the next integer as the version number
            nb_exps = len(self.comet_api.get_experiments(self.workspace, self.project_name))
            return nb_exps + 1
        else:
            return None
