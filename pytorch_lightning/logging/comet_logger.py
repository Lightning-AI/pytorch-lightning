from torch import is_tensor

try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml.papi import API
except ImportError:
    raise ImportError('Missing comet_ml package.')

from .base import LightningLoggerBase, rank_zero_only

class CometLogger(LightningLoggerBase):
    def __init__(self, api_key, workspace, rest_api_key=None, project_name=None, experiment_name=None, *args, **kwargs):
        """
        Initialize a Comet.ml logger

        :param api_key: API key, found on Comet.ml
        :param workspace: Name of workspace for this user
        :param project_name: Optional. Send your experiment to a specific project.
        Otherwise will be sent to Uncategorized Experiments.
        If project name does not already exists Comet.ml will create a new project.
        :param rest_api_key: Optional. Rest API key found in Comet.ml settings. This is used to determine version number
        :param experiment_name: Optional. String representing the name for this particular experiment on Comet.ml
        """
        super(CometLogger, self).__init__()
        self.experiment = CometExperiment(api_key=api_key, workspace=workspace, project_name=project_name, *args, **kwargs)

        self.workspace = workspace
        self.project_name = project_name

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
                print("Failed to set experiment name for comet.ml logger")

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.log_parameters(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
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

