import os
from warnings import warn
from argparse import Namespace
from pkg_resources import parse_version

import torch
import csv
from torch.utils.tensorboard import SummaryWriter

from .base import LightningLoggerBase, rank_zero_only


class TensorBoardLogger(LightningLoggerBase):
    r"""

    Log to local file system in TensorBoard format

    Implemented using :class:`torch.utils.tensorboard.SummaryWriter`. Logs are saved to
    `os.path.join(save_dir, name, version)`

    .. _tf-logger:

    Example
    ------------------

    .. code-block:: python

        logger = TensorBoardLogger("tb_logs", name="my_model")
        trainer = Trainer(logger=logger)
        trainer.train(model)

    Args:
        save_dir (str): Save directory
        name (str): Experiment name. Defaults to "default".  If it is the empty string then no per-experiment
            subdirectory is used.
        version (int|str): Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise version_${version} is used.
        \**kwargs  (dict): Other arguments are passed directly to the :class:`SummaryWriter` constructor.

    """
    NAME_CSV_TAGS = 'meta_tags.csv'

    def __init__(self, save_dir, name="default", version=None, **kwargs):
        super().__init__()
        self.save_dir = save_dir
        self._name = name
        self._version = version

        self._experiment = None
        self.tags = {}
        self.kwargs = kwargs

    @property
    def root_dir(self):
        """
        Parent directory for all tensorboard checkpoint subdirectories.
        If the experiment name parameter is None or the empty string, no experiment subdirectory is used
        and checkpoint will be saved in save_dir/version_dir
        """
        if self.name is None or len(self.name) == 0:
            return self.save_dir
        else:
            return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self):
        """
        The directory for this run's tensorboard checkpoint.  By default, it is named 'version_${self.version}'
        but it can be overridden by passing a string value for the constructor's version parameter
        instead of None or an int
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def experiment(self):
        r"""

         Actual tensorboard object. To use tensorboard features do the following.

         Example::

             self.logger.experiment.some_tensorboard_function()

         """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = SummaryWriter(log_dir=self.log_dir, **self.kwargs)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        if params is None:
            return

        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)
        params = dict(params)

        if parse_version(torch.__version__) < parse_version("1.3.0"):
            warn(
                f"Hyperparameter logging is not available for Torch version {torch.__version__}."
                " Skipping log_hyperparams. Upgrade to Torch 1.3.0 or above to enable"
                " hyperparameter logging."
            )
        else:
            from torch.utils.tensorboard.summary import hparams
            exp, ssi, sei = hparams(params, {})
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)
        # some alternative should be added
        self.tags.update(params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(k, v, step)

    @rank_zero_only
    def save(self):
        try:
            self.experiment.flush()
        except AttributeError:
            # you are using PT version (<v1.2) which does not have implemented flush
            self.experiment._get_file_writer().flush()

        dir_path = self.log_dir
        if not os.path.isdir(dir_path):
            dir_path = self.save_dir

        # prepare the file path
        meta_tags_path = os.path.join(dir_path, self.NAME_CSV_TAGS)

        # save the metatags file
        with open(meta_tags_path, 'w', newline='') as csvfile:
            fieldnames = ['key', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'key': 'key', 'value': 'value'})
            for k, v in self.tags.items():
                writer.writerow({'key': k, 'value': v})

    @rank_zero_only
    def finalize(self, status):
        self.save()

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self.save_dir, self.name)
        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
