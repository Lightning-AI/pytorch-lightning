r"""

.. _wandb:

WandbLogger
-------------
"""
import os
from argparse import Namespace
from typing import Optional, List, Dict, Union, Any

import torch.nn as nn

try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `wandb` logger which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install wandb`.')

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only


class WandbLogger(LightningLoggerBase):
    """
    Logger for `W&B <https://www.wandb.com/>`_.

    Args:
        name (str): display name for the run.
        save_dir (str): path where data is saved.
        offline (bool): run offline (data can be streamed later to wandb servers).
        id or version (str): sets the version, mainly used to resume a previous run.
        anonymous (bool): enables or explicitly disables anonymous logging.
        project (str): the name of the project to which this run will belong.
        tags (list of str): tags associated with this run.
        log_model (bool): save checkpoints in wandb dir to upload on W&B servers.

    Example
    --------
    .. code-block:: python

        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer

        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)
    """

    def __init__(self,
                 name: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 offline: bool = False,
                 id: Optional[str] = None,
                 anonymous: bool = False,
                 version: Optional[str] = None,
                 project: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 log_model: bool = False,
                 experiment=None,
                 entity=None):
        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._anonymous = 'allow' if anonymous else None
        self._id = version or id
        self._tags = tags
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._entity = entity
        self._log_model = log_model

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state['_id'] = self._experiment.id if self._experiment is not None else None

        # cannot be pickled
        state['_experiment'] = None
        return state

    @property
    def experiment(self) -> Run:
        r"""

          Actual wandb object. To use wandb features do the following.

          Example::

              self.logger.experiment.some_wandb_function()

          """
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            self._experiment = wandb.init(
                name=self._name, dir=self._save_dir, project=self._project, anonymous=self._anonymous,
                reinit=True, id=self._id, resume='allow', tags=self._tags, entity=self._entity)
            # save checkpoints in wandb dir to upload on W&B servers
            if self._log_model:
                self.save_dir = self._experiment.dir
        return self._experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        self.experiment.config.update(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            metrics['global_step'] = step
        self.experiment.log(metrics)

    @property
    def name(self) -> str:
        # don't create an experiment if we don't have one
        name = self._experiment.project_name() if self._experiment else None
        return name

    @property
    def version(self) -> str:
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else None
