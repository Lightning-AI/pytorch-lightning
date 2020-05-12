"""
Weights and Biases
------------------
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

from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


class WandbLogger(LightningLoggerBase):
    """
    Log using `Weights and Biases <https://www.wandb.com/>`_. Install it with pip:

    .. code-block:: bash

        pip install wandb

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved.
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        anonymous: Enables or explicitly disables anonymous logging.
        version: Sets the version, mainly used to resume a previous run.
        project: The name of the project to which this run will belong.
        tags: Tags associated with this run.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        experiment: WandB experiment object
        entity: The team posting this run (default: your username or your default team)
        group: A unique string shared by all runs in a given group

    Example:
        >>> from pytorch_lightning.loggers import WandbLogger
        >>> from pytorch_lightning import Trainer
        >>> wandb_logger = WandbLogger()
        >>> trainer = Trainer(logger=wandb_logger)

    See Also:
        - `Tutorial <https://app.wandb.ai/cayush/pytorchlightning/reports/
          Use-Pytorch-Lightning-with-Weights-%26-Biases--Vmlldzo2NjQ1Mw>`__
          on how to use W&B with Pytorch Lightning.

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
                 entity=None,
                 group: Optional[str] = None):
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
        self._group = group

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

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_wandb_function()

        """
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            self._experiment = wandb.init(
                name=self._name, dir=self._save_dir, project=self._project, anonymous=self._anonymous,
                reinit=True, id=self._id, resume='allow', tags=self._tags, entity=self._entity,
                group=self._group)
            # save checkpoints in wandb dir to upload on W&B servers
            if self._log_model:
                self.save_dir = self._experiment.dir
        return self._experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.experiment.log(metrics, step=step)

    @property
    def name(self) -> str:
        # don't create an experiment if we don't have one
        name = self._experiment.project_name() if self._experiment else None
        return name

    @property
    def version(self) -> str:
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else None
