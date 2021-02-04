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
Weights and Biases Logger
-------------------------
"""
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warning_utils import WarningCache

_WANDB_AVAILABLE = _module_available("wandb")

try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:
    # needed for test mocks, these tests shall be updated
    wandb, Run = None, None


class WandbLogger(LightningLoggerBase):
    r"""
    Log using `Weights and Biases <https://www.wandb.com/>`_.

    Install it with pip:

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
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        experiment: WandB experiment object.
        prefix: A string to put at the beginning of metric keys.
        \**kwargs: Additional arguments like `entity`, `group`, `tags`, etc. used by
            :func:`wandb.init` can be passed as keyword arguments in this logger.

    Example::

    .. code-block:: python

        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer
        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)

    Note: When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
    make sure to use `commit=False` so the logging step does not increase.

    See Also:
        - `Tutorial <https://app.wandb.ai/cayush/pytorchlightning/reports/
          Use-Pytorch-Lightning-with-Weights-%26-Biases--Vmlldzo2NjQ1Mw>`__
          on how to use W&B with Pytorch Lightning.

    """

    LOGGER_JOIN_CHAR = '-'

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        id: Optional[str] = None,
        anonymous: bool = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: bool = False,
        experiment=None,
        prefix: str = '',
        **kwargs
    ):
        if wandb is None:
            raise ImportError('You want to use `wandb` logger which is not installed yet,'  # pragma: no-cover
                              ' install it with `pip install wandb`.')

        if offline and log_model:
            raise MisconfigurationException(
                f'Providing log_model={log_model} and offline={offline} is an invalid configuration'
                ' since model checkpoints cannot be uploaded in offline mode.\n'
                'Hint: Set `offline=False` to log your model.'
            )

        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._anonymous = 'allow' if anonymous else None
        self._id = version or id
        self._project = project
        self._experiment = experiment
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._kwargs = kwargs
        # logging multiple Trainer on a single W&B run (k-fold, resuming, etc)
        self._step_offset = 0
        self.warning_cache = WarningCache()

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state['_id'] = self._experiment.id if self._experiment is not None else None

        # cannot be pickled
        state['_experiment'] = None
        return state

    @property
    @rank_zero_experiment
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
                id=self._id, resume='allow', **self._kwargs) if wandb.run is None else wandb.run

            # offset logging step when resuming a run
            self._step_offset = self._experiment.step

            # save checkpoints in wandb dir to upload on W&B servers
            if self._save_dir is None:
                self._save_dir = self._experiment.dir
        return self._experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metrics = self._add_prefix(metrics)
        if step is not None and step + self._step_offset < self.experiment.step:
            self.warning_cache.warn('Trying to log at a previous step. Use `commit=False` when logging metrics manually.')
        self.experiment.log(metrics, step=(step + self._step_offset) if step is not None else None)

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._experiment.project_name() if self._experiment else self._name

    @property
    def version(self) -> Optional[str]:
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else self._id

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # offset future training logged on same W&B run
        if self._experiment is not None:
            self._step_offset = self._experiment.step

        # upload all checkpoints from saving dir
        if self._log_model:
            wandb.save(os.path.join(self.save_dir, "*.ckpt"))
