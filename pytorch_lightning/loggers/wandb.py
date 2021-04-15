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
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()

_WANDB_AVAILABLE = _module_available("wandb")

if _WANDB_AVAILABLE:
    import wandb
    from wandb.dummy import Dummy
    from wandb.wandb_run import Run
else:
    # needed for test mocks, these tests shall be updated
    # mypy complains about assignment to type
    wandb, Run, Dummy = None, None, None  # type: ignore

if TYPE_CHECKING:
    import torch


class WandbLogger(LightningLoggerBase):
    r"""
    Log using `Weights and Biases <https://www.wandb.com/>`_.

    Install it with pip:

    .. code-block:: bash

        pip install wandb

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved (wandb dir by default).
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        version: Same as id.
        anonymous: Enables or explicitly disables anonymous logging.
        project: The name of the project to which this run will belong.
        log_model: Save checkpoints in wandb dir to upload on W&B servers.
        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        \**kwargs: Additional arguments like `entity`, `group`, `tags`, etc. used by
            :func:`wandb.init` can be passed as keyword arguments in this logger.

    Raises:
        ImportError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline``is set to ``True``.

    Example::

        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer
        wandb_logger = WandbLogger()
        trainer = Trainer(logger=wandb_logger)

    Note: When logging manually through `wandb.log` or `trainer.logger.experiment.log`,
    make sure to use `commit=False` so the logging step does not increase.

    See Also:
        - `Tutorial <https://colab.research.google.com/drive/16d1uctGaw2y9KhGBlINNTsWpmlXdJwRW?usp=sharing>`__
          on how to use W&B with PyTorch Lightning
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__

    """

    LOGGER_JOIN_CHAR = '-'

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = False,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment: Optional[Run] = None,
        prefix: Optional[str] = '',
        sync_step: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        if wandb is None:
            raise ImportError(
                'You want to use `wandb` logger which is not installed yet,'  # pragma: no-cover
                ' install it with `pip install wandb`.'
            )

        if offline and log_model:
            raise MisconfigurationException(
                f'Providing log_model={log_model} and offline={offline} is an invalid configuration'
                ' since model checkpoints cannot be uploaded in offline mode.\n'
                'Hint: Set `offline=False` to log your model.'
            )

        if sync_step is not None:
            warning_cache.warn(
                "`WandbLogger(sync_step=(True|False))` is deprecated in v1.2.1 and will be removed in v1.5."
                " Metrics are now logged separately and automatically synchronized.", DeprecationWarning
            )

        super().__init__()
        self._name = name
        self._save_dir = save_dir
        self._offline = offline
        self._id = version or id
        self._anonymous = 'allow' if anonymous else None
        self._project = project
        self._log_model = log_model
        self._prefix = prefix
        self._experiment: Optional[Union[Run, Dummy]] = experiment
        self._kwargs = kwargs

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state['_id'] = self._experiment.id if self._experiment is not None else None

        # cannot be pickled
        state['_experiment'] = None
        return state

    # https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @rank_zero_experiment
    def experiment(self) -> Union[Run, Dummy]:
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
                name=self._name,
                dir=self._save_dir,
                project=self._project,
                anonymous=self._anonymous,
                id=self._id,
                resume='allow',
                **self._kwargs
            ) if wandb.run is None else wandb.run

            # save checkpoints in wandb dir to upload on W&B servers
            if self._save_dir is None:
                self._save_dir = self._experiment.dir

            # define default x-axis (for latest wandb versions)
            getattr(self._experiment, 'define_metric', lambda *_: None)("trainer/global_step")
            getattr(self._experiment, 'define_metric',
                    lambda *_: None)("*", step_metric='trainer/global_step', step_sync=True)

        return self._experiment

    def watch(self, model: 'torch.nn.Module', log: str = 'gradients', log_freq: int = 100) -> None:
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union['torch.Tensor', float]], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metrics = self._add_prefix(metrics)

        metrics = apply_to_collection(metrics, torch.Tensor, lambda x: x.item())
        if step is not None:
            metrics.update({'trainer/global_step': step})
        self.experiment.log(metrics)

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def name(self) -> str:
        # don't create an experiment if we don't have one
        return self._experiment.project_name() if self._experiment else self._name

    @property
    def version(self) -> str:
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else self._id

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # upload all checkpoints from saving dir
        if self._log_model:
            wandb.save(os.path.join(str(self.save_dir), "*.ckpt"))
