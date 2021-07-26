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
import operator
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union
from weakref import ReferenceType

import torch.nn as nn

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _compare_version
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()

_WANDB_AVAILABLE = _module_available("wandb")
_WANDB_GREATER_EQUAL_0_10_22 = _compare_version("wandb", operator.ge, "0.10.22")

try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:
    # needed for test mocks, these tests shall be updated
    wandb, Run = None, None


class WandbLogger(LightningLoggerBase):
    r"""
    Log using `Weights and Biases <https://docs.wandb.ai/integrations/lightning>`_.

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
        log_model: Log checkpoints created by :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
            as W&B artifacts.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        \**kwargs: Arguments passed to :func:`wandb.init` like `entity`, `group`, `tags`, etc.

    Raises:
        ImportError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline``is set to ``True``.

    Example::

        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning import Trainer

        # instrument experiment with W&B
        wandb_logger = WandbLogger(project='MNIST', log_model='all')
        trainer = Trainer(logger=wandb_logger)

        # log gradients and model topology
        wandb_logger.watch(model)

    See Also:
        - `Demo in Google Colab <http://wandb.me/lightning>`__ with model logging
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        sync_step: Optional[bool] = None,
        **kwargs,
    ):
        if wandb is None:
            raise ImportError(
                "You want to use `wandb` logger which is not installed yet,"  # pragma: no-cover
                " install it with `pip install wandb`."
            )

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        if log_model and not _WANDB_GREATER_EQUAL_0_10_22:
            warning_cache.warn(
                f"Providing log_model={log_model} requires wandb version >= 0.10.22"
                " for logging associated model metadata.\n"
                "Hint: Upgrade with `pip install --ugrade wandb`."
            )

        if sync_step is not None:
            warning_cache.deprecation(
                "`WandbLogger(sync_step=(True|False))` is deprecated in v1.2.1 and will be removed in v1.5."
                " Metrics are now logged separately and automatically synchronized."
            )

        super().__init__()
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time = {}
        self._checkpoint_callback = None
        # set wandb init arguments
        anonymous_lut = {True: "allow", False: None}
        self._wandb_init = dict(
            name=name,
            project=project,
            id=version or id,
            dir=save_dir,
            resume="allow",
            anonymous=anonymous_lut.get(anonymous, anonymous),
        )
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        state["_id"] = self._experiment.id if self._experiment is not None else None

        # cannot be pickled
        state["_experiment"] = None
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
                os.environ["WANDB_MODE"] = "dryrun"
            self._experiment = wandb.init(**self._wandb_init) if wandb.run is None else wandb.run

        # define default x-axis (for latest wandb versions)
        if getattr(self._experiment, "define_metric", None):
            self._experiment.define_metric("trainer/global_step")
            self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment

    def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100):
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = self._add_prefix(metrics)
        if step is not None:
            self.experiment.log({**metrics, "trainer/global_step": step})
        else:
            self.experiment.log(metrics)

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

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        # log checkpoints as artifacts
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # log checkpoints as artifacts
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        # get checkpoints to be saved with associated score
        checkpoints = {
            checkpoint_callback.last_model_path: checkpoint_callback.current_score,
            checkpoint_callback.best_model_path: checkpoint_callback.best_model_score,
            **checkpoint_callback.best_k_models,
        }
        checkpoints = sorted((Path(p).stat().st_mtime, p, s) for p, s in checkpoints.items() if Path(p).is_file())
        checkpoints = [
            c for c in checkpoints if c[1] not in self._logged_model_time.keys() or self._logged_model_time[c[1]] < c[0]
        ]

        # log iteratively all new checkpoints
        for t, p, s in checkpoints:
            metadata = (
                {
                    "score": s,
                    "original_filename": Path(p).name,
                    "ModelCheckpoint": {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                            "_every_n_val_epochs",
                        ]
                        # ensure it does not break if `ModelCheckpoint` args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
                if _WANDB_GREATER_EQUAL_0_10_22
                else None
            )
            artifact = wandb.Artifact(name=f"model-{self.experiment.id}", type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt")
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
