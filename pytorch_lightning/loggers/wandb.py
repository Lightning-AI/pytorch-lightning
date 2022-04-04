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
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
from weakref import ReferenceType

import torch.nn as nn

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _WANDB_GREATER_EQUAL_0_10_22, _WANDB_GREATER_EQUAL_0_12_10
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params, _flatten_dict, _sanitize_callable_params
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

try:
    import wandb
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run = None, None


class WandbLogger(LightningLoggerBase):
    r"""
    Log using `Weights and Biases <https://docs.wandb.ai/integrations/lightning>`_.

    **Installation and set-up**

    Install with pip:

    .. code-block:: bash

        pip install wandb

    Create a `WandbLogger` instance:

    .. code-block:: python

        from pytorch_lightning.loggers import WandbLogger

        wandb_logger = WandbLogger(project="MNIST")

    Pass the logger instance to the `Trainer`:

    .. code-block:: python

        trainer = Trainer(logger=wandb_logger)

    A new W&B run will be created when training starts if you have not created one manually before with `wandb.init()`.

    **Log metrics**

    Log from :class:`~pytorch_lightning.core.lightning.LightningModule`:

    .. code-block:: python

        class LitModule(LightningModule):
            def training_step(self, batch, batch_idx):
                self.log("train/loss", loss)

    Use directly wandb module:

    .. code-block:: python

        wandb.log({"train/loss": loss})

    **Log hyper-parameters**

    Save :class:`~pytorch_lightning.core.lightning.LightningModule` parameters:

    .. code-block:: python

        class LitModule(LightningModule):
            def __init__(self, *args, **kwarg):
                self.save_hyperparameters()

    Add other config parameters:

    .. code-block:: python

        # add one parameter
        wandb_logger.experiment.config["key"] = value

        # add multiple parameters
        wandb_logger.experiment.config.update({key1: val1, key2: val2})

        # use directly wandb module
        wandb.config["key"] = value
        wandb.config.update()

    **Log gradients, parameters and model topology**

    Call the `watch` method for automatically tracking gradients:

    .. code-block:: python

        # log gradients and model topology
        wandb_logger.watch(model)

        # log gradients, parameter histogram and model topology
        wandb_logger.watch(model, log="all")

        # change log frequency of gradients and parameters (100 steps by default)
        wandb_logger.watch(model, log_freq=500)

        # do not log graph (in case of errors)
        wandb_logger.watch(model, log_graph=False)

    The `watch` method adds hooks to the model which can be removed at the end of training:

    .. code-block:: python

        wandb_logger.unwatch(model)

    **Log model checkpoints**

    Log model checkpoints at the end of training:

    .. code-block:: python

        wandb_logger = WandbLogger(log_model=True)

    Log model checkpoints as they get created during training:

    .. code-block:: python

        wandb_logger = WandbLogger(log_model="all")

    Custom checkpointing can be set up through :class:`~pytorch_lightning.callbacks.ModelCheckpoint`:

    .. code-block:: python

        # log model only if `val_accuracy` increases
        wandb_logger = WandbLogger(log_model="all")
        checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
        trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])

    `latest` and `best` aliases are automatically set to easily retrieve a model checkpoint:

    .. code-block:: python

        # reference can be retrieved in artifacts panel
        # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
        checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"

        # download checkpoint locally (if not already cached)
        run = wandb.init(project="MNIST")
        artifact = run.use_artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()

        # load checkpoint
        model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    **Log media**

    Log text with:

    .. code-block:: python

        # using columns and data
        columns = ["input", "label", "prediction"]
        data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]
        wandb_logger.log_text(key="samples", columns=columns, data=data)

        # using a pandas DataFrame
        wandb_logger.log_text(key="samples", dataframe=my_dataframe)

    Log images with:

    .. code-block:: python

        # using tensors, numpy arrays or PIL images
        wandb_logger.log_image(key="samples", images=[img1, img2])

        # adding captions
        wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

        # using file path
        wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

    More arguments can be passed for logging segmentation masks and bounding boxes. Refer to
    `Image Overlays documentation <https://docs.wandb.ai/guides/track/log/media#image-overlays>`_.

    **Log Tables**

    `W&B Tables <https://docs.wandb.ai/guides/data-vis>`_ can be used to log, query and analyze tabular data.

    They support any type of media (text, image, video, audio, molecule, html, etc) and are great for storing,
    understanding and sharing any form of data, from datasets to model predictions.

    .. code-block:: python

        columns = ["caption", "image", "sound"]
        data = [["cheese", wandb.Image(img_1), wandb.Audio(snd_1)], ["wine", wandb.Image(img_2), wandb.Audio(snd_2)]]
        wandb_logger.log_table(key="samples", columns=columns, data=data)

    See Also:
        - `Demo in Google Colab <http://wandb.me/lightning>`__ with hyperparameter search and model logging
        - `W&B Documentation <https://docs.wandb.ai/integrations/lightning>`__

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved (wandb dir by default).
        offline: Run offline (data can be streamed later to wandb servers).
        id: Sets the version, mainly used to resume a previous run.
        version: Same as id.
        anonymous: Enables or explicitly disables anonymous logging.
        project: The name of the project to which this run will belong.
        log_model: Log checkpoints created by :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
            as W&B artifacts. `latest` and `best` aliases are automatically set.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        \**kwargs: Arguments passed to :func:`wandb.init` like `entity`, `group`, `tags`, etc.

    Raises:
        ModuleNotFoundError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline`` is set to ``True``.

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
        log_model: Union[str, bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
        **kwargs,
    ):
        if wandb is None:
            raise ModuleNotFoundError(
                "You want to use `wandb` logger which is not installed yet,"
                " install it with `pip install wandb`."  # pragma: no-cover
            )

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        if log_model and not _WANDB_GREATER_EQUAL_0_10_22:
            rank_zero_warn(
                f"Providing log_model={log_model} requires wandb version >= 0.10.22"
                " for logging associated model metadata.\n"
                "Hint: Upgrade with `pip install --upgrade wandb`."
            )

        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
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
        # start wandb run (to create an attach_id for distributed modes)
        if _WANDB_GREATER_EQUAL_0_12_10:
            wandb.require("service")
            _ = self.experiment

    def __getstate__(self):
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.project_name()

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

        .. code-block:: python

            self.logger.experiment.some_wandb_function()

        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment

    def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100, log_graph: bool = True):
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            self.experiment.log({**metrics, "trainer/global_step": step})
        else:
            self.experiment.log(metrics)

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log a Table containing any object type (text, image, audio, video, molecule, html, etc).

        Can be defined either with `columns` and `data` or with `dataframe`.
        """

        metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_text(
        self,
        key: str,
        columns: List[str] = None,
        data: List[List[str]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log text as a Table.

        Can be defined either with `columns` and `data` or with `dataframe`.
        """

        self.log_table(key, columns, data, dataframe, step)

    @rank_zero_only
    def log_image(self, key: str, images: List[Any], step: Optional[int] = None, **kwargs: str) -> None:
        """Log images (tensors, numpy arrays, PIL Images or file paths).

        Optional kwargs are lists passed to each image (ex: caption, masks, boxes).
        """
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step)

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.
        """
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if the experiment exists else the name given to the constructor.
        """
        # don't create an experiment if we don't have one
        return self._experiment.project_name() if self._experiment else self._name

    @property
    def version(self) -> Optional[str]:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if the experiment exists else the id given to the constructor.
        """
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
