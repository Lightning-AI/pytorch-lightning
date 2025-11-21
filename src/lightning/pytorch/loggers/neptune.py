# Copyright The Lightning AI team.
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
Neptune Logger
--------------
"""

import logging
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Optional, Union

from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.imports import _raise_enterprise_not_available
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from neptune import Run
    from neptune.handler import Handler

log = logging.getLogger(__name__)


class NeptuneLogger(Logger):
    r"""Log using `Neptune <https://docs.neptune.ai/integrations/lightning/>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune

    or conda:

    .. code-block:: bash

        conda install -c conda-forge neptune-client

    **Quickstart**

    Pass a NeptuneLogger instance to the Trainer to log metadata with Neptune:

    .. code-block:: python


        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import NeptuneLogger
        import neptune

        neptune_logger = NeptuneLogger(
            api_key=neptune.ANONYMOUS_API_TOKEN,  # replace with your own
            project="common/pytorch-lightning-integration",  # format "workspace-name/project-name"
            tags=["training", "resnet"],  # optional
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    **How to use NeptuneLogger?**

    Use the logger anywhere in your :class:`~lightning.pytorch.core.LightningModule` as follows:

    .. code-block:: python

        from neptune.types import File
        from lightning.pytorch import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                acc = ...
                self.append("train/loss", loss)

            def any_lightning_module_function_or_hook(self):
                # log images
                img = ...
                self.logger.experiment["train/misclassified_images"].append(File.as_image(img))

                # generic recipe
                metadata = ...
                self.logger.experiment["your/metadata/structure"] = metadata

    Note that the syntax ``self.logger.experiment["your/metadata/structure"].append(metadata)`` is specific to
    Neptune and extends the logger capabilities. It lets you log various types of metadata, such as
    scores, files, images, interactive visuals, and CSVs.
    Refer to the `Neptune docs <https://docs.neptune.ai/logging/methods>`_
    for details.
    You can also use the regular logger methods ``log_metrics()``, and ``log_hyperparams()`` with NeptuneLogger.

    **Log after fitting or testing is finished**

    You can log objects after the fitting or testing methods are finished:

    .. code-block:: python

        neptune_logger = NeptuneLogger(project="common/pytorch-lightning-integration")

        trainer = pl.Trainer(logger=neptune_logger)
        model = ...
        datamodule = ...
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        # Log objects after `fit` or `test` methods
        # model summary
        neptune_logger.log_model_summary(model=model, max_depth=-1)

        # generic recipe
        metadata = ...
        neptune_logger.experiment["your/metadata/structure"] = metadata

    **Log model checkpoints**

    If you have :class:`~lightning.pytorch.callbacks.ModelCheckpoint` configured,
    the Neptune logger automatically logs model checkpoints.
    Model weights will be uploaded to the "model/checkpoints" namespace in the Neptune run.
    You can disable this option with:

    .. code-block:: python

        neptune_logger = NeptuneLogger(log_model_checkpoints=False)

    **Pass additional parameters to the Neptune run**

    You can also pass ``neptune_run_kwargs`` to add details to the run, like ``tags`` or ``description``:

    .. testcode::
        :skipif: not _NEPTUNE_AVAILABLE

        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            project="common/pytorch-lightning-integration",
            name="lightning-run",
            description="mlp quick run with pytorch-lightning",
            tags=["mlp", "quick-run"],
        )
        trainer = Trainer(max_epochs=3, logger=neptune_logger)

    Check `run documentation <https://docs.neptune.ai/api/neptune/#init_run>`_
    for more info about additional run parameters.

    **Details about Neptune run structure**

    Runs can be viewed as nested dictionary-like structures that you can define in your code.
    Thanks to this you can easily organize your metadata in a way that is most convenient for you.

    The hierarchical structure that you apply to your metadata is reflected in the Neptune web app.

    See also:
        - Read about
          `what objects you can log to Neptune <https://docs.neptune.ai/logging/what_you_can_log/>`_.
        - Check out an `example run <https://app.neptune.ai/o/common/org/pytorch-lightning-integration/e/PTL-1/all>`_
          with multiple types of metadata logged.
        - For more detailed examples, see the
          `user guide <https://docs.neptune.ai/integrations/lightning/>`_.

    Args:
        api_key: Optional.
            Neptune API token, found on https://www.neptune.ai upon registration.
            You should save your token to the `NEPTUNE_API_TOKEN`
            environment variable and leave the api_key argument out of your code.
            Instructions: `Setting your API token <https://docs.neptune.ai/setup/setting_api_token/>`_.
        project: Optional.
            Name of a project in the form "workspace-name/project-name", for example "tom/mask-rcnn".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable is used.
            You need to create the project on https://www.neptune.ai first.
        name: Optional. Editable name of the run.
            The run name is displayed in the Neptune web app.
        run: Optional. Default is ``None``. A Neptune ``Run`` object.
            If specified, this existing run will be used for logging, instead of a new run being created.
            You can also pass a namespace handler object; for example, ``run["test"]``, in which case all
            metadata is logged under the "test" namespace inside the run.
        log_model_checkpoints: Optional. Default is ``True``. Log model checkpoint to Neptune.
            Works only if ``ModelCheckpoint`` is passed to the ``Trainer``.
        prefix: Optional. Default is ``"training"``. Root namespace for all metadata logging.
        \**neptune_run_kwargs: Additional arguments like ``tags``, ``description``, ``capture_stdout``, etc.
            used when a run is created.

    Raises:
        ModuleNotFoundError:
            If the required Neptune package is not installed.
        ValueError:
            If an argument passed to the logger's constructor is incorrect.

    """

    LOGGER_JOIN_CHAR = "/"
    PARAMETERS_KEY = "hyperparams"
    ARTIFACTS_KEY = "artifacts"

    def __init__(
        self,
        *,  # force users to call `NeptuneLogger` initializer with `kwargs`
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        run: Optional[Union["Run", "Handler"]] = None,
        log_model_checkpoints: Optional[bool] = True,
        prefix: str = "training",
        **neptune_run_kwargs: Any,
    ):
        _raise_enterprise_not_available()
        super().__init__()
        from pytorch_lightning_enterprise.loggers.neptune import NeptuneLogger as EnterpriseNeptuneLogger

        self.logger_impl = EnterpriseNeptuneLogger(
            api_key=api_key,
            project=project,
            name=name,
            run=run,
            log_model_checkpoints=log_model_checkpoints,
            prefix=prefix,
            **neptune_run_kwargs,
        )

    @property
    @rank_zero_experiment
    def experiment(self) -> "Run":
        r"""Actual Neptune run object. Allows you to use neptune logging features in your
        :class:`~lightning.pytorch.core.LightningModule`.

        Example::

            class LitModel(LightningModule):
                def training_step(self, batch, batch_idx):
                    # log metrics
                    acc = ...
                    self.logger.experiment["train/acc"].append(acc)

                    # log images
                    img = ...
                    self.logger.experiment["train/misclassified_images"].append(File.as_image(img))

        Note that the syntax ``self.logger.experiment["your/metadata/structure"].append(metadata)``
        is specific to Neptune and extends the logger capabilities.
        It lets you log various types of metadata, such as scores, files,
        images, interactive visuals, and CSVs. Refer to the
        `Neptune docs <https://docs.neptune.ai/logging/methods>`_
        for more detailed explanations.
        You can also use the regular logger methods ``log_metrics()``, and ``log_hyperparams()``
        with NeptuneLogger.

        """
        return self.logger_impl.experiment

    @property
    @rank_zero_experiment
    def run(self) -> "Run":
        return self.logger_impl.run

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        return self.logger_impl.log_hyperparams(params)

    @override
    @rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: dict[str, Union[Tensor, float]], step: Optional[int] = None
    ) -> None:
        """Log metrics (numeric values) in Neptune runs.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values.
            step: Step number at which the metrics should be recorded

        """
        return self.logger_impl.log_metrics(metrics, step)

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        return self.logger_impl.finalize(status)

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """Gets the save directory of the experiment which in this case is ``None`` because Neptune does not save
        locally.

        Returns:
            the root directory where experiment logs get saved

        """
        return self.logger_impl.save_dir

    @rank_zero_only
    def log_model_summary(self, model: "pl.LightningModule", max_depth: int = -1) -> None:
        return self.logger_impl.log_model_summary(model, max_depth)

    @override
    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: Checkpoint) -> None:
        """Automatically log checkpointed model. Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance

        """
        return self.logger_impl.after_save_checkpoint(checkpoint_callback)

    @property
    @override
    def name(self) -> Optional[str]:
        """Return the experiment name or 'offline-name' when exp is run in offline mode."""
        return self.logger_impl.name

    @property
    @override
    def version(self) -> Optional[str]:
        """Return the experiment version.

        It's Neptune Run's short_id

        """
        return self.logger_impl.version
