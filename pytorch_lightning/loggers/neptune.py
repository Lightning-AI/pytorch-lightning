#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
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
#
"""
Neptune Logger
--------------
"""
__all__ = [
    'NeptuneLogger',
]

import logging
import operator
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import torch

from pytorch_lightning import __version__
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only
from pytorch_lightning.utilities.imports import _compare_version

_NEPTUNE_AVAILABLE = _module_available("neptune")
_NEPTUNE_GREATER_EQUAL_0_9 = _NEPTUNE_AVAILABLE and _compare_version("neptune", operator.ge, "0.9.0")

if _module_available("neptune"):
    from neptune import __version__ as neptune_versions

    _NEPTUNE_AVAILABLE = neptune_versions.startswith('0.9.') or neptune_versions.startswith('1.')
else:
    _NEPTUNE_AVAILABLE = False

if _NEPTUNE_AVAILABLE and _NEPTUNE_GREATER_EQUAL_0_9:
    try:
        from neptune import new as neptune
        from neptune.new.run import Run
        from neptune.new.exceptions import NeptuneLegacyProjectException, NeptuneOfflineModeFetchException
    except ImportError:
        import neptune
        from neptune.run import Run
        from neptune.exceptions import NeptuneLegacyProjectException, NeptuneOfflineModeFetchException
else:
    # needed for test mocks, and function signatures
    neptune, Run = None, None
log = logging.getLogger(__name__)

INTEGRATION_VERSION_KEY = 'source_code/integrations/pytorch-lightning'

LEGACY_NEPTUNE_INIT_KWARGS = [
    'project_name',
    'offline_mode',
    'experiment_name',
    'experiment_id',
    'params',
    'properties',
    'upload_source_files',
    'abort_callback',
    'logger',
    'upload_stdout',
    'upload_stderr',
    'send_hardware_metrics',
    'run_monitoring_thread',
    'handle_uncaught_exceptions',
    'git_info',
    'hostname',
    'notebook_id',
    'notebook_path',
]


class NeptuneLogger(LightningLoggerBase):
    r"""
    Log using `Neptune <https://neptune.ai>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune-client

    Pass NeptuneLogger instance to the Trainer to log metadata with Neptune:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        # Arguments passed to the "NeptuneLogger" are used to create new run in neptune.
        # We are using an "api_key" for the anonymous user "neptuner" but you can use your own.
        neptune_logger = NeptuneLogger(
            api_key='ANONYMOUS',
            project='common/new-pytorch-lightning-integration',
            name='lightning-run',  # Optional
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from neptune.new.types import File

        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                acc = ...
                self.logger.experiment['train/acc'].log(acc)

                # log images
                img = ...
                self.logger.experiment['train/misclassified_images'].log(File.as_image(img))

            def any_lightning_module_function_or_hook(self):
                # log model checkpoint
                ...
                self.logger.experiment['checkpoints/epoch37'].upload('epoch=37.ckpt')

                # generic recipe
                metadata = ...
                self.logger.experiment['your/metadata/structure'].log(metadata)

    Check `Neptune docs <https://docs.neptune.ai/user-guides/logging-and-managing-runs-results/logging-runs-data>`_
    for more info about how to log various types metadata (scores, files, images, interactive visuals, CSVs, etc.).

    **Log after training is finished**

    If you want to log objects after the training is finished use ``close_after_fit=False``:

    .. code-block:: python

        neptune_logger = NeptuneLogger(
            ...
            close_after_fit=False,
            ...
        )
        trainer = Trainer(logger=neptune_logger)
        trainer.fit(model)

        # Log metadata after trainer.fit() is done, for example diagnostics chart
        from neptune.new.types import File
        from scikitplot.metrics import plot_confusion_matrix
        import matplotlib.pyplot as plt
        ...
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        neptune_logger.experiment['test/confusion_matrix'].upload(File.as_image(fig))

    **Pass additional parameters to Neptune run**

    You can also pass `kwargs` to specify the run in the greater detail, like ``tags`` and ``description``:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            project='common/new-pytorch-lightning-integration',
            name='lightning-run',
            description='mlp quick run with pytorch-lightning',
            tags=['mlp', 'quick-run'],
            )
        trainer = Trainer(max_epochs=3, logger=neptune_logger)

    Check `run documentation <https://docs.neptune.ai/essentials/api-reference/run>`_
    for more info about additional run parameters.

    **Details about Neptune run structure**

    Runs can be viewed as nested dictionary-like structures that you can define in your code.
    Thanks to this you can easily organize your metadata in a way that is most convenient for you.

    The hierarchical structure that you apply to your metadata will be reflected later in the UI.

    You can organize this way any type of metadata - images, parameters, metrics, model checkpoint, CSV files, etc.

    See Also:
        You can read about `what object you can log to Neptune <https://docs.neptune.ai/user-guides/
        logging-and-managing-runs-results/logging-runs-data#what-objects-can-you-log-to-neptune>`_.
        Also check `example run <https://app.neptune.ai/o/common/org/new-pytorch-lightning-integration/e/NEWPL-8/all>`_
        with multiple type of metadata logged.

    Args:
        api_key: Optional.
            Neptune API token, found on https://neptune.ai upon registration.
            Read: `how to find and set Neptune API token <https://docs.neptune.ai/administration/security-and-privacy/
            how-to-find-and-set-neptune-api-token>`_.
            It is recommended to keep it in the `NEPTUNE_API_TOKEN`
            environment variable and then you can drop ``api_key=None``.
        project: Optional.
            Qualified name of a project in a form of "my_workspace/my_project" for example "tom/mask-rcnn".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
        close_after_fit: Optional default ``True``.
            If ``False`` the run will not be closed after training
            and additional metrics, images or artifacts can be logged.
        name: Optional. Editable name of the run.
            Run name appears in the "all metadata/sys" section in Neptune UI.
        run: Optional. Default is ``None``. The ID of the existing run.
            If specified (e.g. 'ABC-42'), connect to run with `sys/id` in project_name.
            Input argument "name" will be overridden based on fetched run data.
        prefix: A string to put at the beginning of metric keys.
        base_namespace: Parent namespace under which parameters and metrics will be stored.
        \**kwargs: Additional arguments like ``tags``, ``description``, ``capture_stdout``, ``capture_stderr`` etc.
            used when run is created.

    Raises:
        ImportError:
            If required Neptune package in version >=0.9 is not installed on the device.
        TypeError:
            If configured project has not been migrated to new structure yet.
    """

    LOGGER_JOIN_CHAR = '/'
    PARAMETERS_KEY = 'parameters'
    METRICS_KEY = 'metrics'
    ARTIFACTS_KEY = 'artifacts'

    def __init__(
            self,
            api_key: Optional[str] = None,
            project: Optional[str] = None,
            close_after_fit: Optional[bool] = True,
            name: Optional[str] = None,
            run: Optional[str] = None,
            prefix: str = '',
            base_namespace: str = '',
            **neptune_run_kwargs):
        used_legacy_kwargs = [
            legacy_kwarg for legacy_kwarg in neptune_run_kwargs.keys()
            if legacy_kwarg in LEGACY_NEPTUNE_INIT_KWARGS
        ]
        if used_legacy_kwargs:
            raise ValueError(
                f"Following kwargs used by you are deprecated: {used_legacy_kwargs}.\n"
                "If you are looking for the Neptune logger using legacy Python API it has been renamed to"
                " NeptuneLegacyLogger. The NeptuneLogger was re-written to use the neptune.new Python API"
                " (learn more: https://neptune.ai/blog/neptune-new).\n"
                "You should use arguments accepted by either NeptuneLogger.init or neptune.init"
            )

        super().__init__()
        self._project = project
        self._api_key = api_key
        self._neptune_run_kwargs = neptune_run_kwargs
        self._close_after_fit = close_after_fit
        self._name = name
        self._run_to_load = run  # particular id of exp to load e.g. 'ABC-42'
        self._prefix = prefix
        self._base_namespace = base_namespace

        self._run_instance = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Run instance can't be pickled
        state["_run_instance"] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual Neptune run object. Allows you to use neptune logging features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule`.

        Example::

            class LitModel(LightningModule):
                def training_step(self, batch, batch_idx):
                    # log metrics
                    acc = ...
                    self.logger.experiment["train/acc"].log(acc)

                    # log images
                    img = ...
                    self.logger.experiment["train/misclassified_images"].log(File.as_image(img))
        """
        return self.run

    @property
    def run(self) -> Run:
        if self._run_instance is None:
            try:
                self._run_instance = neptune.init(
                    project=self._project,
                    api_token=self._api_key,
                    run=self._run_to_load,
                    name=self._name,
                    **self._neptune_run_kwargs,
                )
                self._run_instance[INTEGRATION_VERSION_KEY] = __version__
            except NeptuneLegacyProjectException as e:
                raise TypeError(f"""
                    Project {self._project} has not been imported to new structure yet.
                    You can still integrate it with `NeptuneLegacyLogger`.
                    """) from e

        return self._run_instance

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        r"""
        Log hyper-parameters to the run.

        Params will be logged using the ``param__`` scheme, for example: ``param__batch_size``, ``param__lr``.

        **Note**

        You can also log parameters by directly using the logger instance:
        ``neptune_logger.experiment['model/hyper-parameters'] = params_dict``.

        In this way you can keep hierarchical structure of the parameters.

        Args:
            params: `dict`.
                Python dictionary structure with parameters.

        Example::

            from pytorch_lightning.loggers import NeptuneLogger

            PARAMS = {
                'batch_size': 64,
                'lr': 0.07,
                'decay_factor': 0.97
            }

            neptune_logger = NeptuneLogger(
                api_key='ANONYMOUS',
                close_after_fit=False,
                project='common/new-pytorch-lightning-integration'
            )

            neptune_logger.log_hyperparams(PARAMS)
        """
        params = self._convert_params(params)

        parameters_key = self.PARAMETERS_KEY
        if self._base_namespace:
            parameters_key = f'{self._base_namespace}/{parameters_key}'

        self.run[parameters_key] = params

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        """Log metrics (numeric values) in Neptune runs.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values.
            step: Step number at which the metrics should be recorded, currently ignored.
        """
        assert rank_zero_only.rank == 0, 'run tried to log from global_rank != 0'

        metrics = self._add_prefix(metrics)
        metrics_key = self.METRICS_KEY
        if self._base_namespace:
            metrics_key = f'{self._base_namespace}/{metrics_key}'

        for key, val in metrics.items():
            # `step` is ignored because Neptune expects strictly increasing step values which
            # Lighting does not always guarantee.
            self.experiment[f'{metrics_key}/{key}'].log(val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if self._close_after_fit:
            self.run.stop()

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory of the experiment which in this case is ``None`` because Neptune does not save
        locally.

        Returns:
            None
        """
        return None

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if not in offline mode else "offline-name".
        """
        return 'NeptuneLogger'

    @property
    def version(self) -> str:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if not in offline mode else "offline-id-1234".
        """
        return self.run._short_id

    def _raise_deprecated_api_usage(self, f_name, sample_code):
        raise ValueError(f"The function you've used is deprecated.\n"
                         f"If you are looking for the Neptune logger using legacy Python API it has been renamed to"
                         f" NeptuneLegacyLogger. The NeptuneLogger was re-written to use the neptune.new Python API"
                         f" (learn more: https://neptune.ai/blog/neptune-new).\n"
                         f"Instead of `logger.{f_name}` you can use:\n"
                         f"\t{sample_code}")

    @rank_zero_only
    def log_metric(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_metric", f"logger.run['{self.METRICS_KEY}/key'].log(42)")

    @rank_zero_only
    def log_text(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_text", f"logger.run['{self.METRICS_KEY}/key'].log('text')")

    @rank_zero_only
    def log_image(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_image",
                                         f"logger.run['{self.METRICS_KEY}/key'].log(File('path_to_image'))")

    @rank_zero_only
    def log_artifact(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_artifact",
                                         f"logger.run['{self.ARTIFACTS_KEY}/key'].log('path_to_file')")

    @rank_zero_only
    def set_property(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_artifact", f"logger.run['{self.PARAMETERS_KEY}/key'].log(value)")

    @rank_zero_only
    def append_tags(self, *args, **kwargs):
        self._raise_deprecated_api_usage("append_tags",
                                         "logger.run['sys/tags'].add(['foo', 'bar'])")
