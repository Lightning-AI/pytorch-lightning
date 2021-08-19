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
    "NeptuneLogger",
]

import logging
import operator
import os
from argparse import Namespace
from functools import reduce
from typing import Any, Dict, Optional, Set, Union
from weakref import ReferenceType

import torch

from pytorch_lightning import __version__
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only
from pytorch_lightning.utilities.imports import _compare_version
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

_NEPTUNE_AVAILABLE = _module_available("neptune")
_NEPTUNE_GREATER_EQUAL_0_9 = _NEPTUNE_AVAILABLE and _compare_version("neptune", operator.ge, "0.9.0")

if _NEPTUNE_AVAILABLE and _NEPTUNE_GREATER_EQUAL_0_9:
    try:
        from neptune import new as neptune
        from neptune.new.run import Run
        from neptune.new.exceptions import NeptuneLegacyProjectException
    except ImportError:
        import neptune
        from neptune.run import Run
        from neptune.exceptions import NeptuneLegacyProjectException
else:
    # needed for test mocks, and function signatures
    neptune, Run = None, None


log = logging.getLogger(__name__)

INTEGRATION_VERSION_KEY = "source_code/integrations/pytorch-lightning"

# kwargs used in NeptuneLogger for legacy client (current NeptuneLegacyLogger)
LEGACY_NEPTUNE_INIT_KWARGS = [
    "project_name",
    "offline_mode",
    "experiment_name",
    "experiment_id",
    "params",
    "properties",
    "upload_source_files",
    "abort_callback",
    "logger",
    "upload_stdout",
    "upload_stderr",
    "send_hardware_metrics",
    "run_monitoring_thread",
    "handle_uncaught_exceptions",
    "git_info",
    "hostname",
    "notebook_id",
    "notebook_path",
]

# kwargs used in legacy NeptuneLogger from neptune-pytorch-lightning package
LEGACY_NEPTUNE_LOGGER_KWARGS = [
    "base_namespace",
    "close_after_fit",
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
            api_key="ANONYMOUS",
            project="common/new-pytorch-lightning-integration",
            name="lightning-run",  # Optional
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from neptune.new.types import File

        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                acc = ...
                self.logger.experiment["train/acc"].log(acc)

                # log images
                img = ...
                self.logger.experiment["train/misclassified_images"].log(File.as_image(img))

            def any_lightning_module_function_or_hook(self):
                # log model checkpoint
                ...
                self.logger.experiment["checkpoints/epoch37"].upload("epoch=37.ckpt")

                # generic recipe
                metadata = ...
                self.logger.experiment["your/metadata/structure"].log(metadata)

    Check `Neptune docs <https://docs.neptune.ai/user-guides/logging-and-managing-runs-results/logging-runs-data>`_
    for more info about how to log various types metadata (scores, files, images, interactive visuals, CSVs, etc.).

    **Log after training is finished**

    If you want to log objects after the training is finished use ``close_after_fit=False``:

    .. code-block:: python

        neptune_logger = NeptuneLogger(
            close_after_fit=False,
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
        neptune_logger.experiment["test/confusion_matrix"].upload(File.as_image(fig))

    **Pass additional parameters to Neptune run**

    You can also pass `kwargs` to specify the run in the greater detail, like ``tags`` and ``description``:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            project="common/new-pytorch-lightning-integration",
            name="lightning-run",
            description="mlp quick run with pytorch-lightning",
            tags=["mlp", "quick-run"],
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
            If specified (e.g. "ABC-42"), connect to run with `sys/id` in project_name.
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

    LOGGER_JOIN_CHAR = "/"
    PARAMETERS_KEY = "hyperparams"
    ARTIFACTS_KEY = "artifacts"

    def __init__(
            self,
            *,  # force users to call `NeptuneLogger` initializer with `kwargs`
            api_key: Optional[str] = None,
            project: Optional[str] = None,
            name: Optional[str] = None,
            run: Optional['Run'] = None,
            log_model_checkpoints: Optional[bool] = True,
            prefix: str = "training",
            **neptune_run_kwargs):

        # verify if user passed proper init arguments
        self._verify_input_arguments(api_key, project, name, run, neptune_run_kwargs)

        super().__init__()
        self._api_key = api_key
        self._project = project
        self._name = name
        self._neptune_run_kwargs = neptune_run_kwargs
        self._log_model_checkpoints = log_model_checkpoints
        self._prefix = prefix

        self._run_instance = run  # if run is None, instance will be initialized in first call to `run()`

    def _construct_path_with_prefix(self, *keys) -> str:
        """Return sequence of keys joined by `LOGGER_JOIN_CHAR`, started with
        `_prefix` if defined."""
        if self._prefix:
            return self.LOGGER_JOIN_CHAR.join([self._prefix, *keys])
        return self.LOGGER_JOIN_CHAR.join(keys)

    @staticmethod
    def _verify_input_arguments(
            api_key: Optional[str],
            project: Optional[str],
            name: Optional[str],
            run: Optional['Run'],
            neptune_run_kwargs: dict):

        # check if user used legacy kwargs expected in `NeptuneLegacyLogger`
        used_legacy_kwargs = [
            legacy_kwarg for legacy_kwarg in neptune_run_kwargs
            if legacy_kwarg in LEGACY_NEPTUNE_INIT_KWARGS
        ]
        if used_legacy_kwargs:
            raise ValueError(
                # TODO: product - review text
                f"Following kwargs used by you are deprecated: {used_legacy_kwargs}.\n"
                "If you are looking for the Neptune logger using legacy Python API it has been renamed to"
                " NeptuneLegacyLogger. The NeptuneLogger was re-written to use the neptune.new Python API"
                " (learn more: https://neptune.ai/blog/neptune-new).\n"
                "You should use arguments accepted by either NeptuneLogger.init or neptune.init"
            )

        # check if user used legacy kwargs expected in `NeptuneLogger` from neptune-pytorch-lightning package
        used_legacy_neptune_kwargs = [
            legacy_kwarg for legacy_kwarg in neptune_run_kwargs
            if legacy_kwarg in LEGACY_NEPTUNE_LOGGER_KWARGS
        ]
        if used_legacy_neptune_kwargs:
            raise ValueError(
                # TODO: product - review text
                f"Following kwargs used by you are deprecated: {used_legacy_neptune_kwargs}.\n"
            )

        # check if user passed new client `Run` object
        if run is not None and not isinstance(run, Run):
            raise ValueError(
                # TODO: product - review text
                "Run parameter expected to be of type `neptune.new.Run`.\n"
                " NeptuneLegacyLogger. The NeptuneLogger was re-written to use the neptune.new Python API"
                " (learn more: https://neptune.ai/blog/neptune-new)."
            )

        # check if user passed redundant neptune.init arguments when passed run
        any_neptune_init_arg_passed = any(
            (arg is not None for arg in [api_key, project, name])
        ) or neptune_run_kwargs
        if run is not None and any_neptune_init_arg_passed:
            raise ValueError(
                # TODO: product - review text
                "When run object is passed you can't specify other neptune properties.\n"
                " (learn more: https://neptune.ai/blog/neptune-new)."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Run instance can"t be pickled
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
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # skipcq: PYL-W0221
        r"""
        Log hyper-parameters to the run.

        Params will be logged using the ``param__`` scheme, for example: ``param__batch_size``, ``param__lr``.

        **Note**

        You can also log parameters by directly using the logger instance:
        ``neptune_logger.experiment["model/hyper-parameters"] = params_dict``.

        In this way you can keep hierarchical structure of the parameters.

        Args:
            params: `dict`.
                Python dictionary structure with parameters.

        Example::

            from pytorch_lightning.loggers import NeptuneLogger

            PARAMS = {
                "batch_size": 64,
                "lr": 0.07,
                "decay_factor": 0.97
            }

            neptune_logger = NeptuneLogger(
                api_key="ANONYMOUS",
                close_after_fit=False,
                project="common/new-pytorch-lightning-integration"
            )

            neptune_logger.log_hyperparams(PARAMS)
        """
        params = self._convert_params(params)
        params = self._sanitize_callable_params(params)

        parameters_key = self.PARAMETERS_KEY
        parameters_key = self._construct_path_with_prefix(parameters_key)

        self.run[parameters_key] = params

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        """Log metrics (numeric values) in Neptune runs.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values.
            step: Step number at which the metrics should be recorded, currently ignored.
        """
        if rank_zero_only.rank != 0:
            raise ValueError("run tried to log from global_rank != 0")

        metrics = self._add_prefix(metrics)

        for key, val in metrics.items():
            # `step` is ignored because Neptune expects strictly increasing step values which
            # Lighting does not always guarantee.
            self.experiment[key].log(val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status:
            self.experiment[
                self._construct_path_with_prefix("status")
            ] = status

        super().finalize(status)

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory of the experiment which in this case is ``None`` because Neptune does not save
        locally.

        Returns:
            the root directory where experiment logs get saved
        """
        return os.path.join(os.getcwd(), ".neptune")

    def log_model_summary(self, model, max_depth=-1):
        model_str = str(ModelSummary(model=model, max_depth=max_depth))
        self.experiment[
            self._construct_path_with_prefix("model/summary")
        ] = neptune.types.File.from_content(content=model_str, extension="txt")

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        """
        Called after model checkpoint callback saves a new checkpoint

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        if not self._log_model_checkpoints:
            return

        file_names = set()
        checkpoints_namespace = self._construct_path_with_prefix("model/checkpoints")

        # save last model
        if checkpoint_callback.last_model_path:
            model_last_name = self._get_full_model_name(checkpoint_callback.last_model_path, checkpoint_callback)
            file_names.add(model_last_name)
            self.experiment[f"{checkpoints_namespace}/{model_last_name}"].upload(
                checkpoint_callback.last_model_path
            )

        # save best k models
        for key in checkpoint_callback.best_k_models.keys():
            model_name = self._get_full_model_name(key, checkpoint_callback)
            file_names.add(model_name)
            self.experiment[f"{checkpoints_namespace}/{model_name}"].upload(key)

        # remove old models logged to experiment if they are not part of best k models at this point
        if self.experiment.exists(checkpoints_namespace):
            exp_structure = self.experiment.get_structure()
            uploaded_model_names = self._get_full_model_names_from_exp_structure(
                exp_structure,
                checkpoints_namespace
            )

            for file_to_drop in list(uploaded_model_names - file_names):
                del self.experiment[f"{checkpoints_namespace}/{file_to_drop}"]

        # log best model path and best model score
        if checkpoint_callback.best_model_path:
            self.experiment[
                self._construct_path_with_prefix("model/best_model_path")
            ] = checkpoint_callback.best_model_path
        if checkpoint_callback.best_model_score:
            self.experiment[
                self._construct_path_with_prefix("model/best_model_score")
            ] = checkpoint_callback.best_model_score.cpu().detach().numpy()

    @staticmethod
    def _get_full_model_name(model_path: str, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> str:
        """Returns model name which is string `modle_path` appended to `checkpoint_callback.dirpath`.
        The problem here is that model name may contain slashes('/')."""
        expected_model_path = f"{checkpoint_callback.dirpath}/"
        if not model_path.startswith(expected_model_path):
            raise ValueError(f"{model_path} was expected to start with {expected_model_path}.")
        return model_path[len(expected_model_path):]

    @classmethod
    def _get_full_model_names_from_exp_structure(cls, exp_structure: dict, namespace: str) -> Set[str]:
        """Returns all paths to properties which were already logged in `namespace`"""
        structure_keys = namespace.split(cls.LOGGER_JOIN_CHAR)
        uploaded_models_dict = reduce(lambda d, k: d[k], [exp_structure, *structure_keys])
        return set(cls._dict_paths(uploaded_models_dict))

    @classmethod
    def _dict_paths(cls, d: dict, path_in_build: str = None):
        for k, v in d.items():
            path = f"{path_in_build}/{k}" if path_in_build is not None else k
            if not isinstance(v, dict):
                yield path
            else:
                yield from cls._dict_paths(v, path)

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment if not in offline mode else "offline-name".
        """
        return "NeptuneLogger"

    @property
    def version(self) -> str:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if not in offline mode else "offline-id-1234".
        """
        return self.run._short_id  # skipcq: PYL-W0212

    @staticmethod
    def _raise_deprecated_api_usage(f_name, sample_code):
        # TODO: product - review text
        raise ValueError(f"The function you've used is deprecated.\n"
                         f"If you are looking for the Neptune logger using legacy Python API it has been renamed to"
                         f" NeptuneLegacyLogger. The NeptuneLogger was re-written to use the neptune.new Python API"
                         f" (learn more: https://neptune.ai/blog/neptune-new).\n"
                         f"Instead of `logger.{f_name}` you can use:\n"
                         f"\t{sample_code}")

    @rank_zero_only
    def log_metric(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_metric", f"logger.run['{self._prefix}/key'].log(42)")

    @rank_zero_only
    def log_text(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_text", f"logger.run['{self._prefix}/key'].log('text')")

    @rank_zero_only
    def log_image(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_image",
                                         f"logger.run['{self._prefix}/key'].log(File('path_to_image'))")

    @rank_zero_only
    def log_artifact(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_artifact",
                                         f"logger.run['{self._prefix}/{self.ARTIFACTS_KEY}/key'].log('path_to_file')")

    @rank_zero_only
    def set_property(self, *args, **kwargs):
        self._raise_deprecated_api_usage("log_artifact",
                                         f"logger.run['{self._prefix}/{self.PARAMETERS_KEY}/key'].log(value)")

    @rank_zero_only
    def append_tags(self, *args, **kwargs):
        self._raise_deprecated_api_usage("append_tags",
                                         "logger.run['sys/tags'].add(['foo', 'bar'])")
