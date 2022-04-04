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
Neptune Logger
--------------
"""
__all__ = [
    "NeptuneLogger",
]

import logging
import os
import warnings
from argparse import Namespace
from functools import reduce
from typing import Any, Callable, Dict, Generator, Mapping, Optional, Sequence, Set, Union
from weakref import ReferenceType

import torch

from pytorch_lightning import __version__
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.imports import _NEPTUNE_AVAILABLE, _NEPTUNE_GREATER_EQUAL_0_9
from pytorch_lightning.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_only

if _NEPTUNE_AVAILABLE and _NEPTUNE_GREATER_EQUAL_0_9:
    try:
        from neptune import new as neptune
        from neptune.new.exceptions import NeptuneLegacyProjectException, NeptuneOfflineModeFetchException
        from neptune.new.run import Run
        from neptune.new.types import File as NeptuneFile
    except ModuleNotFoundError:
        import neptune
        from neptune.exceptions import NeptuneLegacyProjectException, NeptuneOfflineModeFetchException
        from neptune.run import Run
        from neptune.types import File as NeptuneFile
else:
    # needed for test mocks, and function signatures
    neptune, Run, NeptuneFile = None, None, None

log = logging.getLogger(__name__)

_INTEGRATION_VERSION_KEY = "source_code/integrations/pytorch-lightning"

# kwargs used in previous NeptuneLogger version, now deprecated
_LEGACY_NEPTUNE_INIT_KWARGS = [
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
_LEGACY_NEPTUNE_LOGGER_KWARGS = [
    "base_namespace",
    "close_after_fit",
]


class NeptuneLogger(LightningLoggerBase):
    r"""
    Log using `Neptune <https://neptune.ai>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune-client

    or conda:

    .. code-block:: bash

        conda install -c conda-forge neptune-client

    **Quickstart**

    Pass NeptuneLogger instance to the Trainer to log metadata with Neptune:

    .. code-block:: python


        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            api_key="ANONYMOUS",  # replace with your own
            project="common/pytorch-lightning-integration",  # format "<WORKSPACE/PROJECT>"
            tags=["training", "resnet"],  # optional
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    **How to use NeptuneLogger?**

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from neptune.new.types import File
        from pytorch_lightning import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                acc = ...
                self.log("train/loss", loss)

            def any_lightning_module_function_or_hook(self):
                # log images
                img = ...
                self.logger.experiment["train/misclassified_images"].log(File.as_image(img))

                # generic recipe
                metadata = ...
                self.logger.experiment["your/metadata/structure"].log(metadata)

    Note that syntax: ``self.logger.experiment["your/metadata/structure"].log(metadata)`` is specific to Neptune
    and it extends logger capabilities. Specifically, it allows you to log various types of metadata
    like scores, files, images, interactive visuals, CSVs, etc.
    Refer to the `Neptune docs <https://docs.neptune.ai/you-should-know/logging-metadata#essential-logging-methods>`_
    for more detailed explanations.
    You can also use regular logger methods ``log_metrics()``, and ``log_hyperparams()`` with NeptuneLogger
    as these are also supported.

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
        neptune_logger.experiment["your/metadata/structure"].log(metadata)

    **Log model checkpoints**

    If you have :class:`~pytorch_lightning.callbacks.ModelCheckpoint` configured,
    Neptune logger automatically logs model checkpoints.
    Model weights will be uploaded to the: "model/checkpoints" namespace in the Neptune Run.
    You can disable this option:

    .. code-block:: python

        neptune_logger = NeptuneLogger(project="common/pytorch-lightning-integration", log_model_checkpoints=False)

    **Pass additional parameters to the Neptune run**

    You can also pass ``neptune_run_kwargs`` to specify the run in the greater detail, like ``tags`` or ``description``:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            project="common/pytorch-lightning-integration",
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
        - Read about
          `what object you can log to Neptune <https://docs.neptune.ai/you-should-know/what-can-you-log-and-display>`_.
        - Check `example run <https://app.neptune.ai/o/common/org/pytorch-lightning-integration/e/PTL-1/all>`_
          with multiple types of metadata logged.
        - For more detailed info check
          `user guide <https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning>`_.

    Args:
        api_key: Optional.
            Neptune API token, found on https://neptune.ai upon registration.
            Read: `how to find and set Neptune API token <https://docs.neptune.ai/administration/security-and-privacy/
            how-to-find-and-set-neptune-api-token>`_.
            It is recommended to keep it in the `NEPTUNE_API_TOKEN`
            environment variable and then you can drop ``api_key=None``.
        project: Optional.
            Name of a project in a form of "my_workspace/my_project" for example "tom/mask-rcnn".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
        name: Optional. Editable name of the run.
            Run name appears in the "all metadata/sys" section in Neptune UI.
        run: Optional. Default is ``None``. The Neptune ``Run`` object.
            If specified, this `Run`` will be used for logging, instead of a new Run.
            When run object is passed you can't specify other neptune properties.
        log_model_checkpoints: Optional. Default is ``True``. Log model checkpoint to Neptune.
            Works only if ``ModelCheckpoint`` is passed to the ``Trainer``.
        prefix: Optional. Default is ``"training"``. Root namespace for all metadata logging.
        \**neptune_run_kwargs: Additional arguments like ``tags``, ``description``, ``capture_stdout``, etc.
            used when run is created.

    Raises:
        ModuleNotFoundError:
            If required Neptune package in version >=0.9 is not installed on the device.
        TypeError:
            If configured project has not been migrated to new structure yet.
        ValueError:
            If argument passed to the logger's constructor is incorrect.
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
        run: Optional["Run"] = None,
        log_model_checkpoints: Optional[bool] = True,
        prefix: str = "training",
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
        **neptune_run_kwargs,
    ):
        # verify if user passed proper init arguments
        self._verify_input_arguments(api_key, project, name, run, neptune_run_kwargs)
        if neptune is None:
            raise ModuleNotFoundError(
                "You want to use the `Neptune` logger which is not installed yet, install it with"
                " `pip install neptune-client`."
            )

        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self._log_model_checkpoints = log_model_checkpoints
        self._prefix = prefix
        self._run_name = name
        self._project_name = project
        self._api_key = api_key
        self._run_instance = run
        self._neptune_run_kwargs = neptune_run_kwargs
        self._run_short_id = None

        if self._run_instance is not None:
            self._retrieve_run_data()

            # make sure that we've log integration version for outside `Run` instances
            self._run_instance[_INTEGRATION_VERSION_KEY] = __version__

    def _retrieve_run_data(self):
        try:
            self._run_instance.wait()
            self._run_short_id = self._run_instance["sys/id"].fetch()
            self._run_name = self._run_instance["sys/name"].fetch()
        except NeptuneOfflineModeFetchException:
            self._run_short_id = "OFFLINE"
            self._run_name = "offline-name"

    @property
    def _neptune_init_args(self):
        args = {}
        # Backward compatibility in case of previous version retrieval
        try:
            args = self._neptune_run_kwargs
        except AttributeError:
            pass

        if self._project_name is not None:
            args["project"] = self._project_name

        if self._api_key is not None:
            args["api_token"] = self._api_key

        if self._run_short_id is not None:
            args["run"] = self._run_short_id

        # Backward compatibility in case of previous version retrieval
        try:
            if self._run_name is not None:
                args["name"] = self._run_name
        except AttributeError:
            pass

        return args

    def _construct_path_with_prefix(self, *keys) -> str:
        """Return sequence of keys joined by `LOGGER_JOIN_CHAR`, started with `_prefix` if defined."""
        if self._prefix:
            return self.LOGGER_JOIN_CHAR.join([self._prefix, *keys])
        return self.LOGGER_JOIN_CHAR.join(keys)

    @staticmethod
    def _verify_input_arguments(
        api_key: Optional[str],
        project: Optional[str],
        name: Optional[str],
        run: Optional["Run"],
        neptune_run_kwargs: dict,
    ):
        legacy_kwargs_msg = (
            "Following kwargs are deprecated: {legacy_kwargs}.\n"
            "If you are looking for the Neptune logger using legacy Python API,"
            " it's still available as part of neptune-contrib package:\n"
            "  - https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html\n"
            "The NeptuneLogger was re-written to use the neptune.new Python API\n"
            "  - https://neptune.ai/blog/neptune-new\n"
            "  - https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning\n"
            "You should use arguments accepted by either NeptuneLogger.init() or neptune.init()"
        )

        # check if user used legacy kwargs expected in `NeptuneLegacyLogger`
        used_legacy_kwargs = [
            legacy_kwarg for legacy_kwarg in neptune_run_kwargs if legacy_kwarg in _LEGACY_NEPTUNE_INIT_KWARGS
        ]
        if used_legacy_kwargs:
            raise ValueError(legacy_kwargs_msg.format(legacy_kwargs=used_legacy_kwargs))

        # check if user used legacy kwargs expected in `NeptuneLogger` from neptune-pytorch-lightning package
        used_legacy_neptune_kwargs = [
            legacy_kwarg for legacy_kwarg in neptune_run_kwargs if legacy_kwarg in _LEGACY_NEPTUNE_LOGGER_KWARGS
        ]
        if used_legacy_neptune_kwargs:
            raise ValueError(legacy_kwargs_msg.format(legacy_kwargs=used_legacy_neptune_kwargs))

        # check if user passed new client `Run` object
        if run is not None and not isinstance(run, Run):
            raise ValueError(
                "Run parameter expected to be of type `neptune.new.Run`.\n"
                "If you are looking for the Neptune logger using legacy Python API,"
                " it's still available as part of neptune-contrib package:\n"
                "  - https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html\n"
                "The NeptuneLogger was re-written to use the neptune.new Python API\n"
                "  - https://neptune.ai/blog/neptune-new\n"
                "  - https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning\n"
            )

        # check if user passed redundant neptune.init arguments when passed run
        any_neptune_init_arg_passed = any(arg is not None for arg in [api_key, project, name]) or neptune_run_kwargs
        if run is not None and any_neptune_init_arg_passed:
            raise ValueError(
                "When an already initialized run object is provided"
                " you can't provide other neptune.init() parameters.\n"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Run instance can't be pickled
        state["_run_instance"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._run_instance = neptune.init(**self._neptune_init_args)

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

        Note that syntax: ``self.logger.experiment["your/metadata/structure"].log(metadata)``
        is specific to Neptune and it extends logger capabilities.
        Specifically, it allows you to log various types of metadata like scores, files,
        images, interactive visuals, CSVs, etc. Refer to the
        `Neptune docs <https://docs.neptune.ai/you-should-know/logging-metadata#essential-logging-methods>`_
        for more detailed explanations.
        You can also use regular logger methods ``log_metrics()``, and ``log_hyperparams()``
        with NeptuneLogger as these are also supported.
        """
        return self.run

    @property
    @rank_zero_experiment
    def run(self) -> Run:
        try:
            if not self._run_instance:
                self._run_instance = neptune.init(**self._neptune_init_args)
                self._retrieve_run_data()
                # make sure that we've log integration version for newly created
                self._run_instance[_INTEGRATION_VERSION_KEY] = __version__

            return self._run_instance
        except NeptuneLegacyProjectException as e:
            raise TypeError(
                f"Project {self._project_name} has not been migrated to the new structure."
                " You can still integrate it with the Neptune logger using legacy Python API"
                " available as part of neptune-contrib package:"
                " https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html\n"
            ) from e

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:  # skipcq: PYL-W0221
        r"""
        Log hyper-parameters to the run.

        Hyperparams will be logged under the "<prefix>/hyperparams" namespace.

        Note:

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
                project="common/pytorch-lightning-integration"
            )

            neptune_logger.log_hyperparams(PARAMS)
        """
        params = _convert_params(params)
        params = _sanitize_callable_params(params)

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

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for key, val in metrics.items():
            # `step` is ignored because Neptune expects strictly increasing step values which
            # Lightning does not always guarantee.
            self.run[key].log(val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status:
            self.run[self._construct_path_with_prefix("status")] = status

        super().finalize(status)

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory of the experiment which in this case is ``None`` because Neptune does not save
        locally.

        Returns:
            the root directory where experiment logs get saved
        """
        return os.path.join(os.getcwd(), ".neptune")

    @rank_zero_only
    def log_model_summary(self, model, max_depth=-1):
        model_str = str(ModelSummary(model=model, max_depth=max_depth))
        self.run[self._construct_path_with_prefix("model/summary")] = neptune.types.File.from_content(
            content=model_str, extension="txt"
        )

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        """Automatically log checkpointed model. Called after model checkpoint callback saves a new checkpoint.

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
            self.run[f"{checkpoints_namespace}/{model_last_name}"].upload(checkpoint_callback.last_model_path)

        # save best k models
        for key in checkpoint_callback.best_k_models.keys():
            model_name = self._get_full_model_name(key, checkpoint_callback)
            file_names.add(model_name)
            self.run[f"{checkpoints_namespace}/{model_name}"].upload(key)

        # log best model path and checkpoint
        if checkpoint_callback.best_model_path:
            self.run[self._construct_path_with_prefix("model/best_model_path")] = checkpoint_callback.best_model_path

            model_name = self._get_full_model_name(checkpoint_callback.best_model_path, checkpoint_callback)
            file_names.add(model_name)
            self.run[f"{checkpoints_namespace}/{model_name}"].upload(checkpoint_callback.best_model_path)

        # remove old models logged to experiment if they are not part of best k models at this point
        if self.run.exists(checkpoints_namespace):
            exp_structure = self.run.get_structure()
            uploaded_model_names = self._get_full_model_names_from_exp_structure(exp_structure, checkpoints_namespace)

            for file_to_drop in list(uploaded_model_names - file_names):
                del self.run[f"{checkpoints_namespace}/{file_to_drop}"]

        # log best model score
        if checkpoint_callback.best_model_score:
            self.run[self._construct_path_with_prefix("model/best_model_score")] = (
                checkpoint_callback.best_model_score.cpu().detach().numpy()
            )

    @staticmethod
    def _get_full_model_name(model_path: str, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> str:
        """Returns model name which is string `model_path` appended to `checkpoint_callback.dirpath`."""
        expected_model_path = f"{checkpoint_callback.dirpath}{os.path.sep}"
        if not model_path.startswith(expected_model_path):
            raise ValueError(f"{model_path} was expected to start with {expected_model_path}.")
        # Remove extension from filepath
        filepath, _ = os.path.splitext(model_path[len(expected_model_path) :])

        return filepath

    @classmethod
    def _get_full_model_names_from_exp_structure(cls, exp_structure: dict, namespace: str) -> Set[str]:
        """Returns all paths to properties which were already logged in `namespace`"""
        structure_keys = namespace.split(cls.LOGGER_JOIN_CHAR)
        uploaded_models_dict = reduce(lambda d, k: d[k], [exp_structure, *structure_keys])
        return set(cls._dict_paths(uploaded_models_dict))

    @classmethod
    def _dict_paths(cls, d: dict, path_in_build: str = None) -> Generator:
        for k, v in d.items():
            path = f"{path_in_build}/{k}" if path_in_build is not None else k
            if not isinstance(v, dict):
                yield path
            else:
                yield from cls._dict_paths(v, path)

    @property
    def name(self) -> str:
        """Return the experiment name or 'offline-name' when exp is run in offline mode."""
        return self._run_name

    @property
    def version(self) -> str:
        """Return the experiment version.

        It's Neptune Run's short_id
        """
        return self._run_short_id

    @staticmethod
    def _signal_deprecated_api_usage(f_name, sample_code, raise_exception=False):
        msg_suffix = (
            f"If you are looking for the Neptune logger using legacy Python API,"
            f" it's still available as part of neptune-contrib package:\n"
            f"  - https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html\n"
            f"The NeptuneLogger was re-written to use the neptune.new Python API\n"
            f"  - https://neptune.ai/blog/neptune-new\n"
            f"  - https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning\n"
            f"Instead of `logger.{f_name}` you can use:\n"
            f"\t{sample_code}"
        )

        if not raise_exception:
            warnings.warn(
                "The function you've used is deprecated in v1.5.0 and will be removed in v1.7.0. " + msg_suffix
            )
        else:
            raise ValueError("The function you've used is deprecated.\n" + msg_suffix)

    @rank_zero_only
    def log_metric(self, metric_name: str, metric_value: Union[torch.Tensor, float, str], step: Optional[int] = None):
        key = f"{self._prefix}/{metric_name}"
        self._signal_deprecated_api_usage("log_metric", f"logger.run['{key}'].log(42)")
        if torch.is_tensor(metric_value):
            metric_value = metric_value.cpu().detach()

        self.run[key].log(metric_value, step=step)

    @rank_zero_only
    def log_text(self, log_name: str, text: str, step: Optional[int] = None) -> None:
        key = f"{self._prefix}/{log_name}"
        self._signal_deprecated_api_usage("log_text", f"logger.run['{key}].log('text')")
        self.run[key].log(str(text), step=step)

    @rank_zero_only
    def log_image(self, log_name: str, image: Union[str, Any], step: Optional[int] = None) -> None:
        key = f"{self._prefix}/{log_name}"
        self._signal_deprecated_api_usage("log_image", f"logger.run['{key}'].log(File('path_to_image'))")
        if isinstance(image, str):
            # if `img` is path to file, convert it to file object
            image = NeptuneFile(image)
        self.run[key].log(image, step=step)

    @rank_zero_only
    def log_artifact(self, artifact: str, destination: Optional[str] = None) -> None:
        key = f"{self._prefix}/{self.ARTIFACTS_KEY}/{artifact}"
        self._signal_deprecated_api_usage("log_artifact", f"logger.run['{key}].log('path_to_file')")
        self.run[key].log(destination)

    def set_property(self, *args, **kwargs):
        self._signal_deprecated_api_usage(
            "log_artifact", f"logger.run['{self._prefix}/{self.PARAMETERS_KEY}/key'].log(value)", raise_exception=True
        )

    def append_tags(self, *args, **kwargs):
        self._signal_deprecated_api_usage(
            "append_tags", "logger.run['sys/tags'].add(['foo', 'bar'])", raise_exception=True
        )
