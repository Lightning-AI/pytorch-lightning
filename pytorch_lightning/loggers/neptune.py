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
from argparse import Namespace
from typing import Any, Dict, Iterable, Optional, Union

import torch
from torch import is_tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only

_NEPTUNE_AVAILABLE = _module_available("neptune")

if _NEPTUNE_AVAILABLE:
    import neptune
    from neptune.experiments import Experiment
else:
    # needed for test mocks, these tests shall be updated
    neptune, Experiment = None, None


class NeptuneLogger(LightningLoggerBase):
    r"""
    Log using `Neptune <https://neptune.ai>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune-client

    The Neptune logger can be used in the online mode or offline (silent) mode.
    To log experiment data in online mode, :class:`NeptuneLogger` requires an API key.
    In offline mode, the logger does not connect to Neptune.

    **ONLINE MODE**

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
        # We are using an api_key for the anonymous user "neptuner" but you can use your own.
        neptune_logger = NeptuneLogger(
            api_key='ANONYMOUS',
            project_name='shared/pytorch-lightning-integration',
            experiment_name='default',  # Optional,
            params={'max_epochs': 10},  # Optional,
            tags=['pytorch-lightning', 'mlp']  # Optional,
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    **OFFLINE MODE**

    .. testcode::

        from pytorch_lightning.loggers import NeptuneLogger

        # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
        neptune_logger = NeptuneLogger(
            offline_mode=True,
            project_name='USER_NAME/PROJECT_NAME',
            experiment_name='default',  # Optional,
            params={'max_epochs': 10},  # Optional,
            tags=['pytorch-lightning', 'mlp']  # Optional,
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    Use the logger anywhere in you :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                self.logger.experiment.log_metric('acc_train', ...)
                # log images
                self.logger.experiment.log_image('worse_predictions', ...)
                # log model checkpoint
                self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
                self.logger.experiment.whatever_neptune_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.log_metric('acc_train', ...)
                self.logger.experiment.log_image('worse_predictions', ...)
                self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
                self.logger.experiment.whatever_neptune_supports(...)

    If you want to log objects after the training is finished use ``close_after_fit=False``:

    .. code-block:: python

        neptune_logger = NeptuneLogger(
            ...
            close_after_fit=False,
            ...
        )
        trainer = Trainer(logger=neptune_logger)
        trainer.fit()

        # Log test metrics
        trainer.test(model)

        # Log additional metrics
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_true, y_pred)
        neptune_logger.experiment.log_metric('test_accuracy', accuracy)

        # Log charts
        from scikitplot.metrics import plot_confusion_matrix
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        neptune_logger.experiment.log_image('confusion_matrix', fig)

        # Save checkpoints folder
        neptune_logger.experiment.log_artifact('my/checkpoints')

        # When you are done, stop the experiment
        neptune_logger.experiment.stop()

    See Also:
        - An `Example experiment <https://ui.neptune.ai/o/shared/org/
          pytorch-lightning-integration/e/PYTOR-66/charts>`_ showing the UI of Neptune.
        - `Tutorial <https://docs.neptune.ai/integrations/pytorch_lightning.html>`_ on how to use
          Pytorch Lightning with Neptune.

    Args:
        api_key: Required in online mode.
            Neptune API token, found on https://neptune.ai.
            Read how to get your
            `API key <https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token>`_.
            It is recommended to keep it in the `NEPTUNE_API_TOKEN`
            environment variable and then you can leave ``api_key=None``.
        project_name: Required in online mode. Qualified name of a project in a form of
            "namespace/project_name" for example "tom/minst-classification".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
        offline_mode: Optional default ``False``. If ``True`` no logs will be sent
            to Neptune. Usually used for debug purposes.
        close_after_fit: Optional default ``True``. If ``False`` the experiment
            will not be closed after training and additional metrics,
            images or artifacts can be logged. Also, remember to close the experiment explicitly
            by running ``neptune_logger.experiment.stop()``.
        experiment_name: Optional. Editable name of the experiment.
            Name is displayed in the experiment’s Details (Metadata section) and
            in experiments view as a column.
        experiment_id: Optional. Default is ``None``. The ID of the existing experiment.
            If specified, connect to experiment with experiment_id in project_name.
            Input arguments "experiment_name", "params", "properties" and "tags" will be overriden based
            on fetched experiment data.
        prefix: A string to put at the beginning of metric keys.
        \**kwargs: Additional arguments like `params`, `tags`, `properties`, etc. used by
            :func:`neptune.Session.create_experiment` can be passed as keyword arguments in this logger.
    """

    LOGGER_JOIN_CHAR = '-'

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        close_after_fit: Optional[bool] = True,
        offline_mode: bool = False,
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        prefix: str = '',
        **kwargs
    ):
        if neptune is None:
            raise ImportError(
                'You want to use `neptune` logger which is not installed yet,'
                ' install it with `pip install neptune-client`.'
            )
        super().__init__()
        self.api_key = api_key
        self.project_name = project_name
        self.offline_mode = offline_mode
        self.close_after_fit = close_after_fit
        self.experiment_name = experiment_name
        self._prefix = prefix
        self._kwargs = kwargs
        self.experiment_id = experiment_id
        self._experiment = None

        log.info(f'NeptuneLogger will work in {"offline" if self.offline_mode else "online"} mode')

    def __getstate__(self):
        state = self.__dict__.copy()

        # Experiment cannot be pickled, and additionally its ID cannot be pickled in offline mode
        state['_experiment'] = None
        if self.offline_mode:
            state['experiment_id'] = None

        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Experiment:
        r"""
        Actual Neptune object. To use neptune features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_neptune_function()

        """

        # Note that even though we initialize self._experiment in __init__,
        # it may still end up being None after being pickled and un-pickled
        if self._experiment is None:
            self._experiment = self._create_or_get_experiment()

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, val in params.items():
            self.experiment.set_property(f'param__{key}', val)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        """
        Log metrics (numeric values) in Neptune experiments.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded, currently ignored
        """
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metrics = self._add_prefix(metrics)
        for key, val in metrics.items():
            # `step` is ignored because Neptune expects strictly increasing step values which
            # Lighting does not always guarantee.
            self.log_metric(key, val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if self.close_after_fit:
            self.experiment.stop()

    @property
    def save_dir(self) -> Optional[str]:
        # Neptune does not save any local files
        return None

    @property
    def name(self) -> str:
        if self.offline_mode:
            return 'offline-name'
        else:
            return self.experiment.name

    @property
    def version(self) -> str:
        if self.offline_mode:
            return 'offline-id-1234'
        else:
            return self.experiment.id

    @rank_zero_only
    def log_metric(
        self, metric_name: str, metric_value: Union[torch.Tensor, float, str], step: Optional[int] = None
    ) -> None:
        """
        Log metrics (numeric values) in Neptune experiments.

        Args:
            metric_name: The name of log, i.e. mse, loss, accuracy.
            metric_value: The value of the log (data-point).
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        if is_tensor(metric_value):
            metric_value = metric_value.cpu().detach()

        if step is None:
            self.experiment.log_metric(metric_name, metric_value)
        else:
            self.experiment.log_metric(metric_name, x=step, y=metric_value)

    @rank_zero_only
    def log_text(self, log_name: str, text: str, step: Optional[int] = None) -> None:
        """
        Log text data in Neptune experiments.

        Args:
            log_name: The name of log, i.e. mse, my_text_data, timing_info.
            text: The value of the log (data-point).
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        self.experiment.log_text(log_name, text, step=step)

    @rank_zero_only
    def log_image(self, log_name: str, image: Union[str, Any], step: Optional[int] = None) -> None:
        """
        Log image data in Neptune experiment

        Args:
            log_name: The name of log, i.e. bboxes, visualisations, sample_images.
            image: The value of the log (data-point).
                Can be one of the following types: PIL image, `matplotlib.figure.Figure`,
                path to image file (str)
            step: Step number at which the metrics should be recorded, must be strictly increasing
        """
        if step is None:
            self.experiment.log_image(log_name, image)
        else:
            self.experiment.log_image(log_name, x=step, y=image)

    @rank_zero_only
    def log_artifact(self, artifact: str, destination: Optional[str] = None) -> None:
        """Save an artifact (file) in Neptune experiment storage.

        Args:
            artifact: A path to the file in local filesystem.
            destination: Optional. Default is ``None``. A destination path.
                If ``None`` is passed, an artifact file name will be used.
        """
        self.experiment.log_artifact(artifact, destination)

    @rank_zero_only
    def set_property(self, key: str, value: Any) -> None:
        """
        Set key-value pair as Neptune experiment property.

        Args:
            key: Property key.
            value: New value of a property.
        """
        self.experiment.set_property(key, value)

    @rank_zero_only
    def append_tags(self, tags: Union[str, Iterable[str]]) -> None:
        """
        Appends tags to the neptune experiment.

        Args:
            tags: Tags to add to the current experiment. If str is passed, a single tag is added.
                If multiple - comma separated - str are passed, all of them are added as tags.
                If list of str is passed, all elements of the list are added as tags.
        """
        if str(tags) == tags:
            tags = [tags]  # make it as an iterable is if it is not yet
        self.experiment.append_tags(*tags)

    def _create_or_get_experiment(self):
        if self.offline_mode:
            project = neptune.Session(backend=neptune.OfflineBackend()).get_project('dry-run/project')
        else:
            session = neptune.Session.with_default_backend(api_token=self.api_key)
            project = session.get_project(self.project_name)

        if self.experiment_id is None:
            exp = project.create_experiment(name=self.experiment_name, **self._kwargs)
            self.experiment_id = exp.id
        else:
            exp = project.get_experiments(id=self.experiment_id)[0]
            self.experiment_name = exp.get_system_properties()['name']
            self.params = exp.get_parameters()
            self.properties = exp.get_properties()
            self.tags = exp.get_tags()

        return exp
