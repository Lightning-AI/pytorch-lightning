from argparse import Namespace
from typing import Any, AnyStr, Dict, List, Literal, Optional, Union

import pandas as pd
from torch import Tensor

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.imports import _module_available
from pytorch_lightning.utilities.rank_zero import rank_zero_only

_CLEARML_AVAILABLE = _module_available("clearml")

try:
    from clearml import Task
except ModuleNotFoundError:
    _CLEARML_AVAILABLE = False


class ClearMLLogger(Logger):
    """Log using `ClearML <https://clear.ml>`_.

    Install it with pip:

    .. code-block:: bash

        pip install clearml

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import ClearMLLogger

        cml_logger = ClearMLLogger(project_name="lightning-project", task_name="task-name")
        trainer = Trainer(logger=cml_logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.module.LightningModule` as follows:

    .. code-block:: python

        from pytorch_lightning import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_clear_ml_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_clear_ml_supports(...)

    Args:
        project_name: Name of the ClearML project
        task_name: Name of the ClearML task
        task_id: Optional ID of an existing ClearML task to be reused.

    Raises:
        ModuleNotFoundError:
            If required ClearML package is not installed on the device.
    """

    def __init__(self, project_name: str, task_name: str, task_id: str = None):
        super().__init__()
        self.project_name = project_name
        self.task_name = task_name
        self.id = task_id
        self._step = 0

        if not self.id:
            self._initialized = True
            self.task = Task.init(project_name=self.project_name, task_name=self.task_name)
        else:
            self._initialized = True
            self.task = Task.get_task(task_id=self.id)

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, Tensor]],
        step: Optional[int] = None,
    ) -> None:
        """
        Records metrics.
        This method logs metrics as as soon as it received them.
        If you want to aggregate metrics for one specific `step`, use the
        :meth:`~pytorch_lightning.loggers.base.Logger.agg_and_log_metrics`
        method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        if step is None:
            step = self._step

        def _handle_value(value: Union[float, Tensor]):
            if isinstance(value, Tensor):
                return value.item()
            return value

        for metric, value in metrics.items():
            self.task.logger.report_scalar(
                title=metric,
                series=metric,
                value=_handle_value(value),
                iteration=step,
            )

        self._step += 1

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, AnyStr], Namespace], *args: Any, **kwargs: Any) -> None:
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used
        """
        self.task.connect(params, *args, *kwargs)

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: List[str] = None,
        data: List[List[Any]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Record a table.

        Args:
            key: Unique identifier of the table
            columns: List of column names to be assigned to the table
            data: List of lists representing a table
            dataframe: `pandas.DataFrame` object representing a table
            step: Step number at which the metrics should be recorded
        """

        table: Optional[Union[pd.DataFrame, List[List[Any]]]] = None

        if dataframe is not None:
            table = dataframe
            if columns is not None:
                table.columns = columns

        if data is not None:
            table = data
            assert len(columns) == len(table[0]), "number of column names should match the total number of columns"
            table.insert(0, columns)

        if table is not None:
            self.task.logger.report_table(title=key, series=key, iteration=step, table_plot=table)

    @rank_zero_only
    def finalize(self, status: Literal["success", "failed", "aborted"] = "sucess") -> None:
        """Finalize the experiment. Mark the task completed or otherwise given the status.

        Args:
            status: Status that the experiment finished with
                (e.g. success, failed, aborted)
        """

        if status == "success":
            self.task.mark_completed()
        elif status == "failed":
            self.task.mark_failed()
        elif status == "aborted":
            self.task.mark_stopped()

    @property
    def name(self) -> Optional[str]:
        """Gets the name of the experiment, being the name of the ClearML task.

        Returns:
            The name of the ClearML task
        """
        return self.task.name

    @property
    def version(self) -> str:
        """Gets the version of the experiment, being the ID of the ClearML task.

        Returns:
            The id of the ClearML task
        """
        return self.task.id
