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
from collections.abc import Iterable, KeysView, Mapping
from typing import TYPE_CHECKING, Any, Optional, SupportsIndex, TypeVar, Union

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from typing_extensions import overload

import lightning.pytorch as pl
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE, _TENSORBOARDX_AVAILABLE
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.pytorch.loggers import CSVLogger, LitLogger, Logger, TensorBoardLogger
from lightning.pytorch.trainer.connectors.logger_connector.result import _METRICS, _OUT_DICT, _PBAR_DICT
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_info

warning_cache = WarningCache()

if TYPE_CHECKING:
    pass


class _LoggerConnector:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.trainer = trainer
        self._progress_bar_metrics: _PBAR_DICT = {}
        self._logged_metrics: _OUT_DICT = {}
        self._callback_metrics: _OUT_DICT = {}
        self._current_fx: Optional[str] = None
        # None: hasn't started, True: first loop iteration, False: subsequent iterations
        self._first_loop_iter: Optional[bool] = None

    def on_trainer_init(
        self,
        logger: Union[bool, Logger, Iterable[Logger], dict[str, Logger]],
        log_every_n_steps: int,
    ) -> None:
        self.configure_logger(logger)
        self.trainer.log_every_n_steps = log_every_n_steps

    @property
    def should_update_logs(self) -> bool:
        trainer = self.trainer
        if trainer.log_every_n_steps == 0:
            return False
        if (loop := trainer._active_loop) is None:
            return True
        if isinstance(loop, pl.loops._FitLoop):
            # `+ 1` because it can be checked before a step is executed, for example, in `on_train_batch_start`
            step = loop.epoch_loop._batches_that_stepped + 1
        elif isinstance(loop, (pl.loops._EvaluationLoop, pl.loops._PredictionLoop)):
            step = loop.batch_progress.current.ready
        else:
            raise NotImplementedError(loop)
        should_log = step % trainer.log_every_n_steps == 0
        return should_log or trainer.should_stop

    def configure_logger(self, logger: Union[bool, Logger, Iterable[Logger], dict[str, Logger]]) -> None:
        if not logger:
            # logger is None or logger is False
            self.trainer.loggers = []
        elif logger is True:
            # default logger
            if _TENSORBOARD_AVAILABLE or _TENSORBOARDX_AVAILABLE:
                logger_ = TensorBoardLogger(save_dir=self.trainer.default_root_dir, version=SLURMEnvironment.job_id())
            else:
                warning_cache.warn(
                    "Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch`"
                    " package, due to potential conflicts with other packages in the ML ecosystem. For this reason,"
                    " `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard`"
                    " or `tensorboardX` packages are found."
                    " Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default"
                )
                logger_ = CSVLogger(save_dir=self.trainer.default_root_dir)  # type: ignore[assignment]
            self.trainer.loggers = [logger_]
        elif isinstance(logger, (Mapping, Iterable)):
            self.trainer.loggers = logger
        else:
            self.trainer.loggers = [logger]

        if not any(isinstance(logger, LitLogger) for logger in self.trainer.loggers):
            rank_zero_info(
                "💡 Tip: For seamless cloud logging and experiment tracking,"
                " try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger,"
                " which logs metrics and artifacts automatically to the Lightning Experiments platform."
            )

    def log_metrics(self, metrics: _OUT_DICT, step: Optional[int] = None) -> None:
        """Logs the metric dict passed in. If `step` parameter is None and `step` key is presented is metrics, uses
        metrics["step"] as a step.

        Args:
            metrics: Metric values
            step: Step for which metrics should be logged. If a `step` metric is logged, this value will
                be used else will default to `self.global_step` during training or the total log step count
                during validation and testing.

        """
        if not self.trainer.loggers or not metrics:
            return

        self._logged_metrics.update(metrics)

        # turn all tensors to scalars
        scalar_metrics = convert_tensors_to_scalars(metrics)

        if step is None:
            step_metric = scalar_metrics.pop("step", None)
            if step_metric is not None:
                step = int(step_metric)
            else:
                # added metrics for convenience
                scalar_metrics.setdefault("epoch", self.trainer.current_epoch)
                step = self.trainer.fit_loop.epoch_loop._batches_that_stepped

        # log actual metrics
        for logger in self.trainer.loggers:
            logger.log_metrics(metrics=scalar_metrics, step=step)
            logger.save()

    """
    Evaluation metric updates
    """

    def _evaluation_epoch_end(self) -> None:
        results = self.trainer._results
        assert results is not None
        results.dataloader_idx = None

    def update_eval_step_metrics(self, step: int) -> None:
        assert isinstance(self._first_loop_iter, bool)
        # logs user requested information to logger
        self.log_metrics(self.metrics["log"], step=step)

    def update_eval_epoch_metrics(self) -> _OUT_DICT:
        assert self._first_loop_iter is None
        if self.trainer.sanity_checking:
            return {}
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics["pbar"])
        self._callback_metrics.update(metrics["callback"])
        self._logged_metrics.update(metrics["log"])
        return metrics["log"]

    def log_eval_end_metrics(self, metrics: _OUT_DICT) -> None:
        assert self._first_loop_iter is None
        if self.trainer.sanity_checking:
            return

        # log all the metrics as a single dict
        self.log_metrics(metrics)

    """
    Train metric updates
    """

    def update_train_step_metrics(self) -> None:
        if self.trainer.fit_loop._should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return

        # when metrics should be logged
        assert isinstance(self._first_loop_iter, bool)
        if self.should_update_logs or self.trainer.fast_dev_run:
            self.log_metrics(self.metrics["log"])

    def update_train_epoch_metrics(self) -> None:
        # add the metrics to the loggers
        assert self._first_loop_iter is None
        self.log_metrics(self.metrics["log"])

        # reset result collection for next epoch
        self.reset_results()

    """
    Utilities and properties
    """

    def on_batch_start(self, batch: Any, dataloader_idx: Optional[int] = None) -> None:
        if self._first_loop_iter is None:
            self._first_loop_iter = True
        elif self._first_loop_iter is True:
            self._first_loop_iter = False

        results = self.trainer._results
        assert results is not None
        # attach reference to the new batch and remove the cached batch_size
        results.batch = batch
        results.batch_size = None
        results.dataloader_idx = dataloader_idx

    def epoch_end_reached(self) -> None:
        self._first_loop_iter = None

    def on_epoch_end(self) -> None:
        assert self._first_loop_iter is None
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics["pbar"])
        self._callback_metrics.update(metrics["callback"])
        self._logged_metrics.update(metrics["log"])
        self._current_fx = None

    def on_batch_end(self) -> None:
        assert isinstance(self._first_loop_iter, bool)
        metrics = self.metrics
        self._progress_bar_metrics.update(metrics["pbar"])
        self._callback_metrics.update(metrics["callback"])
        self._logged_metrics.update(metrics["log"])

        assert self.trainer._results is not None
        # drop the reference to current batch and batch_size
        self.trainer._results.batch = None
        self.trainer._results.batch_size = None

    def should_reset_tensors(self, fx: str) -> bool:
        return self._current_fx != fx and self._first_loop_iter in (None, True)

    def reset_metrics(self) -> None:
        self._progress_bar_metrics = {}
        self._logged_metrics = {}
        self._callback_metrics = {}

    def reset_results(self) -> None:
        results = self.trainer._results
        if results is not None:
            results.reset()

        self._first_loop_iter = None
        self._current_fx = None

    @property
    def metrics(self) -> _METRICS:
        """This function returns either batch or epoch metrics."""
        on_step = self._first_loop_iter is not None
        assert self.trainer._results is not None
        return self.trainer._results.metrics(on_step)

    @property
    def callback_metrics(self) -> _OUT_DICT:
        if self.trainer._results:
            metrics = self.metrics["callback"]
            self._callback_metrics.update(metrics)
        return self._callback_metrics

    @property
    def logged_metrics(self) -> _OUT_DICT:
        if self.trainer._results:
            metrics = self.metrics["log"]
            self._logged_metrics.update(metrics)
        return self._logged_metrics

    @property
    def progress_bar_metrics(self) -> _PBAR_DICT:
        if self.trainer._results:
            metrics = self.metrics["pbar"]
            self._progress_bar_metrics.update(metrics)
        return self._progress_bar_metrics

    def teardown(self) -> None:
        args = (Tensor, move_data_to_device, "cpu")
        self._logged_metrics = apply_to_collection(self._logged_metrics, *args)
        self._progress_bar_metrics = apply_to_collection(self._progress_bar_metrics, *args)
        self._callback_metrics = apply_to_collection(self._callback_metrics, *args)


_T = TypeVar("_T")
_PT = TypeVar("_PT")


class _ListMap(list[_T]):
    """A hybrid container allowing both index and name access.

    This class extends the built-in list to provide dictionary-like access to its elements
    using string keys. It maintains an internal mapping of string keys to list indices,
    allowing users to retrieve elements by their associated names.

    Args:
        __iterable (Union[Iterable[_T], Mapping[str, _T]], optional): An iterable of objects or a mapping
            of string keys to __iterable to initialize the container.

    Raises:
        TypeError: If a Mapping is provided and any of its keys are not of type str.

    Example:
        >>> listmap = _ListMap({'obj1': 1, 'obj2': 2})
        >>> listmap['obj1']  # Access by name
        1
        >>> listmap[0]  # Access by index
        1
        >>> listmap.append(4)  # Append by index
        >>> listmap[2]
        4

    """

    _dict_map: dict[str, int]

    def __init__(self, __iterable: Optional[Union[dict[str, _T], Iterable[_T]]] = None):
        if isinstance(__iterable, Mapping):
            # super inits list with values
            if any(not isinstance(x, str) for x in __iterable):
                raise TypeError("When providing a Mapping, all keys must be of type str.")
            super().__init__(__iterable.values())
            self._dict_map = {key: idx for idx, key in enumerate(__iterable)}
        else:
            super().__init__(__iterable or ())
            self._dict_map = {}

    def __eq__(self, other: Any) -> bool:
        list_eq = super().__eq__(other)
        if isinstance(other, _ListMap):
            list_eq &= other._dict_map == self._dict_map
        return list_eq

    def pop(self, index: SupportsIndex = -1, /):
        if self._dict_map:
            index_int = index.__index__()
            if index_int < 0:
                index_int += len(self)

            for key, idx in list(self._dict_map.items()):
                if idx == index_int:
                    self._dict_map.pop(key)
                elif idx > index_int:
                    self._dict_map[key] -= 1
        return super().pop(index)

    def insert(self, index: SupportsIndex, __object: _T, /) -> None:
        if self._dict_map:
            index_int = index.__index__()
            if index_int < 0:
                index_int += len(self)

            for key, idx in list(self._dict_map.items()):
                if idx >= index_int:
                    self._dict_map[key] += 1

        return super().insert(index, __object)

    @overload
    def __getitem__(self, key: Union[SupportsIndex, str], /) -> _T: ...

    @overload
    def __getitem__(self, key: slice, /) -> list[_T]: ...

    def __getitem__(self, key: Union[SupportsIndex, str, slice], /) -> Union[_T, list[_T]]:
        if self._dict_map and isinstance(key, str):
            return self[self._dict_map[key]]
        return super().__getitem__(key)

    def __contains__(self, item: Union[object, str]) -> bool:
        if isinstance(item, str):
            return item in self._dict_map
        return super().__contains__(item)

    def __repr__(self) -> str:
        ret = super().__repr__()
        return f"{type(self).__name__}({ret}, keys={list(self._dict_map.keys())})"

    def clear(self) -> None:
        self._dict_map.clear()
        return super().clear()

    def keys(self) -> KeysView[str]:
        return self._dict_map.keys()
