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
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torchmetrics import Metric

from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import FxValidator
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException

# re-define the ones from pytorch_lightning.utilities.types without the `Number` type
_METRIC = Union[Metric, torch.Tensor]
_METRIC_COLLECTION = Union[_METRIC, Dict[str, _METRIC]]


class MetricSource(LightningEnum):
    CALLBACK = "callback"
    PBAR = "pbar"
    LOG = "log"


@dataclass
class Sync:
    fn: Callable
    should: bool = False
    op: Union[Any, str] = 'mean'
    group: Optional[Any] = None

    @property
    def __call__(self) -> Callable:
        return partial(self.fn, reduce_op=self.op, group=self.group) if self.should else self.no_op

    @staticmethod
    def no_op(value: Any, *_, **__) -> Any:
        return value


@dataclass
class Metadata:
    fx: str
    name: str
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: Callable = torch.mean
    enable_graph: bool = False
    dataloader_idx: Optional[int] = None
    metric_attribute: Optional[str] = None
    sync: Sync = field(default_factory=Sync)

    @property
    def forked(self) -> bool:
        return self.on_step and self.on_epoch

    def forked_name(self, on_step: bool) -> str:
        if self.forked:
            return f'{self.name}_{"step" if on_step else "epoch"}'
        return self.name

    @property
    def is_mean_reduction(self) -> bool:
        return self.reduce_fx == torch.mean

    @property
    def is_max_reduction(self) -> bool:
        return self.reduce_fx in (torch.max, max)

    @property
    def is_min_reduction(self) -> bool:
        return self.reduce_fx in (torch.min, min)


class ResultMetric(Metric, DeviceDtypeModuleMixin):
    """Wraps the value provided to `:meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""

    def __init__(self, metadata: Metadata, is_tensor: bool) -> None:
        super().__init__()
        self.is_tensor = is_tensor
        self.meta = metadata
        self.has_reset = False
        if is_tensor:
            self.add_state("value", torch.tensor(0, dtype=torch.float))
            if self.meta.is_mean_reduction:
                self.add_state("cumulated_batch_size", torch.tensor(0, dtype=torch.float))

    def __setattr__(self, key, value):
        # performance: skip the `torch.nn.Module.__setattr__` checks
        object.__setattr__(self, key, value)

    def update(self, value: _METRIC, batch_size: torch.Tensor) -> None:
        if self.is_tensor:
            # performance: no need to accumulate on values only logged on_step
            if self.meta.on_step and not self.meta.on_epoch:
                self.value = self.meta.sync(value)
                return
            # perform accumulation with reduction
            if self.meta.is_mean_reduction:
                self.value += value.mean() * batch_size
                self.cumulated_batch_size += batch_size
            elif self.meta.is_max_reduction:
                self.value = max(self.value, value.mean())
            elif self.meta.is_min_reduction:
                self.value = min(self.value, value.mean())
        else:
            self.value = value  # noqa: attribute-defined-outside-init
            self._forward_cache = value._forward_cache

    def compute(self) -> torch.Tensor:
        if self.is_tensor:
            value = self.meta.sync(self.value)
            if self.meta.is_mean_reduction:
                cumulated_batch_size = self.meta.sync(self.cumulated_batch_size)
                # FIXME: might need sum
                return value / cumulated_batch_size
            elif self.meta.is_max_reduction or self.meta.is_min_reduction:
                return value
            raise MisconfigurationException(
                f"Only [min, max, mean] reductions are supported. Found {self.meta.reduce_fx}"
            )
        return self.value.compute()

    def _wrap_compute(self, compute: Any) -> Any:
        # Override to avoid syncing - we handle it ourselves.
        @wraps(compute)
        def wrapped_func(*args, **kwargs):
            if not self._update_called:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.", UserWarning
                )

            # return cached value
            if self._computed is not None:
                return self._computed
            self._computed = compute(*args, **kwargs)
            return self._computed

        return wrapped_func

    def reset(self) -> None:
        if self.is_tensor:
            super().reset()
        else:
            self.value.reset()
        self.has_reset = True

    def forward(self, value: _METRIC, batch_size: torch.Tensor) -> None:
        if self.is_tensor:
            value = value.float()
            self._forward_cache = value
        else:
            self._forward_cache = value._forward_cache

        if self.meta.enable_graph:
            with torch.no_grad():
                self.update(value, batch_size)
        else:
            # performance: skip the `torch.no_grad` context manager by calling `update` directly
            self.update(value, batch_size)

    def __repr__(self) -> str:
        state = f"value={self.value}"
        if self.is_tensor and self.meta.is_mean_reduction:
            state += f", cumulated_batch_size={self.cumulated_batch_size}"
        return f"{self.__class__.__name__}({state})"


class ResultMetricCollection(dict):
    """
    Dict wrapper for easy access to metadata.

    All of the leaf items should be instances of
    :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetric`
    with the same metadata.
    """

    def __init__(self, *args, metadata: Optional[Metadata] = None) -> None:
        super().__init__(*args)
        self.meta = metadata


class ResultCollection(dict):
    """
    Collection (dictionary) of :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetric` or
    :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetricCollection`

    Example:

        # `device` needs to be provided before logging
        result = ResultCollection(True, torch.device("cpu"))

        # you can log to a specific collection.
        # arguments: fx, key, value, metadata
        result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)
        result.log('validation_step', 'recall', torch.tensor(...), on_step=True, on_epoch=True)

        for epoch in epochs:
            for batch_idx, batch in enumerate(dataloader):
                # the batch_idx is used to reset the tensor metrics
                result.batch_idx = batch_idx
                result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)

            result.on_epoch_end_reached = True  # indicate epoch end has been reached
            result.log('training_epoch_end', 'acc', torch.tensor(...), on_step=False, on_epoch=True)`
    """

    DATALOADER_SUFFIX = "/dataloader_idx_{}"

    def __init__(self, training: bool, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.training = training
        self._on_epoch_end_reached = False
        self._minimize = None
        self._current_fx: Optional[str] = None
        self._batch_size = torch.tensor(1, device=device)
        self.batch_idx: Optional[int] = None
        self.device: Optional[torch.device] = device
        self.fx_validator = FxValidator()

    @property
    def batch_size(self) -> torch.Tensor:
        # performance: cache the `batch_size` tensor instead of re-creating it
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = torch.tensor(value, device=self.device)

    @property
    def on_epoch_end_reached(self) -> bool:
        return self._on_epoch_end_reached

    @on_epoch_end_reached.setter
    def on_epoch_end_reached(self, on_epoch_end_reached):
        self._on_epoch_end_reached = on_epoch_end_reached
        self.batch_idx = None

    @property
    def metrics(self) -> Dict[str, _METRIC_COLLECTION]:
        """This function returns either batch or epoch metrics depending on ``on_epoch_end_reached``."""
        return self.get_epoch_metrics() if self.on_epoch_end_reached else self.get_batch_metrics()

    @property
    def minimize(self) -> Optional[torch.Tensor]:
        """
        The :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` loss
        will be saved as the ``minimize`` attribute.
        """
        return self._minimize

    @minimize.setter
    def minimize(self, loss: Optional[torch.Tensor]) -> None:
        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                raise ValueError(f"`Result.minimize` must be a `torch.Tensor`, found: {loss}")
            if loss.grad_fn is None:
                raise RuntimeError("`Result.minimize` must have a `grad_fn`")
        self._minimize = loss

    @property
    def extra(self) -> Dict[str, Any]:
        """
        Extras are any keys other than the loss returned by
        :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`
        """
        return self.get('_extra', {})

    @extra.setter
    def extra(self, extra: Dict[str, Any]) -> None:

        def check_fn(v):
            if v.grad_fn is not None:
                raise MisconfigurationException(
                    'You passed a tensor with `grad_fn` when calling `self.log()`.'
                    f' The extra values are {extra}'
                )

        apply_to_collection(extra, torch.Tensor, check_fn)
        self['_extra'] = extra

    def log(
        self,
        fx: str,
        name: str,
        value: _METRIC_COLLECTION,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = False,
        on_epoch: bool = True,
        reduce_fx: Callable = torch.mean,
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_fn: Callable = Sync.no_op,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        dataloader_idx: Optional[int] = None,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
    ) -> None:
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # move metrics to cpu on TPU.
        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        if on_step and self.on_epoch_end_reached:
            # `FxValidator` should avoid this ever happening. Either a bug there or a bug in the logic order.
            raise RuntimeError(
                "Logging `on_step` when `on_epoch_end_reached` isn't allowed. This shouldn't have happened."
            )

        # storage key
        key = f"{fx}.{name}"
        # add dataloader_suffix to both key and fx
        if dataloader_idx is not None:
            key += f'.{dataloader_idx}'
            fx += f'.{dataloader_idx}'

        meta = Metadata(
            fx=fx,
            name=name,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            dataloader_idx=dataloader_idx,
            metric_attribute=metric_attribute,
            sync=Sync(
                should=sync_dist,
                fn=sync_dist_fn,
                op=sync_dist_op,
                group=sync_dist_group,
            )
        )
        if key not in self:
            self.register_key(key, meta, value)
        elif meta != self[key].meta:
            raise MisconfigurationException(
                f'You called `self.log({key}, ...)` twice with different arguments. This is not allowed'
            )

        if self.should_reset_tensors(fx):
            # when restarting an new epoch, reset the tensors
            self._reset(fx, metrics=False)

        if batch_size is not None:
            self.batch_size = batch_size

        self.update_metrics(key, value)
        self._current_fx = fx

    def register_key(self, key: str, meta: Metadata, value: _METRIC_COLLECTION) -> None:
        """Create one ResultMetric object per value. Value can be provided as a nested collection"""

        def fn(v: _METRIC) -> ResultMetric:
            metric = ResultMetric(meta, isinstance(v, torch.Tensor))
            return metric.to(self.device)

        value = apply_to_collection(value, (torch.Tensor, Metric), fn)
        if isinstance(value, dict):
            value = ResultMetricCollection(value, metadata=meta)
        self[key] = value

    def should_reset_tensors(self, fx: str) -> bool:
        # reset tensor metrics only when the hook changed and reloading the dataloader
        return self._current_fx != fx and self.batch_idx in (None, 0)

    def update_metrics(self, key: str, value: _METRIC_COLLECTION) -> None:

        def fn(result_metric, v):
            # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
            result_metric.forward(v.to(self.device), self.batch_size)
            result_metric.has_reset = False

        apply_to_collections(self[key], value, ResultMetric, fn)

    @staticmethod
    def _get_cache(on_step: bool, result_metric: ResultMetric) -> Optional[torch.Tensor]:
        cache = None
        if on_step and result_metric.meta.on_step:
            cache = result_metric._forward_cache
        elif not on_step and result_metric.meta.on_epoch:
            if not result_metric._computed:
                result_metric.compute()
            cache = result_metric._computed
        if cache is not None and not result_metric.meta.enable_graph:
            return cache.detach()
        return cache

    @staticmethod
    def _to_item(t: torch.Tensor) -> float:
        return t.item()

    def valid_items(self) -> Generator:
        """This function is used to iterate over current valid metrics."""
        return ((k, v) for k, v in self.items()
                if not k == "_extra" and not (isinstance(v, ResultMetric) and v.has_reset))

    def _forked_name(self, result_metric: ResultMetric, on_step: bool) -> Tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step)
        dl_idx = result_metric.meta.dataloader_idx
        if dl_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dl_idx)
            name += dataloader_suffix
            forked_name += dataloader_suffix
        return name, forked_name

    def get_metrics(self, on_step: bool) -> Dict[str, _METRIC_COLLECTION]:
        metrics = {k: {} for k in MetricSource}

        for key, result_metric in self.valid_items():

            # extract forward_cache or computed from the ResultMetric. ignore when the output is None
            value = apply_to_collection(
                result_metric, ResultMetric, partial(self._get_cache, on_step), include_none=False
            )

            # check if the collection is empty
            has_tensor = False

            def any_tensor(_):
                nonlocal has_tensor
                has_tensor = True

            apply_to_collection(value, torch.Tensor, any_tensor)
            if not has_tensor:
                continue

            name, forked_name = self._forked_name(result_metric, on_step)

            # populate logging metrics
            if result_metric.meta.logger:
                metrics[MetricSource.LOG][forked_name] = value

            # populate callback metrics. callback metrics don't take `_step` forked metrics
            if self.training or result_metric.meta.on_epoch and not on_step:
                metrics[MetricSource.CALLBACK][name] = value
                metrics[MetricSource.CALLBACK][forked_name] = value

            # populate progress_bar metrics. convert tensors to numbers
            if result_metric.meta.prog_bar:
                value = apply_to_collection(value, torch.Tensor, self._to_item, include_none=False)
                metrics[MetricSource.PBAR][forked_name] = value

        return metrics

    def get_batch_metrics(self) -> Dict[str, _METRIC_COLLECTION]:
        return self.get_metrics(on_step=True)

    def get_epoch_metrics(self) -> Dict[str, _METRIC_COLLECTION]:
        return self.get_metrics(on_step=False)

    def to(self, *args, **kwargs) -> 'ResultCollection':
        """Move all data to the given device."""
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, Metric)):
                self[k] = v.to(*args, **kwargs)
        return self

    def cpu(self) -> 'ResultCollection':
        """Move all data to CPU."""
        return self.to(device="cpu")

    def _reset(self, fx: Optional[str] = None, metrics: Optional[bool] = None) -> None:

        def fn(item: ResultMetric) -> None:
            requested_type = metrics is None or metrics ^ item.is_tensor
            same_fx = fx is None or fx == item.meta.fx
            if requested_type and same_fx:
                item.reset()

        apply_to_collection(self, ResultMetric, fn)

    def reset(self, metrics: Optional[bool] = None) -> None:
        """
        Reset the result collection

        Args:
            metrics: If True, only ``torchmetrics.Metric`` results are reset,
                if False, only ``torch.Tensors`` are reset,
                if ``None``, both are.
        """
        self._reset(metrics=metrics)
        self.on_epoch_end_reached = False
        self._current_fx = None

    def extract_batch_size(self, batch: Any) -> None:
        try:
            self.batch_size = self._extract_batch_size(batch)
        except RecursionError:
            self.batch_size = 1

    def _extract_batch_size(self, batch: Any) -> int:
        """
        Recursively unpack a batch to find a torch.Tensor.

        Returns:
            ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.
        """
        if isinstance(batch, torch.Tensor):
            size = batch.size(0)
        elif isinstance(batch, str):
            return len(batch)
        elif isinstance(batch, dict):
            sample = next(iter(batch.values()), 1)
            size = self._extract_batch_size(sample)
        elif isinstance(batch, Iterable):
            sample = next(iter(batch), 1)
            size = self._extract_batch_size(sample)
        else:
            size = 1
        return size

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.training}, {self.device}, {repr(self)})'

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        # can't deepcopy tensors with grad_fn
        minimize = d.get('_minimize')
        if minimize is not None:
            d['_minimize'] = minimize.detach()
        return d
