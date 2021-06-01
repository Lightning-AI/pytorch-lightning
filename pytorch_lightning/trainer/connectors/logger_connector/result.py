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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric

from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import FxValidator
from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _METRIC


class MetricSource(LightningEnum):
    CALLBACK = "callback"
    PBAR = "pbar"
    LOG = "log"


@dataclass
class Metadata:
    fx: str
    name: str
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: Callable = torch.mean
    dataloader_idx: Optional[int] = None
    metric_attribute: Optional[str] = None
    has_reset: bool = False

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
        super().__init__(compute_on_step=is_tensor)
        self.is_tensor = is_tensor
        self.meta = metadata
        if is_tensor:
            self.add_state("value", torch.tensor(0, dtype=torch.float))
            if self.meta.is_mean_reduction:
                self.add_state("cumulated_batch_size", torch.tensor(0, dtype=torch.float))
        # FIXME: self.value when not tensor?

    def update(self, value: _METRIC, batch_size: Optional[int] = None) -> None:
        # FIXME: support for non-tensor. sync returns tensor always
        if self.is_tensor:
            if self.meta.is_mean_reduction:
                self.value += value.float().mean() * batch_size
                self.cumulated_batch_size += batch_size

            elif self.meta.is_max_reduction:
                self.value = max(self.value, value.float().mean())

            elif self.meta.is_min_reduction:
                self.value = min(self.value, value.float().mean())
        else:
            self.value = value  # noqa: attribute-defined-outside-init
            self._forward_cache = value._forward_cache

    def compute(self) -> torch.Tensor:
        if self.is_tensor:
            if self.meta.is_mean_reduction:
                return torch.sum(self.value) / torch.sum(self.cumulated_batch_size)
            elif self.meta.is_max_reduction or self.meta.is_min_reduction:
                return self.value
            raise MisconfigurationException(
                f"Only [min, max, mean] reductions are supported. Found {self.meta.reduce_fx}"
            )
        return self.value.compute()

    def reset(self) -> None:
        if self.is_tensor:
            super().reset()
        else:
            self.value.reset()
        self.meta.has_reset = True

    def forward(self, value: _METRIC, *args, **kwargs) -> torch.Tensor:
        """Overridden to avoid `self._forward_cache = None` after `update`"""
        prev_fwd_cache = getattr(value, '_forward_cache', None)
        out = super().forward(value, *args, **kwargs)
        if out is None:
            self._forward_cache = prev_fwd_cache
        return out

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

    def __init__(self, metadata: Metadata) -> None:
        super().__init__()
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
        self.batch_size: int = 1
        self.batch_idx: Optional[int] = None
        self.device: Optional[torch.device] = device
        self.fx_validator = FxValidator()

    @property
    def on_epoch_end_reached(self) -> bool:
        return self._on_epoch_end_reached

    @on_epoch_end_reached.setter
    def on_epoch_end_reached(self, on_epoch_end_reached):
        self._on_epoch_end_reached = on_epoch_end_reached
        self.batch_idx = None

    @property
    def metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """This function returns either batch or epoch metrics depending on ``on_epoch_end_reached``."""
        return self.get_epoch_metrics() if self.on_epoch_end_reached else self.get_batch_metrics()

    @property
    def minimize(self) -> Optional[Tensor]:
        """
        The :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` loss
        will be saved as the ``minimize`` attribute.
        """
        return self._minimize

    @minimize.setter
    def minimize(self, loss: Optional[torch.Tensor]) -> None:
        if loss is not None:
            if not isinstance(loss, Tensor):
                raise ValueError(f"`Result.minimize` must be a `torch.Tensor`, found: {loss}")
            if loss.grad_fn is None:
                raise RuntimeError("`Result.minimize` must have a `grad_fn`")
        self._minimize = loss

    @property
    def extra(self) -> Dict:
        """
        Extras are any keys other than the loss returned by
        :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`
        """
        return self.get('_extra', {})

    @extra.setter
    def extra(self, extra: Dict) -> None:

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
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = False,
        on_epoch: bool = True,
        reduce_fx: Callable = torch.mean,
        enable_graph: bool = False,
        dataloader_idx: Optional[int] = None,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
    ):
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # move metrics to cpu on TPU.
        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        if on_step and self.on_epoch_end_reached:
            raise RuntimeError(
                "Logging `on_step` when `on_epoch_end_reached` isn't allowed. This shouldn't have happened."
            )

        # storage key
        key = f"{fx}.{name}"
        # add dataloader_suffix to both key and fx
        if dataloader_idx is not None:
            # use as ResultCollection key
            key += f'.{dataloader_idx}'
            # used to decide when to reset
            fx += f'.{dataloader_idx}'

        if key not in self:
            # create metadata object if storage key doesn't exist in self
            meta = Metadata(
                fx=fx,
                name=name,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                dataloader_idx=dataloader_idx,
                metric_attribute=metric_attribute,
            )
            # create one ResultMetric object per value.
            # value can be provided as a nested collection.
            self.to_result_metric(key, meta, value)

        if self.should_reset_tensors(fx):
            # when restarting an new epoch, reset the tensors
            self._reset(fx, metrics=False)
        self.update_metrics(key, value, batch_size)
        self._current_fx = fx

    def to_result_metric(self, key: str, meta: Metadata, value: Union[Dict, torch.Tensor]) -> None:

        def fn(v: Union[torch.Tensor, Metric]) -> ResultMetric:
            metric = ResultMetric(meta, isinstance(v, torch.Tensor))
            return metric.to(self.device)

        if isinstance(value, dict):
            rmc = ResultMetricCollection(meta)
            rmc.update(value)
            apply_to_collection(rmc, (torch.Tensor, Metric), fn)
            value = rmc
        else:
            value = fn(value)

        self[key] = value

    def should_reset_tensors(self, fx: str) -> bool:
        # reset tensor metrics only when the hook changed and reloading the dataloader
        return self._current_fx != fx and self.batch_idx in (None, 0)

    def update_metrics(self, key: str, value: Union[Dict, torch.Tensor], batch_size: Optional[int]) -> None:
        batch_size = torch.tensor(batch_size or self.batch_size, device=self.device)

        def fn(result_metric, v):
            # call the forward function of ResultMetric
            result_metric(v.to(self.device), batch_size)
            result_metric.meta.has_reset = False

        apply_to_collections(self[key], value, ResultMetric, fn)

    @staticmethod
    def _get_forward_cache(result_metric: ResultMetric) -> Optional[torch.Tensor]:
        if result_metric.meta.on_step:
            return result_metric._forward_cache.detach()

    @staticmethod
    def _to_item(t: torch.Tensor) -> float:
        return t.item()

    def valid_items(self) -> Generator:
        """This function is used to iterate over current valid metrics."""
        return ((k, v) for k, v in self.items()
                if not k == "_extra" and not (isinstance(v, ResultMetric) and v.meta.has_reset))

    def _forked_name(self, result_metric: ResultMetric, on_step: bool) -> Tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step)
        dl_idx = result_metric.meta.dataloader_idx
        if dl_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dl_idx)
            name += dataloader_suffix
            forked_name += dataloader_suffix
        return name, forked_name

    def get_metrics(self, on_step: bool) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = {k: {} for k in MetricSource}

        # either extract `forward_cache` or `computed` from `ResultMetric` objects
        fn = self._get_forward_cache if on_step else self._get_computed_cache

        # iterate over all stored metrics.
        for key, result_metric in self.valid_items():

            # extract forward_cache or computed from the ResultMetric
            # ignore when the output of fn is None
            value = apply_to_collection(result_metric, ResultMetric, fn, include_none=False)

            # detect if the value is None. This can be nested.
            is_none = False

            def any_none(_):
                nonlocal is_none
                is_none = True

            apply_to_collection(value, type(None), any_none)
            if is_none:
                continue

            name, forked_name = self._forked_name(result_metric, on_step)

            # populate logging metrics
            if result_metric.meta.logger:
                metrics[MetricSource.LOG][forked_name] = value

            # populate callback metrics. callback metrics don't take `_step` forked metrics
            if self.training or result_metric.meta.on_epoch and not on_step:
                metrics[MetricSource.CALLBACK][name] = value
                metrics[MetricSource.CALLBACK][forked_name] = value

            # populate progress_bar metrics. values should be converted to float
            if result_metric.meta.prog_bar:
                value = apply_to_collection(value, torch.Tensor, self._to_item, include_none=False)
                metrics[MetricSource.PBAR][forked_name] = value

        return metrics

    def get_batch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.get_metrics(on_step=True)

    def get_epoch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.get_metrics(on_step=False)

    @staticmethod
    def _get_computed_cache(result_metric: ResultMetric) -> Optional[torch.Tensor]:
        if not result_metric.meta.on_epoch:
            return
        if not result_metric._computed:
            result_metric.compute()
        return result_metric._computed.detach()

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
