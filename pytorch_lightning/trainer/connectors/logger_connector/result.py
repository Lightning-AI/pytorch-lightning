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
from dataclasses import asdict, dataclass, replace
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torchmetrics import Metric

from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections, move_data_to_device
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.metrics import metrics_to_scalars
from pytorch_lightning.utilities.warnings import WarningCache

# re-define the ones from pytorch_lightning.utilities.types without the `Number` type
# TODO(@tchaton): Typing-pickle issue on python<3.7 (https://github.com/cloudpipe/cloudpickle/pull/318)
_METRIC = Any  # Union[Metric, torch.Tensor]
_METRIC_COLLECTION = Union[_METRIC, Mapping[str, _METRIC]]

warning_cache = WarningCache()


class MetricSource(LightningEnum):
    CALLBACK = "callback"
    PBAR = "pbar"
    LOG = "log"


@dataclass
class _Sync:
    fn: Optional[Callable] = None
    _should: bool = False
    rank_zero_only: bool = False
    op: Optional[str] = None
    group: Optional[Any] = None

    def __post_init__(self) -> None:
        self._generate_sync_fn()

    @property
    def should(self) -> bool:
        return self._should

    @should.setter
    def should(self, should: bool) -> None:
        self._should = should
        # `self._fn` needs to be re-generated.
        self._generate_sync_fn()

    def _generate_sync_fn(self) -> None:
        """Used to compute the syncing function and cache it."""
        fn = self.no_op if self.fn is None or not self.should or self.rank_zero_only else self.fn
        # save the function as `_fn` as the meta are being re-created and the object references need to match.
        self._fn = partial(fn, reduce_op=self.op, group=self.group)

    @property
    def __call__(self) -> Any:
        return self._fn

    @staticmethod
    def no_op(value: Any, *_, **__) -> Any:
        return value


@dataclass
class _Metadata:
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
    _sync: Optional[_Sync] = None

    def __post_init__(self) -> None:
        self._parse_reduce_fx()

    def _parse_reduce_fx(self) -> None:
        error = (
            "Only `self.log(..., reduce_fx={min,max,mean,sum})` are currently supported."
            " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`."
            f" Found: {self.reduce_fx}"
        )
        if isinstance(self.reduce_fx, str):
            reduce_fx = self.reduce_fx.lower()
            if reduce_fx == "avg":
                reduce_fx = "mean"
            if reduce_fx not in ("min", "max", "mean", "sum"):
                raise MisconfigurationException(error)
            self.reduce_fx = getattr(torch, reduce_fx)
        elif self.is_custom_reduction:
            raise MisconfigurationException(error)

    @property
    def sync(self) -> Optional[_Sync]:
        return self._sync

    @sync.setter
    def sync(self, sync: _Sync) -> None:
        if sync.op is None:
            sync.op = self.reduce_fx.__name__
        self._sync = sync

    @property
    def forked(self) -> bool:
        return self.on_step and self.on_epoch

    def forked_name(self, on_step: bool) -> str:
        if self.forked:
            return f'{self.name}_{"step" if on_step else "epoch"}'
        return self.name

    @property
    def is_mean_reduction(self) -> bool:
        return self.reduce_fx is torch.mean

    @property
    def is_sum_reduction(self) -> bool:
        return self.reduce_fx in (torch.sum, sum)

    @property
    def is_max_reduction(self) -> bool:
        return self.reduce_fx in (torch.max, max)

    @property
    def is_min_reduction(self) -> bool:
        return self.reduce_fx in (torch.min, min)

    @property
    def is_custom_reduction(self) -> bool:
        return not (self.is_mean_reduction or self.is_max_reduction or self.is_min_reduction or self.is_sum_reduction)

    def __getstate__(self) -> dict:
        # drop the `sync.fn` to avoid potential pickle errors
        # need to drop `fn` first otherwise `asdict` produces a `RecursionError`
        copy = replace(self, _sync=replace(self.sync, fn=None))
        d = asdict(copy)
        # delete the `None` value so it does not override
        del d["_sync"]["fn"]
        return d

    def __setstate__(self, state: dict, sync_fn: Optional[Callable] = None) -> None:
        d = {**state, "_sync": _Sync(**state["_sync"], fn=sync_fn)}
        self.__dict__.update(d)

    @classmethod
    def _reconstruct(cls, state: dict, sync_fn: Optional[Callable] = None) -> "_Metadata":
        meta = cls(state["fx"], state["name"])
        meta.__setstate__(state, sync_fn=sync_fn)
        return meta


class ResultMetric(Metric, DeviceDtypeModuleMixin):
    """Wraps the value provided to `:meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""

    def __init__(self, metadata: _Metadata, is_tensor: bool) -> None:
        super().__init__()
        self.is_tensor = is_tensor
        self.meta = metadata
        self.has_reset = False
        if is_tensor:
            self.add_state("value", torch.tensor(0, dtype=torch.float), dist_reduce_fx=torch.sum)
            if self.meta.is_mean_reduction:
                self.add_state("cumulated_batch_size", torch.tensor(0, dtype=torch.float), dist_reduce_fx=torch.sum)

    def update(self, value: _METRIC, batch_size: torch.Tensor) -> None:
        if self.is_tensor:
            value = value.float()
            # performance: no need to accumulate on values only logged on_step
            if self.meta.on_step and not self.meta.on_epoch:
                self._forward_cache = self.value = self.meta.sync(value)
                return
            self._forward_cache = value
            # perform accumulation with reduction
            if self.meta.is_mean_reduction:
                self.value += value.mean() * batch_size
                self.cumulated_batch_size += batch_size
            elif self.meta.is_max_reduction or self.meta.is_min_reduction:
                self.value = self.meta.reduce_fx(self.value, value.mean())
            elif self.meta.is_sum_reduction:
                self.value += value.mean() * batch_size
        else:
            self.value = value  # noqa: attribute-defined-outside-init
            self._forward_cache = value._forward_cache

    def compute(self) -> torch.Tensor:
        if self.is_tensor:
            value = self.meta.sync(self.value)
            if self.meta.is_mean_reduction:
                cumulated_batch_size = self.meta.sync(self.cumulated_batch_size)
                return value / cumulated_batch_size
            return value
        return self.value.compute()

    def reset(self) -> None:
        if self.is_tensor:
            super().reset()
        else:
            self.value.reset()
        self.has_reset = True

    def forward(self, value: _METRIC, batch_size: torch.Tensor) -> None:
        if self.meta.enable_graph:
            with torch.no_grad():
                self.update(value, batch_size)
        else:
            # performance: skip the `torch.no_grad` context manager by calling `update` directly
            self.update(value, batch_size)

    def _wrap_compute(self, compute: Any) -> Any:
        # Override to avoid syncing - we handle it ourselves.
        @wraps(compute)
        def wrapped_func(*args, **kwargs):
            if not self._update_called:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                    UserWarning,
                )

            # return cached value
            if self._computed is not None:
                return self._computed
            self._computed = compute(*args, **kwargs)
            return self._computed

        return wrapped_func

    def __setattr__(self, key: str, value: Any) -> None:
        # performance: skip the `torch.nn.Module.__setattr__` checks
        object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        state = f"{repr(self.meta.name)}, value={self.value}"
        if self.is_tensor and self.meta.is_mean_reduction:
            state += f", cumulated_batch_size={self.cumulated_batch_size}"
        return f"{self.__class__.__name__}({state})"

    def __getstate__(self, drop_value: bool = False) -> dict:
        skip = ["update", "compute", "_update_signature"]
        if not self.is_tensor and drop_value:
            # Avoid serializing ResultMetrics which are passed Metrics
            skip.append("value")
        d = {k: v for k, v in self.__dict__.items() if k not in skip}
        d["meta"] = d["meta"].__getstate__()
        d["_class"] = self.__class__.__name__
        return d

    def __setstate__(self, state: dict, sync_fn: Optional[Callable] = None) -> None:
        d = {**state, "meta": _Metadata._reconstruct(state["meta"], sync_fn=sync_fn)}
        super().__setstate__(d)

    @classmethod
    def _reconstruct(cls, state: dict, sync_fn: Optional[Callable] = None) -> "ResultMetric":
        # need to reconstruct twice because `meta` is used in `__init__`
        meta = _Metadata._reconstruct(state["meta"])
        result_metric = cls(meta, state["is_tensor"])
        result_metric.__setstate__(state, sync_fn=sync_fn)
        return result_metric

    def to(self, *args: Any, **kwargs: Any) -> "DeviceDtypeModuleMixin":
        self.__dict__.update(
            apply_to_collection(self.__dict__, (torch.Tensor, Metric), move_data_to_device, *args, **kwargs)
        )
        return self


class ResultMetricCollection(dict):
    """
    Dict wrapper for easy access to metadata.

    All of the leaf items should be instances of
    :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetric`
    with the same metadata.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

    @property
    def meta(self) -> _Metadata:
        return list(self.values())[0].meta

    def __getstate__(self, drop_value: bool = False) -> dict:
        def getstate(item: ResultMetric) -> dict:
            return item.__getstate__(drop_value=drop_value)

        items = apply_to_collection(dict(self), (ResultMetric, ResultMetricCollection), getstate)
        return {"items": items, "meta": self.meta.__getstate__(), "_class": self.__class__.__name__}

    def __setstate__(self, state: dict, sync_fn: Optional[Callable] = None) -> None:
        def setstate(item: dict) -> Union[Dict[str, ResultMetric], ResultMetric, Any]:
            # recurse through dictionaries to set the state. can't use `apply_to_collection`
            # as it does not recurse items of the same type.
            if not isinstance(item, dict):
                return item
            if item.get("_class") == ResultMetric.__name__:
                return ResultMetric._reconstruct(item, sync_fn=sync_fn)
            return {k: setstate(v) for k, v in item.items()}

        items = setstate(state["items"])
        self.update(items)

    @classmethod
    def _reconstruct(cls, state: dict, sync_fn: Optional[Callable] = None) -> "ResultMetricCollection":
        rmc = cls()
        rmc.__setstate__(state, sync_fn=sync_fn)
        return rmc


class ResultCollection(dict):
    """
    Collection (dictionary) of :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetric` or
    :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetricCollection`

    Example:

        # `device` needs to be provided before logging
        result = ResultCollection(training=True, torch.device("cpu"))

        # you can log to a specific collection.
        # arguments: fx, key, value, metadata
        result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)
        result.log('validation_step', 'recall', torch.tensor(...), on_step=True, on_epoch=True)
    """

    DATALOADER_SUFFIX = "/dataloader_idx_{}"

    def __init__(self, training: bool, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        self.training = training
        self._minimize = None
        self._batch_size = torch.tensor(1, device=device)
        self.device: Optional[Union[str, torch.device]] = device

    @property
    def result_metrics(self) -> List[ResultMetric]:
        o = []

        def append_fn(v: ResultMetric) -> None:
            nonlocal o
            o.append(v)

        apply_to_collection(list(self.values()), ResultMetric, append_fn)
        return o

    @property
    def batch_size(self) -> torch.Tensor:
        # performance: cache the `batch_size` tensor instead of re-creating it
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self._batch_size = torch.tensor(value, device=self.device)

    @property
    def minimize(self) -> Optional[torch.Tensor]:
        """
        The :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` loss
        will be saved as the ``minimize`` attribute.
        """
        return self._minimize

    @minimize.setter
    def minimize(self, loss: Optional[torch.Tensor]) -> None:
        if loss is not None and not isinstance(loss, torch.Tensor):
            raise ValueError(f"`Result.minimize` must be a `torch.Tensor`, found: {loss}")
        self._minimize = loss

    @property
    def extra(self) -> Dict[str, Any]:
        """
        Extras are any keys other than the loss returned by
        :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`
        """
        self.setdefault("_extra", {})
        return self["_extra"]

    @extra.setter
    def extra(self, extra: Dict[str, Any]) -> None:
        def check_fn(v):
            if v.grad_fn is not None:
                warning_cache.deprecation(
                    f"One of the returned values {set(extra.keys())} has a `grad_fn`. We will detach it automatically"
                    " but this behaviour will change in v1.6. Please detach it manually:"
                    " `return {'loss': ..., 'something': something.detach()}`"
                )
                return v.detach()
            return v

        # update instead of replace to keep the extra dict reference. TODO: remove with v1.6 deprecation removal
        extra.update(apply_to_collection(extra, torch.Tensor, check_fn))
        self["_extra"] = extra

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
        sync_dist_fn: Callable = _Sync.no_op,
        sync_dist_group: Optional[Any] = None,
        dataloader_idx: Optional[int] = None,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph:
            value = recursive_detach(value)

        # move metrics to cpu on TPU.
        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        # storage key
        key = f"{fx}.{name}"
        # add dataloader_suffix to both key and fx
        if dataloader_idx is not None:
            key += f".{dataloader_idx}"
            fx += f".{dataloader_idx}"

        meta = _Metadata(
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
        )
        meta.sync = _Sync(_should=sync_dist, fn=sync_dist_fn, group=sync_dist_group, rank_zero_only=rank_zero_only)

        # register logged value if it doesn't exist
        if key not in self:
            self.register_key(key, meta, value)

        # check the stored metadata and the current one match
        elif meta != self[key].meta:
            raise MisconfigurationException(
                f"You called `self.log({name}, ...)` twice in `{fx}` with different arguments. This is not allowed"
            )

        if batch_size is not None:
            self.batch_size = batch_size

        self.update_metrics(key, value)

    def register_key(self, key: str, meta: _Metadata, value: _METRIC_COLLECTION) -> None:
        """Create one ResultMetric object per value. Value can be provided as a nested collection"""

        def fn(v: _METRIC) -> ResultMetric:
            metric = ResultMetric(meta, isinstance(v, torch.Tensor))
            return metric.to(self.device)

        value = apply_to_collection(value, (torch.Tensor, Metric), fn)
        if isinstance(value, dict):
            value = ResultMetricCollection(value)
        self[key] = value

    def update_metrics(self, key: str, value: _METRIC_COLLECTION) -> None:
        def fn(result_metric, v):
            # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
            result_metric.forward(v.to(self.device), self.batch_size)
            result_metric.has_reset = False

        apply_to_collections(self[key], value, ResultMetric, fn)

    @staticmethod
    def _get_cache(result_metric: ResultMetric, on_step: bool) -> Optional[torch.Tensor]:
        cache = None
        if on_step and result_metric.meta.on_step:
            cache = result_metric._forward_cache
        elif not on_step and result_metric.meta.on_epoch:
            if not result_metric._computed:
                # always reduce on epoch end
                should = result_metric.meta.sync.should
                result_metric.meta.sync.should = True
                result_metric.compute()
                result_metric.meta.sync.should = should
            cache = result_metric._computed
        if cache is not None and not result_metric.meta.enable_graph:
            return cache.detach()
        return cache

    def valid_items(self) -> Generator:
        """This function is used to iterate over current valid metrics."""
        return (
            (k, v) for k, v in self.items() if not k == "_extra" and not (isinstance(v, ResultMetric) and v.has_reset)
        )

    def _forked_name(self, result_metric: ResultMetric, on_step: bool) -> Tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step)
        dl_idx = result_metric.meta.dataloader_idx
        if dl_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dl_idx)
            name += dataloader_suffix
            forked_name += dataloader_suffix
        return name, forked_name

    def metrics(self, on_step: bool) -> Dict[MetricSource, Dict[str, _METRIC]]:
        metrics = {k: {} for k in MetricSource}

        for _, result_metric in self.valid_items():

            # extract forward_cache or computed from the ResultMetric. ignore when the output is None
            value = apply_to_collection(result_metric, ResultMetric, self._get_cache, on_step, include_none=False)

            # convert metric collection to dict container.
            if isinstance(value, ResultMetricCollection):
                value = dict(value.items())

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
                metrics[MetricSource.PBAR][forked_name] = metrics_to_scalars(value)

        return metrics

    def reset(self, metrics: Optional[bool] = None, fx: Optional[str] = None) -> None:
        """
        Reset the result collection

        Args:
            metrics: If True, only ``torchmetrics.Metric`` results are reset,
                if False, only ``torch.Tensors`` are reset,
                if ``None``, both are.
            fx: Function to reset
        """

        def fn(item: ResultMetric) -> None:
            requested_type = metrics is None or metrics ^ item.is_tensor
            same_fx = fx is None or fx == item.meta.fx
            if requested_type and same_fx:
                item.reset()

        apply_to_collection(self, ResultMetric, fn)

    def extract_batch_size(self, batch: Any) -> None:
        try:
            self.batch_size = extract_batch_size(batch)
        except RecursionError:
            self.batch_size = 1

    def to(self, *args, **kwargs) -> "ResultCollection":
        """Move all data to the given device."""
        self.update(apply_to_collection(dict(self), (torch.Tensor, Metric), move_data_to_device, *args, **kwargs))

        if self.minimize is not None:
            self.minimize = self.minimize.to(*args, **kwargs)
        self._batch_size = self._batch_size.to(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        return self

    def cpu(self) -> "ResultCollection":
        """Move all data to CPU."""
        return self.to(device="cpu")

    def __str__(self) -> str:
        # sample output: `ResultCollection(minimize=1.23, {})`
        minimize = f"minimize={self.minimize}, " if self.minimize is not None else ""
        # remove empty values
        self_str = str({k: v for k, v in self.items() if v})
        return f"{self.__class__.__name__}({minimize}{self_str})"

    def __repr__(self):
        # sample output: `{True, cpu, minimize=tensor(1.23 grad_fn=<SumBackward0>), {'_extra': {}}}`
        minimize = f"minimize={repr(self.minimize)}, " if self.minimize is not None else ""
        return f"{{{self.training}, {repr(self.device)}, " + minimize + f"{super().__repr__()}}}"

    def __getstate__(self, drop_value: bool = True) -> dict:
        d = self.__dict__.copy()

        # can't deepcopy tensors with grad_fn
        minimize = d["_minimize"]
        if minimize is not None:
            d["_minimize"] = minimize.detach()

        extra = self.get("_extra")
        if extra is not None:
            d["_extra"] = extra

        # all the items should be either `ResultMetric`s or `ResultMetricCollection`s
        items = {k: v.__getstate__(drop_value=drop_value) for k, v in self.items() if k != "_extra"}
        return {**d, "items": items}

    def __setstate__(
        self, state: dict, map_location: Optional[Union[str, torch.device]] = None, sync_fn: Optional[Callable] = None
    ) -> None:
        self.__dict__.update({k: v for k, v in state.items() if k != "items"})

        def setstate(k: str, item: dict) -> Union[ResultMetric, ResultMetricCollection]:
            if not isinstance(item, dict):
                raise ValueError(f"Unexpected value: {item}")
            cls = item["_class"]
            if cls == ResultMetric.__name__:
                cls = ResultMetric
            elif cls == ResultMetricCollection.__name__:
                cls = ResultMetricCollection
            else:
                raise ValueError(f"Unexpected class name: {cls}")
            _sync_fn = sync_fn or (self[k].meta.sync.fn if k in self else None)
            return cls._reconstruct(item, sync_fn=_sync_fn)

        items = {k: setstate(k, v) for k, v in state["items"].items()}
        self.update(items)

        device = map_location or self.device
        self.to(device)

    def state_dict(self, drop_value: bool = True) -> dict:
        return self.__getstate__(drop_value)

    def load_state_dict(
        self,
        state_dict: dict,
        map_location: Optional[Union[str, torch.device]] = None,
        sync_fn: Optional[Callable] = None,
        metrics: Optional[Dict[str, Metric]] = None,
    ) -> None:
        self.__setstate__(state_dict, map_location=map_location, sync_fn=sync_fn)

        if not metrics:
            return

        # iterate through result metrics and re-attached Metric references on reload.
        result_metrics = self.result_metrics
        for metric_attribute, metric in metrics.items():
            for result_metric in result_metrics:
                if result_metric.meta.metric_attribute == metric_attribute:
                    result_metric.value = metric
