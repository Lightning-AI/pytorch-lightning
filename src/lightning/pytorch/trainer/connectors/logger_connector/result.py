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
from collections.abc import Generator
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Optional, Union, cast

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypedDict, override

from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.pytorch.utilities.data import extract_batch_size
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from lightning.pytorch.utilities.memory import recursive_detach
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn
from lightning.pytorch.utilities.warnings import PossibleUserWarning

_VALUE = Union[Metric, Tensor]  # Do not include scalars as they were converted to tensors
_OUT_DICT = dict[str, Tensor]
_PBAR_DICT = dict[str, float]


class _METRICS(TypedDict):
    callback: _OUT_DICT
    log: _OUT_DICT
    pbar: _PBAR_DICT


warning_cache = WarningCache()


@dataclass
class _Sync:
    fn: Optional[Callable] = None
    _should: bool = False
    rank_zero_only: bool = False
    _op: Optional[str] = None
    _group: Optional[Any] = None

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

    @property
    def op(self) -> Optional[str]:
        return self._op

    @op.setter
    def op(self, op: Optional[str]) -> None:
        self._op = op
        # `self._fn` needs to be re-generated.
        self._generate_sync_fn()

    @property
    def group(self) -> Optional[Any]:
        return self._group

    @group.setter
    def group(self, group: Optional[Any]) -> None:
        self._group = group
        # `self._fn` needs to be re-generated.
        self._generate_sync_fn()

    def _generate_sync_fn(self) -> None:
        """Used to compute the syncing function and cache it."""
        fn = self.no_op if self.fn is None or not self.should or self.rank_zero_only else self.fn
        # save the function as `_fn` as the meta are being re-created and the object references need to match.
        # ignore typing, bad support for `partial`: mypy/issues/1484
        self._fn: Callable = partial(fn, reduce_op=self.op, group=self.group)  # type: ignore[arg-type,operator,misc]

    @property
    def __call__(self) -> Any:
        return self._fn

    @staticmethod
    def no_op(value: Any, *_: Any, **__: Any) -> Any:
        return value


@dataclass
class _Metadata:
    fx: str
    name: str
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    # https://github.com/pytorch/pytorch/issues/96197
    reduce_fx: Callable = torch.mean
    enable_graph: bool = False
    add_dataloader_idx: bool = True
    dataloader_idx: Optional[int] = None
    metric_attribute: Optional[str] = None
    _sync: Optional[_Sync] = None

    def __post_init__(self) -> None:
        if not self.on_step and not self.on_epoch:
            raise MisconfigurationException("`self.log(on_step=False, on_epoch=False)` is not useful.")
        self._parse_reduce_fx()

    def _parse_reduce_fx(self) -> None:
        error = (
            "Only `self.log(..., reduce_fx={min,max,mean,sum})` are supported."
            " If you need a custom reduction, please log a `torchmetrics.Metric` instance instead."
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
    def sync(self) -> _Sync:
        assert self._sync is not None
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
            return f"{self.name}_{'step' if on_step else 'epoch'}"
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


class _ResultMetric(Metric):
    """Wraps the value provided to `:meth:`~lightning.pytorch.core.LightningModule.log`"""

    def __init__(self, metadata: _Metadata, is_tensor: bool) -> None:
        super().__init__()
        self.is_tensor = is_tensor
        self.meta = metadata
        self.has_reset = False
        if is_tensor:
            if metadata.is_max_reduction:
                default = float("-inf")
            elif metadata.is_min_reduction:
                default = float("inf")
            else:
                default = 0.0
            # the logged value will be stored in float32 or higher to maintain accuracy
            self.add_state("value", torch.tensor(default, dtype=_get_default_dtype()), dist_reduce_fx=torch.sum)
            if self.meta.is_mean_reduction:
                self.cumulated_batch_size: Tensor
                self.add_state("cumulated_batch_size", torch.tensor(0), dist_reduce_fx=torch.sum)
        # this is defined here only because upstream is missing the type annotation
        self._forward_cache: Optional[Any] = None

    @override
    def update(self, value: _VALUE, batch_size: int) -> None:
        if self.is_tensor:
            value = cast(Tensor, value)
            dtype = _get_default_dtype()
            if not torch.is_floating_point(value):
                warning_cache.warn(
                    # do not include the value to avoid cache misses
                    f"You called `self.log({self.meta.name!r}, ...)` in your `{self.meta.fx}` but the value needs to"
                    f" be floating to be reduced. Converting it to {dtype}."
                    " You can silence this warning by converting the value to floating point yourself."
                    " If you don't intend to reduce the value (for instance when logging the global step or epoch) then"
                    f" you can use `self.logger.log_metrics({{{self.meta.name!r}: ...}})` instead."
                )
                value = value.to(dtype)
            if value.dtype not in (torch.float32, torch.float64):
                value = value.to(dtype)

            if self.meta.on_step:
                self._forward_cache = self.meta.sync(value.clone())  # `clone` because `sync` is in-place
                # performance: no need to accumulate on values only logged on_step
                if not self.meta.on_epoch:
                    self.value = self._forward_cache
                    return

            # perform accumulation with reduction
            if self.meta.is_mean_reduction:
                # do not use `+=` as it doesn't do type promotion
                self.value = self.value + value * batch_size
                self.cumulated_batch_size = self.cumulated_batch_size + batch_size
            elif self.meta.is_max_reduction or self.meta.is_min_reduction:
                self.value = self.meta.reduce_fx(self.value, value)
            elif self.meta.is_sum_reduction:
                self.value = self.value + value
        else:
            value = cast(Metric, value)
            self.value = value
            self._forward_cache = value._forward_cache

    @override
    def compute(self) -> Tensor:
        if self.is_tensor:
            value = self.meta.sync(self.value.clone())  # `clone` because `sync` is in-place
            if self.meta.is_mean_reduction:
                cumulated_batch_size = self.meta.sync(self.cumulated_batch_size)
                return value / cumulated_batch_size
            return value
        return self.value.compute()

    @override
    def reset(self) -> None:
        if self.is_tensor:
            super().reset()
        else:
            self.value.reset()
        self.has_reset = True

    @override
    def forward(self, value: _VALUE, batch_size: int) -> None:
        if self.meta.enable_graph:
            with torch.no_grad():
                self.update(value, batch_size)
        else:
            # performance: skip the `torch.no_grad` context manager by calling `update` directly
            self.update(value, batch_size)

    @override
    def _wrap_compute(self, compute: Any) -> Any:
        # Override to avoid syncing - we handle it ourselves.
        @wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            update_called = self.update_called if _TORCHMETRICS_GREATER_EQUAL_1_0_0 else self._update_called
            if not update_called:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                )

            # return cached value
            if self._computed is not None:
                return self._computed
            self._computed = compute(*args, **kwargs)
            return self._computed

        return wrapped_func

    @override
    def __setattr__(self, key: str, value: Any) -> None:
        # performance: skip the `torch.nn.Module.__setattr__` checks
        object.__setattr__(self, key, value)

    @override
    def __repr__(self) -> str:
        state = f"{repr(self.meta.name)}, value={self.value}"
        if self.is_tensor and self.meta.is_mean_reduction:
            state += f", cumulated_batch_size={self.cumulated_batch_size}"
        return f"{self.__class__.__name__}({state})"

    @override
    def to(self, *args: Any, **kwargs: Any) -> "_ResultMetric":
        d = dict(self.__dict__)
        self.__dict__.update(apply_to_collection(d, (Tensor, Metric), move_data_to_device, *args, **kwargs))
        return self


class _ResultCollection(dict):
    """Collection (dictionary) of :class:`~lightning.pytorch.trainer.connectors.logger_connector.result._ResultMetric`

    Example::

        # you can log to a specific collection.
        # arguments: fx, key, value, metadata

        result = _ResultCollection(training=True)
        result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)
        result.log('validation_step', 'recall', torch.tensor(...), on_step=True, on_epoch=True)

    """

    DATALOADER_SUFFIX = "/dataloader_idx_{}"

    def __init__(self, training: bool) -> None:
        super().__init__()
        self.training = training
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.dataloader_idx: Optional[int] = None

    @property
    def result_metrics(self) -> list[_ResultMetric]:
        return list(self.values())

    def _extract_batch_size(self, value: _ResultMetric, batch_size: Optional[int], meta: _Metadata) -> int:
        # check if we have extracted the batch size already
        if batch_size is None:
            batch_size = self.batch_size

        if batch_size is not None:
            return batch_size

        batch_size = 1
        if self.batch is not None and value.is_tensor and meta.on_epoch and meta.is_mean_reduction:
            batch_size = extract_batch_size(self.batch)
            self.batch_size = batch_size

        return batch_size

    @torch.compiler.disable
    def log(
        self,
        fx: str,
        name: str,
        value: _VALUE,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: bool = False,
        on_epoch: bool = True,
        # https://github.com/pytorch/pytorch/issues/96197
        reduce_fx: Callable = torch.mean,
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_fn: Callable = _Sync.no_op,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """See :meth:`~lightning.pytorch.core.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph:
            value = recursive_detach(value)

        # storage key
        key = f"{fx}.{name}"
        # add dataloader_suffix to both key and fx
        if add_dataloader_idx and self.dataloader_idx is not None:
            key += f".{self.dataloader_idx}"
            fx += f".{self.dataloader_idx}"

        meta = _Metadata(
            fx=fx,
            name=name,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            add_dataloader_idx=add_dataloader_idx,
            dataloader_idx=self.dataloader_idx,
            metric_attribute=metric_attribute,
        )
        meta.sync = _Sync(_should=sync_dist, fn=sync_dist_fn, _group=sync_dist_group, rank_zero_only=rank_zero_only)

        # register logged value if it doesn't exist
        if key not in self:
            metric = _ResultMetric(meta, isinstance(value, Tensor))
            self[key] = metric

        # check the stored metadata and the current one match
        elif meta != self[key].meta:
            raise MisconfigurationException(
                f"You called `self.log({name}, ...)` twice in `{fx}` with different arguments. This is not allowed"
            )
        self[key].to(value.device)

        batch_size = self._extract_batch_size(self[key], batch_size, meta)
        self.update_metrics(key, value, batch_size)

    @torch.compiler.disable
    def update_metrics(self, key: str, value: _VALUE, batch_size: int) -> None:
        result_metric = self[key]
        # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
        result_metric.forward(value, batch_size)
        result_metric.has_reset = False

    @staticmethod
    def _get_cache(result_metric: _ResultMetric, on_step: bool) -> Optional[Tensor]:
        cache = None
        if on_step and result_metric.meta.on_step:
            cache = result_metric._forward_cache
        elif not on_step and result_metric.meta.on_epoch:
            if result_metric._computed is None:
                should = result_metric.meta.sync.should
                if not should and result_metric.is_tensor and _distributed_is_initialized():
                    warning_cache.warn(
                        f"It is recommended to use `self.log({result_metric.meta.name!r}, ..., sync_dist=True)`"
                        " when logging on epoch level in distributed setting to accumulate the metric across"
                        " devices.",
                        category=PossibleUserWarning,
                    )
                result_metric.compute()
                result_metric.meta.sync.should = should

            cache = result_metric._computed

        if cache is not None:
            if not isinstance(cache, Tensor):
                raise ValueError(
                    f"The `.compute()` return of the metric logged as {result_metric.meta.name!r} must be a tensor."
                    f" Found {cache}"
                )
            if not result_metric.meta.enable_graph:
                return cache.detach()

        return cache

    def valid_items(self) -> Generator:
        """This function is used to iterate over current valid metrics."""
        return ((k, v) for k, v in self.items() if not v.has_reset and self.dataloader_idx == v.meta.dataloader_idx)

    def _forked_name(self, result_metric: _ResultMetric, on_step: bool) -> tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step)
        add_dataloader_idx = result_metric.meta.add_dataloader_idx
        dl_idx = result_metric.meta.dataloader_idx
        if add_dataloader_idx and dl_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dl_idx)
            name += dataloader_suffix
            forked_name += dataloader_suffix
        return name, forked_name

    def metrics(self, on_step: bool) -> _METRICS:
        metrics = _METRICS(callback={}, log={}, pbar={})

        for _, result_metric in self.valid_items():
            # extract forward_cache or computed from the _ResultMetric
            value = self._get_cache(result_metric, on_step)
            if not isinstance(value, Tensor):
                continue

            name, forked_name = self._forked_name(result_metric, on_step)

            # populate logging metrics
            if result_metric.meta.logger:
                metrics["log"][forked_name] = value

            # populate callback metrics. callback metrics don't take `_step` forked metrics
            if self.training or result_metric.meta.on_epoch and not on_step:
                metrics["callback"][name] = value
                metrics["callback"][forked_name] = value

            # populate progress_bar metrics. convert tensors to numbers
            if result_metric.meta.prog_bar:
                metrics["pbar"][forked_name] = convert_tensors_to_scalars(value)

        return metrics

    def reset(self, metrics: Optional[bool] = None, fx: Optional[str] = None) -> None:
        """Reset the result collection.

        Args:
            metrics: If True, only ``torchmetrics.Metric`` results are reset,
                if False, only ``torch.Tensors`` are reset,
                if ``None``, both are.
            fx: Function to reset

        """
        for item in self.values():
            requested_type = metrics is None or metrics ^ item.is_tensor
            same_fx = fx is None or fx == item.meta.fx
            if requested_type and same_fx:
                item.reset()

    def to(self, *args: Any, **kwargs: Any) -> "_ResultCollection":
        """Move all data to the given device."""
        self.update(apply_to_collection(dict(self), (Tensor, Metric), move_data_to_device, *args, **kwargs))
        return self

    def cpu(self) -> "_ResultCollection":
        """Move all data to CPU."""
        return self.to(device="cpu")

    def __str__(self) -> str:
        # remove empty values
        self_str = str({k: v for k, v in self.items() if v})
        return f"{self.__class__.__name__}({self_str})"

    def __repr__(self) -> str:
        return f"{{{self.training}, {super().__repr__()}}}"


def _get_default_dtype() -> torch.dtype:
    """The default dtype for new tensors, but no lower than float32."""
    dtype = torch.get_default_dtype()
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32
