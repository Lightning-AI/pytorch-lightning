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
from collections.abc import Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Tuple, Union
from weakref import proxy

import torch
from torch import Tensor
from torchmetrics import Metric

from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _METRIC


class DefaultMetricsKeys(LightningEnum):
    CALLBACK = "callback"
    PBAR = "pbar"
    LOG = "log"


@dataclass
class Metadata:
    fx: str  # TODO: distinction?
    name: str
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: Callable = torch.mean
    dataloader_idx: Optional[int] = None
    is_tensor: bool = True
    lightning_attribute_name: Optional[str] = None

    @property
    def forked(self) -> bool:
        return self.on_step and self.on_epoch

    @property
    def forked_step_name(self) -> str:
        if self.forked:
            return self.name + "_step"
        return self.name

    @property
    def forked_epoch_name(self) -> str:
        if self.forked:
            return self.name + "_epoch"
        return self.name

    @property
    def is_tensor_and_mean_reduction(self) -> bool:
        return self.is_tensor and self.reduce_fx == torch.mean

    @property
    def is_tensor_and_max_reduction(self) -> bool:
        return self.is_tensor and (self.reduce_fx in (torch.max, max))

    @property
    def is_tensor_and_min_reduction(self) -> bool:
        return self.is_tensor and (self.reduce_fx in (torch.min, min))


class ResultMetric(Metric):

    DTYPE = torch.float32

    def __init__(self, metadata: Metadata) -> None:
        super().__init__(compute_on_step=metadata.is_tensor)
        self.meta = metadata
        if self.meta.is_tensor:
            self.add_state("value", torch.tensor(.0, dtype=self.DTYPE))
            if self.meta.is_tensor_and_mean_reduction:
                self.add_state("cumulated_batch_size", torch.tensor(.0, dtype=self.DTYPE))

    def update(self, value: _METRIC, batch_size: Optional[int] = None) -> None:
        if self.meta.is_tensor_and_mean_reduction:
            self.value += value.float().mean() * batch_size
            self.cumulated_batch_size += batch_size

        elif self.meta.is_tensor_and_max_reduction:
            self.value = max(self.value, value.float().mean())

        elif self.meta.is_tensor_and_min_reduction:
            self.value = min(self.value, value.float().mean())

        else:
            self.value = value
            self._forward_cache = value._forward_cache

    def compute(self) -> torch.Tensor:
        if self.meta.is_tensor:
            if self.meta.is_tensor_and_mean_reduction:
                return torch.sum(self.value) / torch.sum(self.cumulated_batch_size)
            elif self.meta.is_tensor_and_max_reduction or self.meta.is_tensor_and_min_reduction:
                return self.meta.fx(self.value)
            else:
                raise MisconfigurationException("Only mean, max are supported.")
        else:
            return self.value.compute()

    def __repr__(self) -> str:
        if self.meta.is_tensor_and_mean_reduction:
            attr = f"value={self.value}, cumulated_batch_size={self.cumulated_batch_size}"
        else:
            attr = f"value={getattr(self, 'value', None)}"
        return f"{self.__class__.__name__}({attr})"

    def reset(self):
        if self.meta.is_tensor:
            super().reset()
        else:
            self.value.reset()

    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)

        if self.compute_on_step:
            self._to_sync = self.dist_sync_on_step

            # save context before switch
            cache = {attr: getattr(self, attr) for attr in self._defaults.keys()}

            # call reset, update, compute, on single batch
            self.reset()
            self.update(*args, **kwargs)
            self._forward_cache = self.compute()

            # restore context
            for attr, val in cache.items():
                setattr(self, attr, val)
            self._to_sync = True
            self._computed = None

            return self._forward_cache


class ResultMeta(Dict):
    pass


class ResultCollection(dict):

    STEP_SUFFIX = "_step"
    EPOCH_SUFFIX = "_epoch"
    DATALOADER_SUFFIX = "/dataloader_idx_{}"

    def __init__(self, is_train: bool) -> None:
        super().__init__()
        self.is_train = is_train
        self._on_epoch_end_reached = False
        self._minimize = None
        self._current_hook_name: Optional[str] = None
        self._batch_idx: Optional[int] = None
        self._root_device: Optional[torch.device] = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    @property
    def root_device(self) -> Optional[torch.device]:
        return self._root_device

    @root_device.setter
    def root_device(self, root_device: torch.device) -> None:
        self._root_device = root_device

    @property
    def batch_idx(self) -> int:
        return self._batch_idx

    @batch_idx.setter
    def batch_idx(self, batch_idx: int) -> None:
        self._batch_idx = batch_idx

    @property
    def on_epoch_end_reached(self) -> bool:
        return self._on_epoch_end_reached

    @on_epoch_end_reached.setter
    def on_epoch_end_reached(self, on_epoch_end_reached):
        self._on_epoch_end_reached = on_epoch_end_reached
        self._batch_idx = None

    @property
    def metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.get_epoch_metrics() if self.on_epoch_end_reached else self.get_batch_metrics()

    @property
    def minimize(self) -> Optional[Tensor]:
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
        return self.get('extra', {})

    @extra.setter
    def extra(self, extra: Dict) -> None:

        def detach_fn(v):
            return v.detach()

        extra = apply_to_collection(extra, torch.Tensor, detach_fn)
        self['extra'] = extra

    def log(
        self,
        hook_name: str,
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
        lightning_attribute_name: Optional[str] = None,
    ):
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs

        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        key = f"{hook_name}.{name}"

        if dataloader_idx:
            key += f'.{dataloader_idx}'

        if on_step and self.on_epoch_end_reached:
            raise MisconfigurationException("Logging `on_step` after `on_epoch_end_reached` isn't authorized.")

        if key not in self:
            meta = Metadata(
                fx=hook_name,
                name=name,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                dataloader_idx=dataloader_idx,
                lightning_attribute_name=lightning_attribute_name,
            )
            self.instance_result_metric(key, meta, value)

        self.update_metrics(hook_name, key, value, batch_size or torch.tensor(1.))

        self._current_hook_name = hook_name

    def instance_result_metric(self, key: str, meta: Metadata, value: Union[Dict, torch.Tensor]) -> None:

        def fn(v):
            assert self.root_device is not None
            nonlocal meta
            meta = deepcopy(meta)
            meta.is_tensor = torch.is_tensor(v)
            metric = ResultMetric(meta)
            return metric.to(self.root_device)

        self[key] = apply_to_collection(value, (torch.Tensor, Metric), fn)
        # cache the meta for reduction
        if not isinstance(self[key], ResultMetric):
            self[key + '.forked'] = meta.forked
            self[key + '.logger'] = meta.logger
            self[key + '.prog_bar'] = meta.prog_bar
            self[key + '.on_epoch'] = meta.on_epoch
            self[key + '.dataloader_idx'] = meta.dataloader_idx

    def update_metrics(self, hook_name: str, key: str, value: Union[Dict, torch.Tensor], batch_size) -> None:

        if isinstance(self._current_hook_name,
                      str) and self._current_hook_name != hook_name and self.batch_idx in (None, 0):
            # when restarting an new epoch, reset the tensor hooks dynamically.
            self.reset_metrics(hook_name, is_tensor=True)

        def fn(result_metric, v):
            assert isinstance(v, (torch.Tensor, Metric))
            result_metric(v.to(self.root_device), batch_size)

        apply_to_collections(self[key], value, ResultMetric, fn)

    @staticmethod
    def _get_forward_cache(result_metric: ResultMetric) -> Optional[torch.Tensor]:
        if not result_metric.meta.on_step:
            return

        return result_metric._forward_cache.detach()

    @staticmethod
    def _to_item(forward_cache: torch.Tensor) -> float:
        return forward_cache.item()

    def valid_metrics(self) -> Generator:
        for key, item in self.items():
            if item is None or isinstance(item, bool) or key == "extra":
                continue
            yield (key, item)

    def _extract_metadata(self, key: str, result_metric, on_step: bool, suffix: str) -> Tuple:
        if isinstance(result_metric, ResultMetric):
            name = result_metric.meta.name
            name_forked = result_metric.meta.forked_step_name if on_step else result_metric.meta.forked_epoch_name
            logger = result_metric.meta.logger
            prog_bar = result_metric.meta.prog_bar
            metric_on_epoch = result_metric.meta.on_epoch
            dataloader_idx = result_metric.meta.dataloader_idx
        else:
            name = key.split('.')[-1]
            name_forked = name + suffix if self[key + '.forked'] else name
            logger = self[key + '.logger']
            prog_bar = self[key + '.prog_bar']
            metric_on_epoch = self[key + '.on_epoch']
            dataloader_idx = self[key + '.dataloader_idx']

        if dataloader_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dataloader_idx)
            name += dataloader_suffix
            name_forked += dataloader_suffix

        return name, name_forked, logger, prog_bar, metric_on_epoch

    def get_metrics(self, on_step: bool) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = {k: {} for k in DefaultMetricsKeys}
        fn = self._get_forward_cache if on_step else self._get_computed_cache
        suffix = self.STEP_SUFFIX if on_step else self.EPOCH_SUFFIX

        # iterate over all stored metrics.
        for key, result_metric in self.valid_metrics():

            # extract forward_cache or computed from the ResultMetric
            # ignore when the output of fn is None
            value = apply_to_collection(result_metric, ResultMetric, fn, remove_none=True)

            # detect if the value is None. This can be nested.
            is_empty = True

            def is_empty_fn(v):
                nonlocal is_empty
                if v is not None:
                    is_empty = False

            # apply detection.
            wrong_dtype = (
                Mapping,
                Sequence,
                NamedTuple,
            )
            apply_to_collection(value, object, is_empty_fn, wrong_dtype=wrong_dtype)

            # skip is the value was actually empty.
            if is_empty:
                continue

            # extract metadata
            name, name_forked, logger, prog_bar, metric_on_epoch = self._extract_metadata(
                key, result_metric, on_step, suffix
            )

            # populate logging metrics
            if logger:
                metrics[DefaultMetricsKeys.LOG][name_forked] = value

            # populate callback metrics
            # callback metrics don't take `_step` forked metrics.
            if not self.is_train and (not metric_on_epoch or on_step):
                pass
            else:
                metrics[DefaultMetricsKeys.CALLBACK][name] = value
                metrics[DefaultMetricsKeys.CALLBACK][name_forked] = value

            # populate progress_bar metrics. By default, the value should be converted to a float.
            if prog_bar:
                value = apply_to_collection(value, torch.Tensor, self._to_item, remove_none=True)
                metrics[DefaultMetricsKeys.PBAR][name_forked] = value

        return metrics

    def get_batch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.get_metrics(on_step=True)

    @staticmethod
    def _get_computed_cache(result_metric: ResultMetric) -> Optional[torch.Tensor]:
        if not result_metric.meta.on_epoch:
            return

        if not result_metric._computed:
            result_metric.compute()

        return result_metric._computed.detach()

    def get_epoch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
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

    def reset_metrics(self, hook_name: str = None, is_tensor: bool = False) -> None:
        """Call at the end of epoch to reset all results provided as `Metric` or `tensor`"""

        def reset_fn(item: ResultMetric) -> None:
            nonlocal hook_name
            nonlocal is_tensor
            if item.meta.is_tensor == is_tensor:
                if isinstance(hook_name, str) and hook_name != item.meta.fx:
                    return
                item.reset()

        apply_to_collection(dict(self.items()), ResultMetric, reset_fn)

    def reset(self):
        self.reset_metrics()
        self.on_epoch_end_reached = False

    def extract_batch_size(self, batch: Any) -> None:
        try:
            self._batch_size = self._extract_batch_size(batch)
        except RecursionError:
            self._batch_size = 1

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

    def __repr__(self) -> str:
        repr = f'{self.__class__.__name__}' + '{\n'
        for k in sorted(self.keys()):
            v = self[k]
            repr += f"  {k}: {v},\n"
        return repr[:-1] + '\n}'

    def state_dict(self):

        def get_state_dict(item: ResultMetric) -> Dict[str, Any]:
            state = item.__getstate__()
            # delete reference to TorchMetrics Metric
            state = deepcopy(state)
            if 'value' in state['_modules'] and isinstance(state['_modules']["value"], Metric):
                del state['_modules']["value"]
            return ResultMeta(**state)

        return {k: apply_to_collection(v, ResultMetric, get_state_dict) for k, v in self.items()}

    def load_from_state_dict(self, state_dict: Dict[str, Any], metrics: Dict[str, Metric]):

        def to_result_metric(item: ResultMeta) -> Dict[str, Any]:
            result_metric = ResultMetric(item["meta"])
            result_metric.__dict__.update(item)
            return result_metric

        state_dict = {k: apply_to_collection(v, ResultMeta, to_result_metric) for k, v in state_dict.items()}

        for k, v in state_dict.items():
            self[k] = v

        if metrics:
            # the metric reference are lost during serialization and
            # they need to be set back during loading

            def re_assign_metric(item):
                nonlocal metrics
                lightning_attribute_name = item.meta.lightning_attribute_name
                if isinstance(lightning_attribute_name, str) and lightning_attribute_name in metrics:
                    item.value = metrics[lightning_attribute_name]

            apply_to_collection(dict(self.items()), ResultMetric, re_assign_metric)
