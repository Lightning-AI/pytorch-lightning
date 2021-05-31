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
    is_tensor: bool = True
    lightning_attribute_name: Optional[str] = None
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

    def __init__(self, metadata: Metadata) -> None:
        super().__init__(compute_on_step=metadata.is_tensor)
        self.meta = metadata
        if self.meta.is_tensor:
            self.add_state("value", torch.tensor(0, dtype=torch.float))
            if self.meta.is_mean_reduction:
                self.add_state("cumulated_batch_size", torch.tensor(0, dtype=torch.float))
        # TODO: self.value when not tensor?

    def update(self, value: _METRIC, batch_size: Optional[int] = None) -> None:
        # TODO: support for non-tensor. sync returns tensor always
        if self.meta.is_tensor:
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
        if self.meta.is_tensor:
            if self.meta.is_mean_reduction:
                return torch.sum(self.value) / torch.sum(self.cumulated_batch_size)
            elif self.meta.is_max_reduction or self.meta.is_min_reduction:
                return self.value
            raise MisconfigurationException(
                f"Only [min, max, mean] reductions are supported. Found {self.meta.reduce_fx}"
            )
        return self.value.compute()

    def __repr__(self) -> str:
        state = f"value={self.value}"
        if self.meta.is_tensor and self.meta.is_mean_reduction:
            state += f", cumulated_batch_size={self.cumulated_batch_size}"
        return f"{self.__class__.__name__}({state})"

    def reset(self) -> None:
        if self.meta.is_tensor:
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


# placeholder for apply_to_collection
class ResultMeta(Dict):
    pass


class ResultCollection(dict):
    """
    Collection (dictionary) of :class:`~pytorch_lightning.trainer.connectors.logger_connector.result.ResultMetric`

    Example:

        # `root_device` needs to be provided before logging
        result = ResultCollection(True, torch.device("cpu"))

        # you can log to a specific collection.
        # arguments: hook_name, key, value, metadata
        result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)
        result.log('validation_step', 'recall', torch.tensor(...), on_step=True, on_epoch=True)

        for epoch in epochs:
            for batch_idx, batch in enumerate(dataloader):
                # the batch_idx is used to reset the tensor metrics
                result.batch_idx = batch_idx
                result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)

            result.on_epoch_end_reached = True  # indicate epoch end has been reached
            result.log('training_epoch_end', 'acc', torch.tensor(...), on_step=False, on_epoch=True)

            # Optionally:
            result.reset_metrics() # reset the `torchmetrics.Metric`
            result.reset() # reset the entire `ResultCollection`
    """

    STEP_SUFFIX = "_step"
    EPOCH_SUFFIX = "_epoch"
    DATALOADER_SUFFIX = "/dataloader_idx_{}"

    def __init__(self, training: bool, root_device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.training = training
        self._on_epoch_end_reached = False
        self._minimize = None
        self._current_hook_name: Optional[str] = None
        self._batch_size: Optional[int] = None
        self._batch_idx: Optional[int] = None
        self._root_device: Optional[torch.device] = root_device
        self.fx_validator = FxValidator()

    @property
    def batch_size(self) -> int:
        return self._batch_size or 1

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
    def batch_idx(self) -> Optional[int]:
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
        """
        This function returns either batch or epoch metrics depending on `on_epoch_end_reached` attribute.
        The metrics are returned as:


        {
            MetricSource.PBAR: {...},
            MetricSource.LOG: {...},
            MetricSource.CALLBACK: {...}
        }
        """
        return self.get_epoch_metrics() if self.on_epoch_end_reached else self.get_batch_metrics()

    @property
    def minimize(self) -> Optional[Tensor]:
        return self._minimize

    @minimize.setter
    def minimize(self, loss: Optional[torch.Tensor]) -> None:
        """
        The `LightningModule.training_step` loss will be saved as the ResultCollection minimize attribute.
        """
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
        """
        The `LightningModule.training_step` extras will be saved as the ResultCollection extra key.
        """

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
        """
        This function is used to log metrics from with
        :meth:`~pytorch_lightning.core.lightning.LightningModule.log`

        Args:

            hook_name: Current hook name
            name: Key provided by the user on logging
            value: Either a number, tensor or a collection of the previous.
            prog_bar: Whether to add this value to the progress bar.
            logger: Whether to log this value to the loggers
            on_step: Whether to use this value during batch iteration.
            on_epoch: Whether to use this value at the end of the batch iteration.
                Automatic reduction will be performed.
            reduce_fx: Which function to use for reduction. Currently support min, max and mean.
            enable_graph: Whether to keep autograd graph when storing the value.
            dataloader_idx: The current dataloader idx. This will be used to automatically
                add `/dataloader_idx_{}` on the metrics.
            batch_size: Current batch size.
            lightning_attribute_name: When providing `nn.Metric` as a value, the ``metric_attribute``
                need to be provided to enable automatic saving / re-loading.

        """
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # move metrics to cpu on TPU.
        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        if isinstance(value, Metric) and lightning_attribute_name is None:
            raise MisconfigurationException(
                "The LightningModule attribute name should be provided when using torchmetrics.Metric"
            )

        # storage key
        key = f"{hook_name}.{name}"

        # add dataloader_suffix  to both key and hook_name
        if dataloader_idx is not None:
            # use as ResultCollection key
            key += f'.{dataloader_idx}'
            # used to decide when to reset
            hook_name += f'.{dataloader_idx}'

        if on_step and self.on_epoch_end_reached:
            raise MisconfigurationException("Logging `on_step` after `on_epoch_end_reached` isn't authorized.")

        if key not in self:
            # create metadata object if storage key doesn't exist in self
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
            # create one ResultMetric object per value.
            # value can be provided as a nested collection.
            self.instance_result_metric(key, meta, value)

        # compute batch_size
        batch_size = torch.tensor(batch_size or self.batch_size, device=self.root_device)

        # update the ResultMetric
        self.update_metrics(hook_name, key, value, batch_size)

        # save current_hook to know when to reset.
        self._current_hook_name = hook_name

    def instance_result_metric(self, key: str, meta: Metadata, value: Union[Dict, torch.Tensor]) -> None:

        def fn(v: Union[torch.Tensor, Metric]) -> ResultMetric:
            # This local function is used to `ResultMetric`.
            # The `Metadata` is_tensor is modified on the fly
            assert self.root_device is not None
            nonlocal meta
            meta = deepcopy(meta)
            meta.is_tensor = torch.is_tensor(v)
            metric = ResultMetric(meta)
            return metric.to(self.root_device)

        # store a mapping between storage key and collection of `ResultMetric`
        self[key] = apply_to_collection(value, (torch.Tensor, Metric), fn)

        # when the value was a nested collection, store some metadata
        # to facilate access for later metrics gathering
        if not isinstance(self[key], ResultMetric):
            self[key + '.forked'] = meta.forked
            self[key + '.logger'] = meta.logger
            self[key + '.prog_bar'] = meta.prog_bar
            self[key + '.on_epoch'] = meta.on_epoch
            self[key + '.dataloader_idx'] = meta.dataloader_idx

    def should_reset_tensors(self, hook_name: str) -> bool:
        # reset tensor metrics only when hook_name changed and starting a new iteration over dataloader.
        return (self._current_hook_name != hook_name and self._batch_idx in (None, 0))

    def update_metrics(
        self, hook_name: str, key: str, value: Union[Dict, torch.Tensor], batch_size: torch.Tensor
    ) -> None:

        if self.should_reset_tensors(hook_name):
            # when restarting an new epoch, reset the tensor hooks dynamically.
            self._reset_metrics(hook_name, is_tensor=True)

        # this function is used to call the forward function of ResultMetric object.
        def fn(result_metric, v):
            assert isinstance(v, (torch.Tensor, Metric))
            result_metric(v.to(self.root_device), batch_size.to(self.root_device))
            result_metric.meta.has_reset = False

        apply_to_collections(self[key], value, ResultMetric, fn)

    @staticmethod
    def _get_forward_cache(result_metric: ResultMetric) -> Optional[torch.Tensor]:
        # skip if meta `on_step` is False
        if not result_metric.meta.on_step:
            return

        # extract `ResultMetric` forward cache
        return result_metric._forward_cache.detach()

    @staticmethod
    def _to_item(t: torch.Tensor) -> float:
        return t.item()

    def valid_metrics(self) -> Generator:
        """
        This function is used to iterate over current valid metrics.
        """
        for key, item in self.items():
            # skip when item is None, bool or extra arguments from training_step.
            if item is None or isinstance(item, bool) or key == "extra":
                continue

            # skip when the metrics hasn't been updated.
            elif isinstance(item, ResultMetric) and item.meta.has_reset:
                continue

            yield (key, item)

    def _extract_metadata(self, key: str, result_metric, on_step: bool, suffix: str) -> Tuple:
        """
        This function is used to extract the metadata for `ResultMetric` and `nested ResultMetrics`.
        """

        if isinstance(result_metric, ResultMetric):
            name = result_metric.meta.name
            name_forked = result_metric.meta.forked_name(on_step)
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

        # add dataloader_suffix is provided.
        if dataloader_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dataloader_idx)
            name += dataloader_suffix
            name_forked += dataloader_suffix

        return name, name_forked, logger, prog_bar, metric_on_epoch

    def get_metrics(self, on_step: bool) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = {k: {} for k in MetricSource}

        # either extract `forward_cache` or `computed` from `ResultMetric` objects
        fn = self._get_forward_cache if on_step else self._get_computed_cache

        # select suffix
        suffix = self.STEP_SUFFIX if on_step else self.EPOCH_SUFFIX

        # iterate over all stored metrics.
        for key, result_metric in self.valid_metrics():

            # extract forward_cache or computed from the ResultMetric
            # ignore when the output of fn is None
            value = apply_to_collection(result_metric, ResultMetric, fn, include_none=False)

            # detect if the value is None. This can be nested.
            is_empty = True

            def is_empty_fn(v):
                nonlocal is_empty
                # update is_empty if any value is not None.
                if v is not None:
                    is_empty = False

            # apply detection.
            # TODO(@tchaton): need to find a way to support NamedTuple
            apply_to_collection(value, object, is_empty_fn, wrong_dtype=(Mapping, Sequence))

            # skip is the value was actually empty.
            if is_empty:
                continue

            # extract metadata
            name, name_forked, logger, prog_bar, metric_on_epoch = self._extract_metadata(
                key, result_metric, on_step, suffix
            )

            # populate logging metrics
            if logger:
                metrics[MetricSource.LOG][name_forked] = value

            # populate callback metrics
            # callback metrics don't take `_step` forked metrics.
            if self.training or metric_on_epoch and not on_step:
                metrics[MetricSource.CALLBACK][name] = value
                metrics[MetricSource.CALLBACK][name_forked] = value

            # populate progress_bar metrics. By default, the value should be converted to a float.
            if prog_bar:
                value = apply_to_collection(value, torch.Tensor, self._to_item, include_none=False)
                metrics[MetricSource.PBAR][name_forked] = value

        return metrics

    def get_batch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.get_metrics(on_step=True)

    @staticmethod
    def _get_computed_cache(result_metric: ResultMetric) -> Optional[torch.Tensor]:
        # skip if meta.on_epoch is False
        if not result_metric.meta.on_epoch:
            return

        # perform reduction is not done alrady
        if not result_metric._computed:
            result_metric.compute()

        # extract computed from ResultMetric.
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

    def _reset_metrics(self, hook_name: str = None, is_tensor: Optional[bool] = None) -> None:
        """Call at the end of epoch to reset all results provided as `Metric` or `tensor`"""

        def reset_fn(item: ResultMetric) -> None:
            nonlocal hook_name
            nonlocal is_tensor
            if is_tensor is None or item.meta.is_tensor == is_tensor:
                if isinstance(hook_name, str) and hook_name != item.meta.fx:
                    return
                item.reset()

        apply_to_collection(dict(self.items()), ResultMetric, reset_fn)

    def reset_metrics(self):
        self._reset_metrics(is_tensor=False)
        self.on_epoch_end_reached = False
        self._current_hook_name = None

    def reset(self):
        """
        This function is used to reset entirely the ResultCollection
        """
        self._reset_metrics()
        self.on_epoch_end_reached = False
        self._current_hook_name = None

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

            # ResultMeta is used as a placeholder for making re-loading simpler
            return ResultMeta(**state)

        return {k: apply_to_collection(v, ResultMetric, get_state_dict) for k, v in self.items()}

    def load_from_state_dict(self, state_dict: Dict[str, Any], metrics: Dict[str, Metric]):

        def to_result_metric(item: ResultMeta) -> Dict[str, Any]:
            # create a new ResultMetric
            result_metric = ResultMetric(item["meta"])
            # update its state
            result_metric.__dict__.update(item)
            # move result_metric to root_device
            return result_metric.to(self.root_device)

        # transform ResultMeta into ResultMetric
        state_dict = {k: apply_to_collection(v, ResultMeta, to_result_metric) for k, v in state_dict.items()}

        # add the state_dict as new key-value into self
        for k, v in state_dict.items():
            self[k] = v

        if metrics:

            # the metric reference are lost during serialization and
            # they need to be set back during loading

            def re_assign_metric(item):
                nonlocal metrics
                lightning_attribute_name = item.meta.lightning_attribute_name
                if isinstance(lightning_attribute_name, str) and lightning_attribute_name in metrics:
                    item.value = metrics[lightning_attribute_name]

            apply_to_collection(dict(self.items()), ResultMetric, re_assign_metric)

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        # can't deepcopy tensors with grad_fn
        minimize = d.get('_minimize')
        if minimize is not None:
            d['_minimize'] = minimize.detach()
        return d
