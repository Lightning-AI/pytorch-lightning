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
import numbers
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple, Union

import torch
from torch import Tensor, tensor
from torchmetrics import Metric

from pytorch_lightning.utilities.distributed import sync_ddp_if_available, tpu_distributed
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class DefaultMetricsKeys(LightningEnum):

    CALLBACK_METRICS = "callback_metrics"
    PBAR_METRICS = "pbar_metrics"
    LOG_METRICS = "log_metrics"


class Result:
    pass


@dataclass
class Metadata:

    hook_name: str
    name: str
    on_prog_bar: bool
    on_logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: Callable = torch.mean
    tbptt_reduce_fx: Callable = torch.mean
    tbptt_pad_token: int = 0
    dataloader_idx: Optional[int] = None
    is_tensor: bool = True

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


class ResultMetric(Metric):

    def __init__(self, metatata: Metadata):
        super().__init__()

        self.meta = metatata
        if self.meta.is_tensor:
            self.add_state("values", [])
            self.add_state("batch_sizes", [])

    def update(self, value: Union[torch.Tensor, Metric], batch_size: int) -> None:
        if self.meta.is_tensor:
            self.values.append(value)
            self.batch_sizes.append(batch_size)
        else:
            self.value = value

    def compute(self) -> torch.Tensor:
        if self.meta.is_tensor:
            if self.reduce_fx == torch.mean:
                return (tensor(self.values) * tensor(self.batch_sizes)).sum() / sum(self.batch_sizes)
            elif self.reduce_fx == torch.max:
                return max(self.values)
            else:
                raise MisconfigurationException("Only mean, max are supported.")
        else:
            return self.value.compute()


class ResultCollection(dict):

    def __init__(self) -> None:
        super().__init__()
        self.minimize: Optional[Tensor] = None
        self.on_epoch_end_reached: bool = False
        self.default_metrics = {
            DefaultMetricsKeys.CALLBACK_METRICS: {},
            DefaultMetricsKeys.PBAR_METRICS: {},
            DefaultMetricsKeys.LOG_METRICS: {},
        }

    @property
    def metrics_fn(self):
        return self.get_epoch_metrics if self.on_epoch_end_reached else self.get_batch_metrics

    @staticmethod
    def extract_batch_size(batch: Any) -> int:
        try:
            return ResultCollection._extract_batch_size(batch)
        except RecursionError:
            return 1

    @staticmethod
    def _extract_batch_size(batch: Any) -> int:
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
            size = ResultCollection._extract_batch_size(sample)
        elif isinstance(batch, Iterable):
            sample = next(iter(batch), 1)
            size = ResultCollection._extract_batch_size(sample)
        else:
            size = 1
        return size

    @staticmethod
    def _sync(
        value,
        sync_fn: Optional[Callable] = None,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        device: torch.device = None,
    ):
        """Sync across workers when using distributed training"""
        if not isinstance(value, (torch.Tensor, numbers.Number)):
            return value

        sync_fn = sync_fn or sync_ddp_if_available
        dist_available = torch.distributed.is_available() and torch.distributed.is_initialized() or tpu_distributed()
        if not sync_dist or not dist_available:
            return value

        # TODO: Find a way to make the reduction only once, so we don't need to clone.
        if isinstance(value, torch.Tensor):
            value = value.clone()
        else:
            value = torch.tensor(value, device=device, dtype=torch.float)
        return sync_fn(value, group=sync_dist_group, reduce_op=sync_dist_op)

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
        tbptt_reduce_fx: Callable = torch.mean,
        tbptt_pad_token: int = 0,
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        sync_fn: Callable = None,
        dataloader_idx: Optional[int] = None,
        device: torch.device = None,
        batch_size: Optional[int] = None,
    ):
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        # TODO: should this be in the caller?
        value = self._sync(
            value,
            sync_fn=sync_fn,
            sync_dist=sync_dist,
            sync_dist_op=sync_dist_op,
            sync_dist_group=sync_dist_group,
            device=device,
        )

        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        if batch_size is None:
            raise MisconfigurationException("`batch_size` should be provided.")

        storage_key = f"{hook_name}.{name}"

        if storage_key not in self:
            result = ResultMetric(
                Metadata(
                    hook_name=hook_name,
                    name=name,
                    on_prog_bar=prog_bar,
                    on_logger=logger,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    reduce_fx=reduce_fx,
                    tbptt_reduce_fx=tbptt_reduce_fx,
                    tbptt_pad_token=tbptt_pad_token,
                    dataloader_idx=dataloader_idx,
                )
            )
            self[storage_key] = result

        self[storage_key](value, batch_size)

    def get_batch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = deepcopy(self.default_metrics)

        for result_metric in self.values():
            if not result_metric.meta.on_step:
                continue

            foward_cache: torch.Tensor = result_metric._forward_cache.detach()
            name: str = result_metric.meta.name
            name_forked: str = result_metric.meta.forked_step_name

            if result_metric.meta.on_prog_bar:
                metrics[DefaultMetricsKeys.PBAR_METRICS][name_forked] = foward_cache

            if result_metric.meta.on_logger:
                metrics[DefaultMetricsKeys.LOG_METRICS][name_forked] = foward_cache

            metrics[DefaultMetricsKeys.CALLBACK_METRICS][name] = foward_cache

        return metrics

    def get_batch_log_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to log at the end of the batch"""
        return self.get_batch_metrics()[DefaultMetricsKeys.LOG_METRICS]

    def get_batch_pbar_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to include in the progress_bar at the end of the batch"""
        return self.get_batch_metrics()[DefaultMetricsKeys.PBAR_METRICS]

    def get_batch_callback_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics for the callbacks at the end of the batch"""
        return self.get_batch_metrics()[DefaultMetricsKeys.CALLBACK_METRICS]

    def get_epoch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = deepcopy(self.default_metrics)

        for result_metric in self.values():
            if not result_metric.meta.on_epoch:
                continue

            if not result_metric._computed:
                result_metric.compute()

            computed: torch.Tensor = result_metric._computed.detach()
            name: str = result_metric.meta.name
            name_forked: str = result_metric.meta.forked_epoch_name

            if result_metric.meta.on_prog_bar:
                metrics[DefaultMetricsKeys.PBAR_METRICS][name_forked] = computed

            if result_metric.meta.on_logger:
                metrics[DefaultMetricsKeys.LOG_METRICS][name_forked] = computed

            metrics[DefaultMetricsKeys.CALLBACK_METRICS][name] = computed

        return metrics

    def get_epoch_log_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to log at the end of the epoch"""
        return self.get_epoch_metrics()[DefaultMetricsKeys.LOG_METRICS]

    def get_epoch_pbar_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to include in the progress_bar at the end of the epoch"""
        return self.get_epoch_metrics()[DefaultMetricsKeys.PBAR_METRICS]

    def get_epoch_callback_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics for the callbacks at the end of the epoch"""
        return self.get_epoch_metrics()[DefaultMetricsKeys.CALLBACK_METRICS]

    def to(self, *args, **kwargs) -> 'ResultCollection':
        """Move all data to the given device."""
        for item in self.values():
            if isinstance(item, ResultMetric):
                item.to(*args, **kwargs)
        return self

    def cpu(self) -> 'Result':
        """Move all data to CPU."""
        return self.to(device="cpu")

    def reset(self) -> None:
        """Call at the end of epoch to reset all metric objects"""
        for item in self.values():
            if isinstance(item, Metric):
                item.reset()
