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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import Tensor
from torchmetrics import Metric

from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _METRIC


class DefaultMetricsKeys(LightningEnum):
    CALLBACK = "callback"
    PBAR = "pbar"
    LOG = "log"


# TODO: remove
class Result:
    pass


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

    def __init__(self, metadata: Metadata) -> None:
        super().__init__()
        self.meta = metadata
        if self.meta.is_tensor:
            # TODO: dist_reduce_fx?
            self.add_state("values", [])
            self.add_state("batch_sizes", [])

    def update(self, value: _METRIC, batch_size: Optional[int] = None) -> None:
        if self.meta.is_tensor:
            self.values.append(value)
            if batch_size is None:
                batch_size = self.extract_batch_size(value)
            self.batch_sizes.append(batch_size)
        else:
            self.value = value

    def compute(self) -> torch.Tensor:
        if self.meta.is_tensor:
            if self.reduce_fx == torch.mean:
                return (torch.tensor(self.values) * torch.tensor(self.batch_sizes)).sum() / sum(self.batch_sizes)
            elif self.reduce_fx == torch.max:
                return max(self.values)
            else:
                raise MisconfigurationException("Only mean, max are supported.")
        else:
            return self.value.compute()

    @staticmethod
    def extract_batch_size(batch: Any) -> int:
        try:
            return ResultMetric._extract_batch_size(batch)
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
            size = ResultMetric._extract_batch_size(sample)
        elif isinstance(batch, Iterable):
            sample = next(iter(batch), 1)
            size = ResultMetric._extract_batch_size(sample)
        else:
            size = 1
        return size


class ResultCollection(dict):

    def __init__(self) -> None:
        super().__init__()
        self.minimize: Optional[Tensor] = None
        self.on_epoch_end_reached: bool = False
        self.default_metrics = {k: {} for k in DefaultMetricsKeys}

    @property
    def metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return self.get_epoch_metrics() if self.on_epoch_end_reached else self.get_batch_metrics()

    @property
    def minimize(self) -> Optional[Tensor]:
        return self.get('minimize', None)

    @minimize.setter
    def minimize(self, val: Optional[torch.Tensor]) -> None:
        if val is not None:
            if not isinstance(val, Tensor):
                raise ValueError(f"`Result.minimize` must be a `torch.Tensor`, found: {val}")
            if val.grad_fn is None:
                raise RuntimeError("`Result.minimize` must have a `grad_fn`")
        self['minimize'] = val

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
    ):
        """See :meth:`~pytorch_lightning.core.lightning.LightningModule.log`"""
        # no metrics should be logged with graphs
        if not enable_graph and isinstance(value, torch.Tensor):
            value = value.detach()

        if isinstance(value, torch.Tensor) and value.device.type == "xla":
            value = value.cpu()

        key = f"{hook_name}.{name}"

        if key not in self:
            result = ResultMetric(
                Metadata(
                    fx=hook_name,
                    name=name,
                    prog_bar=prog_bar,
                    logger=logger,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    reduce_fx=reduce_fx,
                    dataloader_idx=dataloader_idx,
                )
            )
            self[key] = result

        self[key](value, batch_size)

    def get_batch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        # TODO: do we need deepcopy?
        metrics = deepcopy(self.default_metrics)

        for result_metric in self.values():
            if not result_metric.meta.on_step:
                continue

            foward_cache: torch.Tensor = result_metric._forward_cache.detach()

            name_forked = result_metric.meta.forked_step_name
            if result_metric.meta.prog_bar:
                metrics[DefaultMetricsKeys.PBAR][name_forked] = foward_cache
            if result_metric.meta.logger:
                metrics[DefaultMetricsKeys.LOG][name_forked] = foward_cache
            metrics[DefaultMetricsKeys.CALLBACK][result_metric.meta.name] = foward_cache

        return metrics

    # TODO: add_dataloader_idx?
    def get_batch_log_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to log at the end of the batch"""
        return self.get_batch_metrics()[DefaultMetricsKeys.LOG]

    def get_batch_pbar_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to include in the progress_bar at the end of the batch"""
        return self.get_batch_metrics()[DefaultMetricsKeys.PBAR]

    def get_batch_callback_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics for the callbacks at the end of the batch"""
        return self.get_batch_metrics()[DefaultMetricsKeys.CALLBACK]

    def get_epoch_metrics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        metrics = deepcopy(self.default_metrics)

        for result_metric in self.values():
            if not result_metric.meta.on_epoch:
                continue

            if not result_metric._computed:
                result_metric.compute()

            computed: torch.Tensor = result_metric._computed.detach()

            name_forked: str = result_metric.meta.forked_epoch_name
            if result_metric.meta.prog_bar:
                metrics[DefaultMetricsKeys.PBAR][name_forked] = computed
            if result_metric.meta.logger:
                metrics[DefaultMetricsKeys.LOG][name_forked] = computed
            metrics[DefaultMetricsKeys.CALLBACK][result_metric.meta.name] = computed

        return metrics

    def get_epoch_log_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to log at the end of the epoch"""
        return self.get_epoch_metrics()[DefaultMetricsKeys.LOG]

    def get_epoch_pbar_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics to include in the progress_bar at the end of the epoch"""
        return self.get_epoch_metrics()[DefaultMetricsKeys.PBAR]

    def get_epoch_callback_metrics(self, add_dataloader_idx: bool = False) -> Dict[str, torch.Tensor]:
        """Gets the metrics for the callbacks at the end of the epoch"""
        return self.get_epoch_metrics()[DefaultMetricsKeys.CALLBACK]

    def to(self, *args, **kwargs) -> 'ResultCollection':
        """Move all data to the given device."""
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, Metric)):
                self[k] = v.to(*args, **kwargs)
        return self

    def cpu(self) -> 'ResultCollection':
        """Move all data to CPU."""
        return self.to(device="cpu")

    def reset(self) -> None:
        """Call at the end of epoch to reset all metric objects"""
        for item in self.values():
            if isinstance(item, Metric):
                item.reset()
