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
from abc import ABC
from typing import Any, List, Optional, Union

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import rank_zero_deprecation


class TrainerDataLoadingMixin(ABC):
    r"""
    .. deprecated:: v1.6
        The `TrainerDataLoadingMixin` class was deprecated in v1.6 and will be removed in v1.8.
    """

    def prepare_dataloader(self, dataloader: Any, shuffle: bool, mode: Optional[RunningStage] = None) -> Any:
        r"""
        .. deprecated:: v1.6
            `TrainerDataLoadingMixin.prepare_dataloader` was deprecated in v1.6
            and will be removed in v1.8.

        This function handles to following functionalities:

        - Injecting a `DistributedDataSampler` into the `DataLoader` if on a distributed environment
        - Wrapping the datasets and samplers into fault-tolerant components
        """
        rank_zero_deprecation(
            "`TrainerDataLoadingMixin.prepare_dataloader` was deprecated in v1.6 and will be removed in v1.8."
        )
        return self._data_connector._prepare_dataloader(dataloader, shuffle, mode)

    def request_dataloader(
        self, stage: RunningStage, model: Optional["pl.LightningModule"] = None
    ) -> Union[DataLoader, List[DataLoader]]:
        r"""
        .. deprecated:: v1.6
            `TrainerDataLoadingMixin.request_dataloader` was deprecated in v1.6
            and will be removed in v1.8.

        Requests a dataloader from the given model by calling dataloader hooks corresponding to the given stage.

        Returns:
            The requested dataloader
        """
        rank_zero_deprecation(
            "`TrainerDataLoadingMixin.request_dataloader` was deprecated in v1.6 and will be removed in v1.8."
        )
        return self._data_connector._request_dataloader(stage)
