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
r"""
MemoryFormat
===============

changes the model memory format
"""

import torch
from typing import Any, MutableSequence, Optional, Sequence

import pytorch_lightning as pl

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class MemoryFormat(Callback):
    """The `MemoryFormat` callback changes the model memory format to `torch.channels_last` 
    before training starts and returns the original when it ends.

    <https://\\pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_.

    Setting the memory format channels_last usually improves GPU utilization.

    Runs on setup, so it can set the memory format before the model is DDP wrapped.
    """

    def __init__(self, memory_format: torch.memory_format = torch.channels_last, convert_input: bool = False):
        self.memory_format = memory_format
        self.convert_input = convert_input

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self.memory_format in (torch.channels_last, torch.channels_last_3d) and not self.has_layer_benefiting_from_channels_last(pl_module):
            rank_zero_warn(f"model does not have any layers benefiting from {self.memory_format} format", category=RuntimeWarning)

        pl_module.to(memory_format=self.memory_format)

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        pl_module.to(memory_format=torch.contiguous_format)

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        if not self.convert_input:
            return
        
        if not isinstance(batch, MutableSequence):
            rank_zero_warn(f"batch is not a MutableSequence, cannot convert input to {self.memory_format}", category=RuntimeWarning)
            return

        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                batch[i] = item.to(memory_format=self.memory_format)
    
    benefitial_layers = (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.Conv2d, torch.nn.Conv3d)

    def has_layer_benefiting_from_channels_last(self, model: torch.nn.Module) -> bool:
        return any(isinstance(layer, self.benefitial_layers) for layer in model.modules())