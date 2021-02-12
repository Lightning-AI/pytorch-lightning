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
import io
from typing import Any

import torch
from torch import distributed as torch_distrib

from pytorch_lightning.utilities import _GROUP_AVAILABLE

WORLD = None
if _GROUP_AVAILABLE:
    from torch.distributed import group
    WORLD = group.WORLD


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any, group=WORLD):
        if self.rank == 0:
            self._emit(obj, group)
        else:
            obj = self._receive(group)
        return obj

    def _broadcast(self, tensor, src=0, group=WORLD):
        if group is None:
            return torch_distrib.broadcast(tensor, src=src)
        return torch_distrib.broadcast(tensor, src=0, group=group)

    def _emit(self, obj: Any, group=WORLD):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.tensor([len(data)]).long().to(self.device)
        self._broadcast(length_tensor, src=0, group=group)
        data_tensor = torch.ByteTensor(data).to(self.device)
        self._broadcast(data_tensor, src=0, group=group)

    def _receive(self, group=WORLD):
        length_tensor = torch.tensor([0]).long().to(self.device)
        self._broadcast(length_tensor, src=0, group=group)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
        self._broadcast(data_tensor, src=0, group=group)
        buffer = io.BytesIO(data_tensor.cpu().numpy())
        obj = torch.load(buffer)
        return obj
