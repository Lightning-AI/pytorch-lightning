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
import torch
from typing import Any
from torch import distributed as torch_distrib


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any):
        if self.rank == 0:
            self._emit(obj)
        else:
            obj = self._receive()
        return obj

    def _emit(self, obj):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.tensor([len(data)]).long().to(self.device)
        length_tensor = torch_distrib.broadcast(length_tensor, src=0)
        data_tensor = torch.ByteTensor(data).to(self.device)
        data_tensor = torch_distrib.broadcast(data_tensor, src=0)

    def _receive(self):
        length_tensor = torch.tensor([0]).long().to(self.device)
        torch_distrib.broadcast(length_tensor, src=0)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
        torch_distrib.broadcast(data_tensor, src=0)
        buffer = io.BytesIO(data_tensor.cpu().numpy())
        obj = torch.load(buffer)
        return obj
