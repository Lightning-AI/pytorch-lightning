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
from typing import Any, Optional

import torch

from pytorch_lightning.overrides.torch_distributed import broadcast_object_list
from pytorch_lightning.utilities import _SMDIST_AVAILABLE
from pytorch_lightning.utilities.distributed import group as _group

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.distributed as sm_dist


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any, group=_group.WORLD):
        # always wrap into a list so list can be brodcasted.
        obj = [obj]

        if self.rank != 0:
            obj = [None] * len(obj)

        broadcast_object_list(obj, 0, group=group or _group.WORLD)

        return obj[0]


class SMLightningDistributed(LightningDistributed):

    def broadcast(self, obj: Any, group=_group.WORLD):
        if self.rank == 0:
            self._emit(obj, group)
        else:
            obj = self._receive(group)
        return obj

    def _broadcast(self, tensor: torch.Tensor, src: int, group: Optional[Any] = None):
        if group is None:
            return sm_dist.broadcast(tensor, src=src)
        return sm_dist.broadcast(tensor, src=0, group=group)

    def _emit(self, obj: Any, group=_group.WORLD):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.tensor([len(data)]).long().to(self.device)
        self._broadcast(length_tensor, src=0, group=group)
        data_tensor = torch.ByteTensor(data).to(self.device)
        self._broadcast(data_tensor, src=0, group=group)

    def _receive(self, group=_group.WORLD):
        length_tensor = torch.tensor([0]).long().to(self.device)
        self._broadcast(length_tensor, src=0, group=group)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8).to(self.device)
        self._broadcast(data_tensor, src=0, group=group)
        buffer = io.BytesIO(data_tensor.cpu().numpy())
        obj = torch.load(buffer)
        return obj
