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
from typing import Any

from torch.distributed.distributed_c10d import GroupMember

from pytorch_lightning.distributed.dist_utils import broadcast_object_list


class LightningDistributed:

    def __init__(self, rank=None, device=None):
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any, group=None):
        is_list = isinstance(obj, list)

        if not is_list:
            obj = [obj]

        if self.rank != 0:
            obj = [None for _ in range(len(obj))]

        broadcast_object_list(obj, 0, group=group or GroupMember.WORLD)

        if not is_list:
            obj = obj[0]

        return obj
