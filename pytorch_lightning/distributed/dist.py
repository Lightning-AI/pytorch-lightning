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

import torch.distributed

from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.distributed import group as _group


class LightningDistributed:
    """
    .. deprecated:: v1.5
        This class is deprecated in v1.5 and will be removed in v1.7.
        The broadcast logic will be moved to the :class:`DDPStrategy` and :class`DDPSpawnStrategy` classes.

    """

    def __init__(self, rank=None, device=None):
        rank_zero_deprecation(
            "LightningDistributed is deprecated in v1.5 and will be removed in v1.7."
            "Broadcast logic is implemented directly in the :class:`Strategy` implementations."
        )
        self.rank = rank
        self.device = device

    def broadcast(self, obj: Any, group=_group.WORLD):
        # always wrap into a list so it can be broadcasted.
        obj = [obj]

        if self.rank != 0:
            obj = [None] * len(obj)

        torch.distributed.broadcast_object_list(obj, 0, group=group or _group.WORLD)

        return obj[0]
