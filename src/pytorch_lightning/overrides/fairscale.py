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
from typing import List

from lightning_utilities.core.imports import package_available
from torch.optim import Optimizer

from lightning_fabric.plugins import Precision
from lightning_fabric.utilities.imports import _IS_WINDOWS

_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and package_available("fairscale")

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS
else:
    OSS = object


def _reinit_optimizers_with_oss(optimizers: List[Optimizer], precision: Precision, num_nodes: int) -> List["OSS"]:
    for x, optimizer in enumerate(optimizers):
        if not isinstance(optimizer, OSS):
            optim_class = type(optimizer)
            zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
            is_fp16 = precision.precision == "16"
            # For multi-node training, compressing the model shards in fp16 before broadcasting
            # improves performance. When using PyTorch AMP, it will not degrade
            # the model performance.
            zero_optimizer.broadcast_fp16 = is_fp16 and num_nodes > 1
            optimizers[x] = zero_optimizer
            del optimizer
    return optimizers
