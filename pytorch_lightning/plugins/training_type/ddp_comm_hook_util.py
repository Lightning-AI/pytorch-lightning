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
# limitations under the License
from typing import Optional

from pytorch_lightning.utilities import (
    _TORCH_GREATER_EQUAL_1_7,
    _TORCH_GREATER_EQUAL_1_9,
    rank_zero_warn,
    rank_zero_info,
)
from torch.nn.parallel.distributed import DistributedDataParallel


def register_ddp_comm_hook(
    ddp_comm_state: Optional[object],
    ddp_comm_hook: Optional[callable],
    ddp_comm_wrapper: Optional[callable],
    model: DistributedDataParallel,
    is_single_process_single_device: bool,
):
    # register DDP comm hook: https://pytorch.org/docs/master/ddp_comm_hooks.html
    if ddp_comm_hook is None:
        rank_zero_info("No DDP comm hook is provided, skipping.")
        return
    if not _TORCH_GREATER_EQUAL_1_7:
        rank_zero_warn(
            "Not registering DDP comm hook. "
            "To use communication hooks, please use PyTorch version at least 1.7.0."
        )
        return
    if not is_single_process_single_device:
        rank_zero_warn(
            "Not registering DDP comm hook. "
            "To use communication hooks, must be single process single device, see "
            "https://github.com/pytorch/pytorch/blob/e6779d4357ae94cc9f9fedb83a87eb6126016769/torch/nn/parallel/distributed.py#L1035"
        )
    if ddp_comm_wrapper is not None:
        if not _TORCH_GREATER_EQUAL_1_9:
            rank_zero_warn(
                "Not applying DDP comm wrapper. "
                "To use communication wrapper, please use PyTorch version at least 1.9.0."
            )
        else:
            rank_zero_info(
                "DDP comm wrapper is provided, apply ddp_comm_wrapper(ddp_comm_hook)."
            )
            ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)

    rank_zero_info("Registering DDP comm hook.")
    model.register_comm_hook(
        state=ddp_comm_state,
        hook=ddp_comm_hook,
    )
