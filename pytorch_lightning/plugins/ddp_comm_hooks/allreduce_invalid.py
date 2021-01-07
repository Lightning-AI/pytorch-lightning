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
from functools import partial
from typing import Dict

import torch
import torch.distributed as torch_distrib
import torch.distributed as dist

from pytorch_lightning.utilities import LightningEnum, TORCH_GREATER_EQUAL_1_7_0

init_allreduce_hook_with_invalid_tensors = None
allreduce_hook_with_invalid_tensors = None
update_allreduce_hook_with_invalid_tensors = None
allreduce_hook_with_invalid_tensors_err_message = "Hint: `DDP_COMM_HOOKS` are introduced in PyTorch 1.7. Update PyTorch to use this feature"


if TORCH_GREATER_EQUAL_1_7_0:

    def init_allreduce_hook_with_invalid_tensors(trainer):
        def init_hook(state, accumulate_grad_batches: int, world_size: int):
            state["is_invalid"] = []
            state["should_accumulate"] = None
            state["accumulate_grad_batches"] = accumulate_grad_batches
            state["world_size"] = world_size

        return partial(init_hook,
                       accumulate_grad_batches=trainer.accumulate_grad_batches,
                       world_size=trainer.world_size)

    def update_allreduce_hook_with_invalid_tensors(trainer, state):
        state["should_accumulate"] = trainer.train_loop.should_accumulate()

    def allreduce_hook_with_invalid_tensors(state: Dict, bucket: torch_distrib._GradBucket):
        """
        This DDP communication all_reduce hook enables to train with NaN or Inf losses by detecting invalid values and zeroing them for the optimizer step
        """
        group_to_use = torch_distrib.group.WORLD
        tensor = bucket.get_tensors()[0]

        is_invalid = torch.isnan(tensor).any() or not torch.isfinite(tensor).any()
        is_invalid = torch.tensor(int(is_invalid), device=tensor.device)

        if is_invalid:
            tensor = torch.zeros_like(tensor, device=tensor.device)

        if state["should_accumulate"]:
            state["is_invalid"].append(is_invalid)
            return [tensor]

        state["is_invalid"].append(is_invalid)
        state["is_invalid"] = torch.vstack(state["is_invalid"]).sum()

        dist.all_reduce(
            state["is_invalid"],
            group=group_to_use,
            async_op=False,
            op=torch_distrib.ReduceOp.SUM
        )

        fut = dist.all_reduce(
            tensor, group=group_to_use, async_op=True, op=torch_distrib.ReduceOp.SUM
        ).get_future()

        def then_callback(fut):
            tensor = fut.value()[0]

            if not state["should_accumulate"]:
                total_invalid_batches = state["is_invalid"]
                total_batches = state["accumulate_grad_batches"] * state["world_size"]
                number_seen_batches = max(total_batches - total_invalid_batches, 1)

                tensor /= number_seen_batches

                state["is_invalid"] = []
                state["should_accumulate"] = None

            return [tensor]

        return fut.then(then_callback)


class ALLREDUCE_INVALID(LightningEnum):
    INIT_STATE_HOOK = init_allreduce_hook_with_invalid_tensors
    HOOK = allreduce_hook_with_invalid_tensors
    ERR_MSG = allreduce_hook_with_invalid_tensors_err_message
    ON_BEFORE_BACKWARD_ENGINE_EXECUTION = update_allreduce_hook_with_invalid_tensors
