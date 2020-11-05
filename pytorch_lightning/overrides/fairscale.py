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
from typing import Dict, Any, List, Tuple, Union
from torch import nn
import torch
from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel, Gatekeeper
from fairscale.optim import OSS

from pytorch_lightning.utilities import rank_zero_warn


class LightningOSS(OSS):

    def state_dict(self) -> Dict[str, Any]:
        """Return the last known global optimizer state, which consist of a list of the shards.

        .. warning:
            If the state has not been consolidated, this returns a shard's worth, not the global state.

        .. warning:
            Returning the global state is limited to the replica which was responsible for the consolidation.
            The state may also not be up to date, depending on when `consolidate_state_dict` was last called.
        """

        if len(self._all_states) == 0:
            rank_zero_warn("Optimizer state has not been consolidated. Returning the local state")
            rank_zero_warn("Please call `consolidate_state_dict()` beforehand if you meant to save the global state")
            state_dict = self.local_state_dict()
            state_dict["local_state_dict"] = True
            return state_dict

        # Flatten the param_groups, save the partition which logs the rank <> shard correspondence
        partition: List[Tuple[int, int]] = []
        param_groups: List[Dict[Any, Any]] = []

        start = 0
        for i, s in enumerate(self._all_states):
            param_groups.extend(s["param_groups"])
            end = start + len(s["param_groups"])
            partition.append((start, end))
            start = end

        return {
            "state": [s["state"] for s in self._all_states],
            "param_groups": param_groups,
            "partition": partition,
            "local_state_dict": False,
        }


class LightningShardedDataParallel(ShardedDataParallel):
    def __init__(
            self,
            base_model: nn.Module,
            sharded_optimizer: Union[OSS, List[OSS]],
            process_group: Any = None,
            broadcast_buffers: bool = True,
            reduce_buffer_size: int = 2 ** 19,
    ):
        super().__init__(
            base_model=base_model,
            sharded_optimizer=sharded_optimizer,
            process_group=process_group,
            broadcast_buffers=broadcast_buffers,
            reduce_buffer_size=reduce_buffer_size
        )
        self.module = base_model

    def forward(self, *inputs, **kwargs):
        batch, batch_idx = inputs

        if self.broadcast_buffers:
            self.sync_buffers()

        # Reset all the grad reduce and bucket state flags
        self._grad_to_be_reduced = [True for _ in self._grad_to_be_reduced]
        for sharded_optimizer in self.sharded_optimizers:
            for device, per_rank_params in sharded_optimizer.per_device_params.items():
                for r in range(self.world_size):
                    self._bucket_state[sharded_optimizer][device][r] = (
                        0,
                        self._bucket_state[sharded_optimizer][device][r][1],
                    )

        # All inputs need to required_grad for autograd to properly track the first dispatch layer
        for i in batch:
            if isinstance(i, torch.Tensor) and i.is_floating_point():
                i.requires_grad = True
        # Register the model dispatch in the autograd graph
        batch = Gatekeeper.apply(self._reduce_work_handles, *batch)
        if self.base_model.training:
            outputs = self.base_model.training_step(batch, batch_idx, **kwargs)
        elif self.base_model.testing:
            outputs = self.base_model.test_step(batch, batch_idx, **kwargs)
        else:
            outputs = self.base_model.validation_step(batch, batch_idx, **kwargs)
        return outputs
