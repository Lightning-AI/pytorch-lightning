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
from typing import Any, List, Union

from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
from fairscale.optim import OSS
from torch import nn


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

        if self.base_model.training:
            outputs = self.base_model.training_step(*inputs, **kwargs)
        elif self.base_model.testing:
            outputs = self.base_model.test_step(*inputs, **kwargs)
        else:
            outputs = self.base_model.validation_step(*inputs, **kwargs)
        return outputs

    def clear_backward_handles(self):
        # Consume the handles, make sure that all the reduces are done before the optimizer can step
        while len(self._reduce_work_handles) > 0:
            wh, callback = self._reduce_work_handles.pop()
            if wh is not None:
                wh.wait()
                callback()
