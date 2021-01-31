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
import torch

from pytorch_lightning.core.step_result import Result
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin


class DDP2Plugin(DDPPlugin):

    def setup(self, model):
        self._model = model
        # set the task idx
        self.task_idx = self.cluster_environment.local_rank()
        # the difference to DDP is that we don't call children processes here

    def reduce(self, output, *args, **kwargs):
        if isinstance(output, Result):
            output.dp_reduce()

        elif isinstance(output, torch.Tensor):
            output = output.mean()

        return output

    @property
    def root_device(self):
        return self.parallel_devices[0]

    def model_to_device(self):
        # no need to do anything when model is wrapped in torch.nn.DataParallel
        pass

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=self.num_nodes, rank=self.global_rank)
        return distributed_sampler_kwargs

    def set_world_ranks(self):
        self.local_rank = self.task_idx
        self.node_rank = self.cluster_environment.node_rank()
        self.global_rank = self.node_rank
        self.world_size = self.num_nodes
