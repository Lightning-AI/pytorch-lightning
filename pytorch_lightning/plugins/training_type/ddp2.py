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

from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.types import _METRIC_COLLECTION


class DDP2Plugin(DDPPlugin):
    """ DDP2 behaves like DP in one node, but synchronization across nodes behaves like in DDP."""

    @property
    def global_rank(self) -> int:
        return self.node_rank

    @property
    def world_size(self) -> int:
        return self.num_nodes

    def setup(self, model):
        self._model = model
        # set the task idx
        self.task_idx = self.cluster_environment.local_rank()
        # the difference to DDP is that we don't call children processes here

    def reduce(self, collection: _METRIC_COLLECTION, *args, **kwargs) -> _METRIC_COLLECTION:
        """
        Reduces a collection of tensors from all processes. It can be applied to just a single tensor.
        In DDP2, the reduction here is only across local devices within the node.

        Args:
            collection: The collection of tensors to sync and reduce.
            *args: ignored for DDP2
            **kwargs: ignored for DDP2

        Return:
            Reduced tensor values or the same value if it was not or did not contain a tensor.
        """

        def mean(t: torch.Tensor) -> torch.Tensor:
            original_dtype = t.dtype
            return t.float().mean().to(original_dtype)

        return apply_to_collection(collection, torch.Tensor, mean)

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

    @property
    def _is_single_process_single_device(self) -> bool:
        return False

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank)
        self.cluster_environment.set_world_size(self.num_nodes)
