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
import itertools
from typing import Any, Iterator, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DistributedSampler, Sampler

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase


class LightningDistributedModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: LightningModule):
        """
        Wraps the user's LightningModule and redirects the forward call to the appropriate
        method, either ``training_step``, ``validation_step``, ``test_step`` or ``predict``.
        This class is used in combination with :class:`~torch.nn.parallel.DistributedDataParallel` as
        shown in the example.

        Example:

            ddp_model = torch.nn.parallel.DistributedDataParallel(
                module=LightningDistributedModule(lightning_module),
                device_ids=[local_rank],
                ...
            )

        Args:
            pl_module: the model to wrap

        """
        super().__init__(pl_module)


def _find_tensors(obj):  # pragma: no-cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


# In manual_optimization, we need to call reducer prepare_for_backward.
# Note: Keep track of Pytorch DDP and update if there is a change
# https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/parallel/distributed.py#L626-L638
def prepare_for_backward(model: DistributedDataParallel, output: Any):
    if torch.is_grad_enabled() and model.require_backward_grad_sync:
        model.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        if model.find_unused_parameters:
            model.reducer.prepare_for_backward(list(_find_tensors(output)))
        else:
            model.reducer.prepare_for_backward([])
    else:
        model.require_forward_param_sync = False


# Taken from https://github.com/jpuigcerver/PyLaia/blob/v1.0.0/laia/data/unpadded_distributed_sampler.py
class UnrepeatedDistributedSampler(DistributedSampler):
    """
    This sampler doesn't repeat data, instead it
    allows the number of batches per process to be off-by-one between the ranks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))
        self.total_size = len(self.dataset)
        # If any process has at least one batch, every other process needs to
        # have at least one batch, or the DistributedDataParallel could lock up.
        assert self.num_samples >= 1 or self.total_size == 0

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class IndexBatchSampler(BatchSampler):
    """
    This class is used to capture the batch indices.
    """

    def __init__(self, batch_sampler: BatchSampler) -> None:
        self.batch_sampler = batch_sampler
        self.batch_indices: Optional[List[int]] = None

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batch_sampler:
            self.batch_indices = batch
            yield batch

    @property
    def drop_last(self) -> bool:
        return self.batch_sampler.drop_last

    @property
    def batch_size(self) -> int:
        return self.batch_sampler.batch_size

    @property
    def sampler(self) -> Sampler:
        return self.batch_sampler.sampler
