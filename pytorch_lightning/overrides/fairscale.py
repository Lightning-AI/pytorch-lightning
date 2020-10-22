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
from typing import Dict, Any, List, Union, Tuple

from fairscale.nn.data_parallel.sharded_ddp import ModelDispatch, DispatchLayer
from fairscale.optim import OSS
from torch import nn

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


class LightningModelDispatch(ModelDispatch):
    def forward(self, *inputs, **kwargs):  # type: ignore
        if self.broadcast_model_buffers:
            self.sync_buffers(non_blocking=False)
        if self.base_model.training:
            output = self.base_model.training_step(*inputs, **kwargs)
        elif self.base_model.testing:
            output = self.base_model.test_step(*inputs, **kwargs)
        else:
            output = self.base_model.validation_step(*inputs, **kwargs)

        return output


class LightningDispatchLayer(DispatchLayer):
    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore
        ctx.model.dispatch_grads()
        return (None, *grad_outputs)


class LightningShardedDataParallel(nn.Module):
    """
    Wrap the model, and reduce the gradients to the right rank after the backward pass.

    - the partition is given by the sharded optimizer
    - wrap the base model with a model which knows where to reduce each gradient
    - add an autograd function which calls the model grad dispatch on the way back

     Args:
        base_model (nn.Module):
            model to be wrapped
        sharded_optimizer (OSS, or list of OSS):
            the sharded optimizer(s) which will decide the gradient partitioning
    Keyword Args:
        process_group (torch.nn.Optimizer):
            optimizer to shard (default: SGD)
        process_group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffers (bool):
            whether to broadcast model buffers in between ranks at the beginning of each forward pass
        buffer_size (int):
            the size of the buffer in bits used to batch the small parameter tensors (default 128k).
    """

    def __init__(
            self,
            base_model: nn.Module,
            sharded_optimizer: Union[OSS, List[OSS]],
            process_group: Any = None,
            broadcast_buffers: bool = True,
            buffer_size: int = 2 ** 17,
    ):
        super().__init__()
        self.module = base_model  # Required for training reference
        self.model_dispatch = LightningModelDispatch(
            base_model=base_model,
            sharded_optimizer=sharded_optimizer,
            process_group=process_group,
            broadcast_buffers=broadcast_buffers,
            reference_rank=0,
            buffer_size=buffer_size,
        )

    def forward(self, *inputs, **kwargs):
        batch, batch_idx = inputs
        # All inputs need to required_grad for autograd to properly track the first dispatch layer
        for i in batch:
            if i.is_floating_point():
                i.requires_grad = True
        # Register the model dispatch in the autograd graph
        batch = LightningDispatchLayer.apply(self.model_dispatch, *batch)
        # Normal model FW
        outputs = self.model_dispatch(batch, batch_idx, **kwargs)
        return outputs
