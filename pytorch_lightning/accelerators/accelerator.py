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
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.distributed as torch_distrib
from torch.optim import Optimizer

from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.rpc_plugin import RPCPlugin
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.parsing import AttributeDict

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:
    class ReduceOp:
        SUM = None


class Accelerator(object):

    def __init__(self,
                 trainer: Optional = None,
                 cluster_environment: Optional[ClusterEnvironment] = None,
                 ddp_plugin: Optional[DDPPlugin] = None):
        self.trainer = trainer
        self.nickname = None
        self.cluster_environment = cluster_environment
        self.dist = AttributeDict(rank=0, device=None)
        self.ddp_plugin = ddp_plugin

        if trainer is not None:
            self.train_loop = self.trainer.train
            self.validation_loop = self.trainer.run_evaluation
            self.test_loop = self.trainer.run_evaluation

    def setup(self, model):
        pass

    def train(self):
        self.trainer.setup_trainer(self.trainer.model)
        return self.train_or_test()

    def teardown(self):
        # Ensure if necessary all processes are finished
        self.barrier()

    def barrier(self, name: Optional[str] = None):
        pass

    def broadcast(self, obj, src=0):
        return obj

    def train_or_test(self):
        if self.trainer.testing:
            results = self.trainer.run_test()
        else:
            self.trainer.train_loop.setup_training()
            results = self.trainer.train()
        return results

    def batch_to_device(self, batch: Any, device: torch.device):
        model = self.trainer.get_model()
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def training_step_end(self, output):
        return output

    def test_step_end(self, output):
        return output

    def validation_step_end(self, output):
        return output

    def process_dataloader(self, dataloader):
        return dataloader

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        automatic_optimization = self.trainer.train_loop.automatic_optimization

        if not automatic_optimization and self.ddp_plugin is not None:
            # Manually prepare for reduce as user calling backwards manually
            self.ddp_plugin.on_before_manual_backward(self.trainer.model, closure_loss)

        if self.trainer.precision == 16:
            closure_loss = self.trainer.precision_connector.backend.backward(
                closure_loss, optimizer, opt_idx, *args, **kwargs
            )
        else:
            # do backward pass
            model = self.trainer.get_model()
            model.backward(closure_loss, optimizer, opt_idx, *args, **kwargs)

            # once backward has been applied, release graph
            closure_loss = closure_loss.detach()

        if not automatic_optimization and self.ddp_plugin is not None:
            # Manually prepare for reduce as user calling backwards manually
            self.ddp_plugin.on_after_manual_backward(self.trainer.model)
        return closure_loss

    def clip_gradients(self, optimizer, clip_val=None):
        # use the trainer's clip val if none passed
        grad_clip_val = self.trainer.gradient_clip_val
        if clip_val is not None:
            grad_clip_val = clip_val
        grad_clip_val = float(grad_clip_val)

        if grad_clip_val <= 0:
            return
        self._clip_gradients(optimizer, grad_clip_val)

    def _clip_gradients(self, optimizer: Optimizer, grad_clip_val: Union[float, int], norm_type: float = 2.0):
        if self.trainer.amp_backend:
            self.trainer.precision_connector.backend.clip_gradients(grad_clip_val, optimizer, norm_type)
        else:
            model = self.trainer.get_model()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val, norm_type=norm_type)

    def on_train_epoch_end(self, outputs):
        pass

    def on_train_end(self):
        pass

    def early_stopping_should_stop(self, pl_module):
        return self.trainer.should_stop

    def setup_optimizers(self, model):
        if self.trainer.testing:
            return

        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

    def init_ddp_connection(
            self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True
    ) -> None:
        self.ddp_plugin.init_ddp_connection(
            self.trainer,
            self.cluster_environment,
            global_rank,
            world_size,
            is_slurm_managing_tasks,
        )

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        """
        Function to reduce a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to sum.
                Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

        Return:
            reduced value
        """
        raise NotImplementedError()

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        """
        Function to gather a tensor from several distributed processes

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...)
        """
        raise NotImplementedError()

    def optimizer_state(self, optimizer: Optimizer) -> dict:
        """
        Returns state of an optimizer. Allows for syncing/collating optimizer state from processes in custom
        plugins.
        Return:
            Optimizer state dict
        """
        if self.ddp_plugin:
            return self.ddp_plugin.optimizer_state(optimizer)
        return optimizer.state_dict()

    def get_reference_model(self, model) -> LightningModule:
        """
        Override to modify returning base :class:`LightningModule`
        when accessing variable and functions if the accelerator has wrapped the model.

        Example::
            ref_model = accelerator.get_reference_model(model)
            ref_model.training_step(...)

        Args:
            model: Accelerator model.

        Returns: Reference :class:`LightningModule`.

        """
        return model

    def __getstate__(self):
        return {
            'trainer': self.trainer,
            'nickname': self.nickname,
            'cluster_environment': self.cluster_environment,
            'dist': self.dist,
            'ddp_plugin': self.ddp_plugin
        }

    def __setstate__(self, d):
        self.trainer = d['trainer']
        self.nickname = d['nickname']
        self.cluster_environment = d['cluster_environment']
        self.dist = d['dist']
        self.ddp_plugin = d['ddp_plugin']

    def on_save(self, checkpoint):
        return checkpoint

    @property
    def rpc_enabled(self):
        return self.ddp_plugin is not None and isinstance(self.ddp_plugin, RPCPlugin)

    @property
    def distributed_sampler_kwargs(self):
        raise NotImplementedError

    @property
    def require_distributed_sampler(self):
        raise NotImplementedError

    @contextmanager
    def block_ddp_plugin_sync_behaviour(self):
        """
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        cm = self.ddp_plugin.block_backward_sync(self.trainer.model) if self.ddp_plugin else None
        yield cm
