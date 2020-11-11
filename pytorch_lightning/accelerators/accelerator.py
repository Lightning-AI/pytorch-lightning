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
import os
import math
from enum import Enum
from typing import Any, Optional, Union

import torch

from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict
import torch.distributed as torch_distrib
from pytorch_lightning import _logger as log

try:
    from apex import amp
except ImportError:
    amp = None

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:
    class ReduceOp:
        SUM = None

EPSILON = 1e-6
EPSILON_FP16 = 1e-5


class Accelerator(object):

    def __init__(self, trainer=None, cluster_environment=None, ddp_plugin=None):
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
        return closure_loss

    def optimizer_step(self, optimizer, batch_idx, opt_idx, lambda_closure):
        model_ref = self.trainer.get_model()
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        using_native_amp = self.trainer.amp_backend == AMPType.NATIVE
        automatic_optimization = self.trainer.train_loop.automatic_optimization

        # native amp + lbfgs is a no go right now
        if using_native_amp and is_lbfgs:
            raise MisconfigurationException(
                'native PyTorch amp and lbfgs are not compatible.'
                ' To request, please file a Github issue in PyTorch and tag @mcarilli')

        # model hook
        model_ref.optimizer_step(
            epoch=self.trainer.current_epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=opt_idx,
            optimizer_closure=lambda_closure,
            on_tpu=False,  # TPUAccelerator class sets this as True
            using_native_amp=using_native_amp,
            using_lbfgs=is_lbfgs
        )

        # scale when native amp
        if automatic_optimization and using_native_amp:
            self.trainer.scaler.update()

    def optimizer_zero_grad(self, batch_idx, optimizer, opt_idx):
        model_ref = self.trainer.get_model()
        model_ref.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)

    def clip_gradients(self, optimizer, clip_val=None):
        # TODO: separate TPU case from here
        self._clip_gradients(optimizer, clip_val)

    def _clip_gradients(self, optimizer, clip_val=None):
        # use the trainer's clip val if none passed
        grad_clip_val = self.trainer.gradient_clip_val
        if clip_val is not None:
            grad_clip_val = clip_val
        grad_clip_val = float(grad_clip_val)

        # this code is a modification of torch.nn.utils.clip_grad_norm_
        # with TPU support based on https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
        if grad_clip_val <= 0:
            return

        model = self.trainer.get_model()
        if self.trainer.amp_backend == AMPType.APEX:
            parameters = amp.master_params(optimizer)
        else:
            parameters = model.parameters()

        max_norm = grad_clip_val
        norm_type = float(2.0)

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            device = parameters[0].device
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        eps = EPSILON_FP16 if self.trainer.precision == 16 else EPSILON
        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))

    def on_train_epoch_end(self, outputs):
        pass

    def on_train_end(self):
        pass

    def early_stopping_should_stop(self, pl_module):
        return self.trainer.should_stop

    def setup_optimizers(self, model):
        if self.trainer.testing is True:
            return

        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

    def init_ddp_connection(
        self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True
    ) -> None:
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())
        torch_backend = "nccl" if self.trainer.on_gpu else "gloo"

        if not torch.distributed.is_initialized():
            log.info(
                f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
            )
            torch_distrib.init_process_group(
                torch_backend, rank=global_rank, world_size=world_size
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


# TODO: allow user to compare with string even internaly we shall use these Enum to prevent typos...
class BackendType(Enum):
    DP = 'dp'
    DDP = 'ddp'
    DDP2 = 'ddp2'
    DDP_SPAWN = 'ddp_spawn'
    # decuple distrib and device
    DDP_CPU = 'ddp_cpu'
    HOROVOD = 'horovod'
    # this is rather device
    TPU = 'tpu'
