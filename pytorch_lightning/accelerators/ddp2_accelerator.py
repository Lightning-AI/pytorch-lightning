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
import os

import torch
import torch.distributed as torch_distrib

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.distributed.dist import LightningDistributed
from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.accelerator import Accelerator, ReduceOp
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.distributed import rank_zero_only, sync_ddp_if_available
from torch.nn.parallel import DistributedDataParallel
from typing import List, Optional, Union, Any

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDP2Accelerator(Accelerator):

    def __init__(self, trainer, cluster_environment=None, ddp_plugin=None):
        """
        Runs training using DDP2 strategy on a cluster

        Example::

            # default
            trainer = Trainer(accelerator=DDP2Accelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.task_idx = None
        self.dist = LightningDistributed()
        self.nickname = 'ddp2'

    def setup(self, model):
        self.trainer.model = model
        self.task_idx = self.cluster_environment.local_rank()

    def train(self):
        model = self.trainer.model
        return self.ddp_train(process_idx=self.task_idx, mp_queue=None, model=model)

    def training_step(self, args):
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model(*args)
        else:
            output = self.trainer.model(*args)
        return output

    def validation_step(self, args):
        output = self.training_step(args)
        return output

    def test_step(self, args):
        output = self.training_step(args)
        return output

    def barrier(self, name: Optional[str] = None):
        if torch_distrib.is_initialized():
            torch_distrib.barrier()

    def training_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        return output

    def validation_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        return output

    def test_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        return output

    def set_world_ranks(self, process_idx):
        self.trainer.local_rank = self.trainer.node_rank
        self.trainer.global_rank = self.trainer.node_rank
        self.trainer.world_size = self.trainer.num_nodes

    def broadcast(self, obj, src=0):
        return self.dist.broadcast(obj)

    def model_to_device(self, model, process_idx):
        self.trainer.root_gpu = process_idx
        torch.cuda.set_device(self.trainer.root_gpu)
        model.cuda(self.trainer.root_gpu)

    def get_device_ids(self):
        device_ids = self.trainer.data_parallel_device_ids
        return device_ids

    def ddp_train(self, process_idx, mp_queue, model):
        """
        Entry point for ddp

        Args:
            process_idx: current process rank
            mp_queue: multiprocessing queue
            model: pointer to current :class:`LightningModule`

        Returns:
            Dict with evaluation results

        """
        # show progressbar only on progress_rank 0
        if (self.trainer.node_rank != 0 or process_idx != 0) and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # determine which process we are and world size
        self.set_world_ranks(process_idx)

        # set warning rank
        rank_zero_only.rank = self.trainer.global_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self.trainer
        self.init_ddp_connection(
            self.trainer.global_rank,
            self.trainer.world_size,
            self.trainer.is_slurm_managing_tasks
        )

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # on world_size=0 let everyone know training is starting
        if self.trainer.is_global_zero and not torch.distributed.is_initialized():
            log.info('-' * 100)
            log.info(f'distributed_backend={self.trainer.distributed_backend}')
            log.info(f'All DDP processes registered. Starting ddp with {self.trainer.world_size} processes')
            log.info('-' * 100)

        # call sync_bn before .cuda(), configure_apex and configure_ddp
        if self.trainer.sync_batchnorm:
            model = self.configure_sync_batchnorm(model)

        # move the model to the correct device
        self.model_to_device(model, process_idx)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # set model properties before going into wrapper
        self.trainer.model_connector.copy_trainer_model_properties(model)

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        # device ids change depending on the DDP setup
        device_ids = self.get_device_ids()

        # allow user to configure ddp
        model = self.configure_ddp(model, device_ids)

        # set up training routine
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()

        # clean up memory
        torch.cuda.empty_cache()
        return results

    def configure_ddp(
        self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model

    def configure_sync_batchnorm(self, model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model

    def sync_tensor(self,
                    tensor: Union[torch.Tensor],
                    group: Optional[Any] = None,
                    reduce_op: Optional[Union[ReduceOp, str]] = None) -> torch.Tensor:
        return sync_ddp_if_available(tensor, group, reduce_op)
