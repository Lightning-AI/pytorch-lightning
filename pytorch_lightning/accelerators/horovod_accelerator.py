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
from contextlib import ExitStack
from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.distributed import rank_zero_only

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class HorovodAccelerator(Accelerator):
    amp_backend: AMPType

    def __init__(self, trainer, cluster_environment=None):
        super().__init__(trainer, cluster_environment)
        self.nickname = 'horovod'

    def setup(self, model):
        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        if torch.cuda.is_available() and self.trainer.on_gpu:
            # Horovod: pin GPU to local rank
            assert self.trainer.root_gpu == hvd.local_rank()
            torch.cuda.set_device(self.trainer.root_gpu)
            model.cuda(self.trainer.root_gpu)

        # avoid duplicating progress bar
        if hvd.rank() != 0 and self.trainer.progress_bar_callback is not None:
            self.trainer.progress_bar_callback.disable()

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # Horovod: scale the learning rate by the number of workers to account for
        # increased total batch size
        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= hvd.size()

        # Horovod: adjust base LR used by schedulers to match scaled optimizer initial LR
        for scheduler in self.trainer.lr_schedulers:
            scheduler = scheduler['scheduler']
            if isinstance(scheduler, _LRScheduler):
                scheduler.base_lrs = [lr * hvd.size() for lr in scheduler.base_lrs]

        # Horovod: broadcast parameters & optimizer state to ensure consistent initialization
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        for optimizer in self.trainer.optimizers:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        def _filter_named_parameters(model, optimizer):
            opt_params = set([p for group in optimizer.param_groups for p in group.get('params', [])])
            return [(name, p) for name, p in model.named_parameters() if p in opt_params]

        # Horovod: wrap optimizers to perform gradient aggregation via allreduce
        self.trainer.optimizers = [
            hvd.DistributedOptimizer(optimizer, named_parameters=_filter_named_parameters(model, optimizer))
            for optimizer in self.trainer.optimizers
        ]

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        # Update logger rank info from Horovod to avoid race conditions from  different ranks
        # creating directories / writing files in the same locations.
        self.trainer.global_rank = hvd.rank()
        rank_zero_only.rank = self.trainer.global_rank

        self.trainer.model = model

    def train(self):
        with ExitStack() as stack:
            for optimizer in self.trainer.optimizers:
                # Synchronization will be performed explicitly following backward()
                stack.enter_context(optimizer.skip_synchronize())

            # set up training routine
            self.trainer.train_loop.setup_training(self.trainer.model)

            # train or test
            results = self.train_or_test()

        # Make sure all workers have finished training before returning to the user
        hvd.join()
        return results

    def training_step(self, args):
        if self.trainer.on_gpu:
            batch = args[0]
            batch = self.batch_to_device(batch, hvd.local_rank())
            args[0] = batch

        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model.training_step(*args)
        else:
            output = self.trainer.model.training_step(*args)

        return output

    def validation_step(self, args):
        if self.trainer.on_gpu:
            batch = args[0]
            batch = self.batch_to_device(batch, hvd.local_rank())
            args[0] = batch

        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model.validation_step(*args)
        else:
            output = self.trainer.model.validation_step(*args)

        return output

    def test_step(self, args):
        if self.trainer.on_gpu:
            batch = args[0]
            batch = self.batch_to_device(batch, hvd.local_rank())
            args[0] = batch

        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model.test_step(*args)
        else:
            output = self.trainer.model.test_step(*args)
        return output

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        super().backward(closure_loss, optimizer, opt_idx, *args, **kwargs)
        optimizer.synchronize()

    def on_train_epoch_end(self, outputs):
        hvd.join(hvd.local_rank() if self.trainer.on_gpu else -1)

    def barrier(self, name: Optional[str] = None):
        hvd.join()

    def broadcast(self, obj, src=0):
        obj = hvd.broadcast_object(obj, src)
        return obj
