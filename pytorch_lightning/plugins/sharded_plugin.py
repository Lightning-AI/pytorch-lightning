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
from typing import List, cast, Any

from fairscale.optim import OSS

from pytorch_lightning import LightningModule
from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin


class DDPShardedPlugin(DDPPlugin):

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ):
        self._wrap_optimizers(model)
        if model.trainer.testing:  # Revert to standard DDP if using testing
            super().configure_ddp(
                model=model,
                device_ids=device_ids
            )
        else:
            model = LightningShardedDataParallel(model, sharded_optimizer=model.trainer.optimizers)
        return model

    def sync_optim_state(self, model: LightningModule):
        for optimizer in model.trainer.optimizers:
            optimizer.consolidate_state_dict()

    def rank_should_save_optim_state(self, rank):
        return rank == 0  # Only safe to save optimizer state on rank 0

    def sync_backward(self, model: LightningShardedDataParallel):
        # Ensure all backward handles have been called before calling optimizer step
        model = cast(LightningShardedDataParallel, model)
        model.clear_backward_handles()

    def input_to_device(self, args: Any, model: LightningModule):
        batch = args[0]
        batch = model.transfer_batch_to_device(batch, model.trainer.root_gpu)
        args[0] = batch
        return args

    def _wrap_optimizers(self, model):
        trainer = model.trainer
        if trainer.testing is True:
            return

        self._reinit_with_fairscale_oss(trainer)

    def _reinit_with_fairscale_oss(self, trainer):
        """
        Re-initialise optimizers to use OSS wrapper. We need to re-initialise due to
        the parameters being sharded across distributed processes, each optimizing a partition.
        Args:
            trainer: trainer object to reinit optimizers.
        """
        optimizers = trainer.optimizers
        lr_schedulers = trainer.lr_schedulers
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(
                    params=optimizer.param_groups,
                    optim=optim_class,
                    **optimizer.defaults
                )
                optimizers[x] = zero_optimizer
                for scheduler in lr_schedulers:
                    scheduler = scheduler['scheduler']
                    if scheduler.optimizer == optimizer:
                        scheduler.optimizer = zero_optimizer
                del optimizer
