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
from typing import List, Any, Optional

from fairscale.optim import OSS

from pytorch_lightning import LightningModule
from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only


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

    def optimizer_state(self, optimizer: OSS) -> Optional[dict]:
        optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    def on_before_forward(self, args: Any, model: LightningModule):
        batch = args[0]
        batch = model.transfer_batch_to_device(batch, model.trainer.root_gpu)
        args[0] = batch
        return args

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Ensure we only return the state dict from the optimizer on rank 0.
        Other ranks do not have the complete optimizer state.
        Args:
            optimizer: OSS Optimizer
        Returns:
            State dict if rank 0 else None.
        """
        return optimizer.state_dict()

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
