from typing import Optional

from pytorch_lightning.core.optimizer import is_lightning_optimizer
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE, rank_zero_only

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel


class DDPShardedPlugin(DDPPlugin):

    def configure_ddp(self):
        self._wrap_optimizers()
        self._model = LightningShardedDataParallel(
            self.model, sharded_optimizer=self.lightning_module.trainer.optimizers
        )

    def _reinit_optimizers_with_oss(self):
        optimizers = self.lightning_module.trainer.optimizers
        for x, optimizer in enumerate(optimizers):
            if is_lightning_optimizer(optimizer):
                optimizer = optimizer._optimizer
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                optimizers[x] = zero_optimizer
                del optimizer
        trainer = self.lightning_module.trainer
        trainer.optimizers = trainer.convert_to_lightning_optimizers(optimizers)

    def _wrap_optimizers(self):
        trainer = self.model.trainer
        if trainer.testing is True:
            return
        self._reinit_optimizers_with_oss()

    def optimizer_state(self, optimizer: 'OSS') -> Optional[dict]:
        if is_lightning_optimizer(optimizer):
            optimizer = optimizer._optimizer
        optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Retrieves state dict only on rank 0, which contains the entire optimizer state after calling
        :meth:`consolidate_state_dict`.
        """
        return optimizer.state_dict()
