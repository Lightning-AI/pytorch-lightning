from typing import Optional

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.training_type.ddp_spawn import DDPSpawnPlugin
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE, rank_zero_only

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded


class DDPSpawnShardedPlugin(DDPSpawnPlugin):

    def configure_ddp(self):
        self._wrap_optimizers()
        self._model = ShardedDataParallel(
            LightningShardedDataParallel(self.model), sharded_optimizer=self.lightning_module.trainer.optimizers
        )

    def _reinit_optimizers_with_oss(self):
        optimizers = self.lightning_module.trainer.optimizers
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                optimizers[x] = zero_optimizer
                del optimizer
        trainer = self.lightning_module.trainer
        trainer.optimizers = optimizers

    def _wrap_optimizers(self):
        trainer = self.model.trainer
        if trainer.testing:
            return
        self._reinit_optimizers_with_oss()

    def optimizer_state(self, optimizer: 'OSS') -> Optional[dict]:
        if isinstance(optimizer, OSS):
            optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Retrieves state dict only on rank 0, which contains the entire optimizer state after calling
        :meth:`consolidate_state_dict`.
        """
        return optimizer.state_dict()

    @property
    def lightning_module(self) -> LightningModule:
        return unwrap_lightning_module_sharded(self._model)
