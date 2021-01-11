from pytorch_lightning.accelerators.plugins import DDPPlugin
from pytorch_lightning.core.optimizer import is_lightning_optimizer
from pytorch_lightning.utilities import _FAIRSCALE_AVAILABLE

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel


class ShardedPlugin(DDPPlugin):
    def configure_ddp(self):
        self._model = LightningShardedDataParallel(
            self.model,
            sharded_optimizer=self.lightning_module.trainer.optimizers
        )

    def init_ddp_connection(self, global_rank: int, world_size: int) -> None:
        super().init_ddp_connection(global_rank, world_size)
        self._reinit_optimizers_with_oss()

    def _reinit_optimizers_with_oss(self):
        optimizers = self.lightning_module.trainer.optimizers
        for x, optimizer in enumerate(optimizers):
            if is_lightning_optimizer(optimizer):
                optimizer = optimizer._optimizer
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(
                    params=optimizer.param_groups,
                    optim=optim_class,
                    **optimizer.defaults
                )
                optimizers[x] = zero_optimizer
                del optimizer
        self.lightning_module.trainer.convert_to_lightning_optimizers()
