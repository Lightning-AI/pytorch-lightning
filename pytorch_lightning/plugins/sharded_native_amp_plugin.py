from fairscale.optim.grad_scaler import ShardedGradScaler

from pytorch_lightning.plugins.native_amp import NativeAMPPlugin


class ShardedNativeAMPPlugin(NativeAMPPlugin):
    @property
    def scaler(self):
        return ShardedGradScaler()
