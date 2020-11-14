from typing import cast

from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler

from pytorch_lightning.plugins.native_amp import NativeAMPPlugin


class ShardedNativeAMPPlugin(NativeAMPPlugin):
    @property
    def scaler(self):
        return ShardedGradScaler()

    def clip_gradients(self, grad_clip_val, model, optimizer):
        max_norm = grad_clip_val
        norm_type = float(2.0)
        optimizer = cast(OSS, optimizer)
        optimizer.clip_grad_norm(max_norm, norm_type=norm_type)
