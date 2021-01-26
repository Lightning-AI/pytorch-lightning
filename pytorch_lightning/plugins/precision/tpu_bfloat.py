import os
import torch
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin

class TPUHalfPrecisionPlugin(PrecisionPlugin):
    def connect(self, model: torch.nn.Module, optimizers, lr_schedulers):
        os.environ['XLA_USE_BF16'] = str(1)
        return super().connect(model=model, optimizers=optimizers, lr_schedulers=lr_schedulers)