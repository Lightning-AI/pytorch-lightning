import io
import os
from typing import Optional

import torch

from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.plugins.training_type.utils import on_colab_kaggle
from pytorch_lightning.utilities import _TPU_AVAILABLE, rank_zero_warn

if _TPU_AVAILABLE:
    import torch_xla
    import torch_xla.core.xla_model as xm


class SingleTPUPlugin(SingleDevicePlugin):

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0

    def on_tpu(self) -> bool:
        return True

    def pre_training(self) -> None:
        if isinstance(self.device, int):
            self.device = xm.xla_device(self.device)

        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

    def post_training(self) -> None:
        model = self.lightning_module

        if on_colab_kaggle():
            rank_zero_warn("cleaning up... please do not interrupt")
            self.save_spawn_weights(model)
