from typing import Any, Callable, Optional, Union

import torch
from torch.optim import Optimizer

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class TPUAccelerator(Accelerator):

    def setup(self, trainer, model):
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            raise MisconfigurationException(
                "amp + tpu is not supported. "
                "Only bfloats are supported on TPU. Consider using TPUHalfPrecisionPlugin"
            )

        if not isinstance(self.training_type_plugin, (SingleTPUPlugin, TPUSpawnPlugin)):
            raise MisconfigurationException("TPUs only support a single tpu core or tpu spawn training.")
        return super().setup(trainer, model)

    def run_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs):
        xm.optimizer_step(optimizer, optimizer_args={'closure': lambda_closure, **kwargs})

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        """
        Function to gather a tensor from several distributed processes
        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op
        Return:
            A tensor of shape (world_size, batch, ...)
        """
        return xm.all_gather(tensor, group=group, sync_grads=sync_grads)
