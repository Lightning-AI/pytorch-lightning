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
    from torch_xla._patched_functions import clip_grad_norm_

    xla_clip_grad_norm_ = clip_grad_norm_


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
        xm.optimizer_step(optimizer, barrier=False, optimizer_args={'closure': lambda_closure, **kwargs})

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        """
        Function to gather a tensor from several distributed processes
        Args:
            tensor: tensor of shape (batch, ...)
            group: not available with TPUs
            sync_grads: not available with TPUs
        Return:
            A tensor of shape (world_size, batch, ...)
        """
        # todo: Add support for backward with all_gather
        if isinstance(self.training_type_plugin, TPUSpawnPlugin) and self.training_type_plugin.is_distributed:
            return xm.all_gather(tensor).view(-1, *tensor.shape)
        return tensor

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[float, int], norm_type: float = 2.0):

        model = self.lightning_module
        parameters = model.parameters()

        grad_clip_val = float(clip_val)
        if grad_clip_val <= 0:
            return

        max_norm = grad_clip_val

        xla_clip_grad_norm_(parameters, max_norm, norm_type)
