from typing import Callable, Union

import torch
from torch.optim import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin


class DeepSpeedPrecisionPlugin(PrecisionPlugin):

    def __init__(self, precision):
        super().__init__()
        self.precision = precision

    def pre_optimizer_step(
        self, pl_module: LightningModule, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs
    ) -> bool:
        # DeepSpeed not support closures.
        lambda_closure()

        if not pl_module.automatic_optimization:
            pl_module.trainer.call_hook("on_after_backward")
            optimizer.step()

        return False

    def backward(
        self,
        lightning_module: LightningModule,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ):
        # todo: hack around for deepspeed engine to call backward
        # Means that the lightning module backward function is never called
        # This is an issue if the user overrides the backwards function
        deepspeed_engine = lightning_module.trainer.model
        deepspeed_engine.backward(closure_loss, **kwargs)
        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float], norm_type: float = float(2.0)):
        """
        DeepSpeed handles clipping gradients via the training type plugin.
        """
        pass
