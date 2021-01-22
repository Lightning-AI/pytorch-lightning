from contextlib import contextmanager

import torch

from pytorch_lightning.accelerators.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class NativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    def __init__(self):
        self.backend = AMPType.NATIVE
        self.scaler = torch.cuda.amp.GradScaler()

    def pre_optimizer_step(self, optimizer, optimizer_idx):
        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationException(
                f"native PyTorch amp and lbfgs are not compatible (optimizer {optimizer_idx})."
                " To request, please file a Github issue in PyTorch and tag @mcarilli"
            )

    def post_optimizer_step(self, optimizer, optimizer_idx):
        self.scaler.update()

    def backward(
        self,
        model: LightningModule,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ):
        closure_loss = self.scaler.scale(closure_loss)

        automatic_optimization = model.automatic_optimization

        closure_loss = super().backward(model, closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs)

        # unscale gradient to allow analyze within `on_after_backward`
        if not should_accumulate and automatic_optimization:
            self.scaler.unscale_(optimizer)

        return closure_loss

    @contextmanager
    def train_step_context(self):
        yield torch.cuda.amp.autocast()