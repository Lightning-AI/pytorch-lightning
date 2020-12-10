from contextlib import contextmanager
from pytorch_lightning.accelerators.base_plugin import Plugin
from pytorch_lightning.accelerators.scheduler_properties import reinit_scheduler_properties
from pytorch_lightning.core.lightning import LightningModule
from typing import List, Tuple
import torch
from torch.optim import Optimizer

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import AMPType, rank_zero_warn

try:
    from apex import amp
except ImportError:
    amp = None


class PrecisionPlugin(Plugin):
    EPSILON = 1e-6
    precision = 32

    def pre_optimizer_step(self, optimizer, optimizer_idx):
        pass

    def post_optimizer_step(self, optimizer, optimizer_idx):
        pass

    def master_params(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    def connect(self, model: torch.nn.Module, optimizers, lr_schedulers):
        return model, optimizers, lr_schedulers

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
        # TODO: Check where we can get automatic_optimization from (probably when setting up the model after https://github.com/PyTorchLightning/pytorch-lightning/issues/4317)
        automatic_optimization = model.automatic_optimization

        # do backward pass
        if automatic_optimization:
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss


class MixedPrecisionPlugin(PrecisionPlugin):
    EPSILON = 1e-5
    backend: AMPType
    precision = "mixed"


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

        # TODO: Check where we can get automatic_optimization from (probably when setting up the model after https://github.com/PyTorchLightning/pytorch-lightning/issues/4317)
        automatic_optimization = model.automatic_optimization

        closure_loss = super().backward(model, closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs)

        # unscale gradient to allow analyze within `on_after_backward`
        if not should_accumulate and automatic_optimization:
            self.scaler.unscale_(optimizer)

        return closure_loss

    @contextmanager
    def train_step_context(self):
        yield torch.cuda.amp.autocast()


class ApexMixedPrecisionPlugin(MixedPrecisionPlugin):
    def __init__(self, amp_level):
        self.backend = AMPType.APEX
        self.amp_level = amp_level

    def connect(self, model, optimizers, lr_schedulers):
        model, optimizers = self.configure_apex(amp, model, optimizers, self.amp_level)
        reinit_scheduler_properties(optimizers, lr_schedulers)
        return model, optimizers, lr_schedulers

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
        closure_loss = amp.scale_loss(closure_loss, optimizer)

        # enter apex context
        context = closure_loss
        closure_loss = closure_loss.__enter__()

        # do backward pass
        # TODO: not entirely sure, why we need this
        if model is not None and isinstance(model, LightningModule):
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # exit amp context
        a, b, c = None, None, None
        error = context.__exit__(a, b, c)
        if error:
            rank_zero_warn(a, b, c)
            raise Exception("apex unscale error")

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        return closure_loss

    def configure_apex(
        self,
        amp: object,
        model: LightningModule,
        optimizers: List[Optimizer],
        amp_level: str,
    ) -> Tuple[LightningModule, List[Optimizer]]:
        r"""
        Override to init AMP your own way.
        Must return a model and list of optimizers.

        Args:
            amp: pointer to amp library object.
            model: pointer to current :class:`LightningModule`.
            optimizers: list of optimizers passed in :meth:`configure_optimizers`.
            amp_level: AMP mode chosen ('O1', 'O2', etc...)

        Return:
            Apex wrapped model and optimizers

        Examples:
            .. code-block:: python

                # Default implementation used by Trainer.
                def configure_apex(self, amp, model, optimizers, amp_level):
                    model, optimizers = amp.initialize(
                        model, optimizers, opt_level=amp_level,
                    )

                    return model, optimizers
        """
        model, optimizers = amp.initialize(model, optimizers, opt_level=amp_level)
        return model, optimizers