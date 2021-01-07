from contextlib import contextmanager
from typing import List, Tuple
import torch
from torch.optim import Optimizer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import AMPType, _APEX_AVAILABLE, rank_zero_warn
from pytorch_lightning.accelerators.plugins.precision.mixed import MixedPrecisionPlugin

if _APEX_AVAILABLE:
    from apex import amp

class ApexMixedPrecisionPlugin(MixedPrecisionPlugin):
    def __init__(self, amp_level):
        self.backend = AMPType.APEX
        self.amp_level = amp_level

    def master_params(self, optimizer):
        return amp.master_params(optimizer)

    def connect(self, model, optimizers, lr_schedulers):
        model, optimizers = self.configure_apex(amp, model, optimizers, self.amp_level)
        self.reinit_scheduler_properties(optimizers, lr_schedulers)
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

    @staticmethod
    def reinit_scheduler_properties(optimizers: list, schedulers: list):
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']

            for optimizer in optimizers:
                state = None
                idx = 0

                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        if mro in (optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            idx = i
                            state = scheduler.state_dict()
                        else:
                            state = None

                scheduler.__class__.__mro__[idx].__init__(scheduler, optimizer)
                if state is not None:
                    scheduler.load_state_dict(state)