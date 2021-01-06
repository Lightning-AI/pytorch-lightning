import torch
from pytorch_lightning.core import LightningModule
from pytorch_lightning.accelerators.plugins.base_plugin import Plugin


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
        automatic_optimization = model.automatic_optimization

        # do backward pass
        if automatic_optimization:
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        return closure_loss