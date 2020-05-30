import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from pl_examples.models.lightning_template import LightningTemplateModel


class MyModel(LightningTemplateModel):
    def __init__(self, n_freeze_epochs, n_unfreeze_epochs, freeze_lrs, unfreeze_lrs, **kwargs):
        self.n_freeze_epochs = n_freeze_epochs
        self.n_unfreeze_epochs = n_unfreeze_epochs
        self.freeze_lrs = freeze_lrs
        self.unfreeze_lrs = unfreeze_lrs
        super().__init__(**kwargs)

    def model_splits(self):
        return [nn.Sequential(self.c_d1, self.c_d1_bn), self.c_d2]

    def configure_optimizers(self):
        param_groups = self.get_optimizer_param_groups(0)
        opt = torch.optim.Adam(param_groups)
        # Dummy sched so LearningRateLogger does not complain
        sched = OneCycleLR(opt, 0, 10)
        return [opt], [sched]

    def on_epoch_start(self):
        if self.current_epoch == 0:
            # Freeze all but last layer (imagine this is the head)
            self.freeze_to(-1)
            self.replace_lr_scheduler(self.n_freeze_epochs, self.freeze_lrs, pct_start=.9)

        if self.current_epoch == self.n_freeze_epochs:
            # Unfreeze all layers, we can also use `unfreeze`, but `freeze_to` has the
            # additional property of only considering parameters returned by `model_splits`
            self.freeze_to(0)
            self.replace_lr_scheduler(self.n_unfreeze_epochs, self.unfreeze_lrs, pct_start=.2)

    # Currently specific to OneCycleLR
    def replace_lr_scheduler(self, n_epochs, lrs, pct_start):
        total_steps = len(model.train_dataloader()) * n_epochs
        opt = self.trainer.optimizers[0]
        sched = OneCycleLR(opt, lrs, total_steps, pct_start=pct_start)
        sched = {'scheduler': sched, 'interval': 'step'}
        scheds = self.trainer.configure_schedulers([sched])
        # Replace scheduler and update lr logger
        self.trainer.lr_schedulers = scheds
        lr_logger.on_train_start(self.trainer, self)

# if __name__ == '__main__':
n_param_groups = 2
freeze_lrs = [0] * (n_param_groups - 1) + [1e-3]
unfreeze_lrs = np.linspace(5e-6, 5e-4, n_param_groups).tolist()

# HACK: Have to define `lr_logger` globally because we're calling `lr_logger.on_train_start` inside `model.on_epoch_start`
lr_logger = pl.callbacks.LearningRateLogger()
model = MyModel(
    n_freeze_epochs=1,
    n_unfreeze_epochs=3,
    freeze_lrs=freeze_lrs,
    unfreeze_lrs=unfreeze_lrs,
    batch_size=512,
)
trainer = pl.Trainer(max_epochs=4, gpus=1, callbacks=[lr_logger])
trainer.fit(model)
