import torch
from pytorch_lightning import data_loader

from .lm_test_module_base import LightningTestModelBase
from . import lm_test_modules_callbacks as callbacks


class LightningTestModel(LightningTestModelBase):

    def validation_step(self, data_batch, batch_i):
        return callbacks.validation_step(self, data_batch, batch_i)

    def validation_end(self, outputs):
        return callbacks.validation_end(self, outputs)

    def test_step(self, data_batch, batch_i):
        return callbacks.test_step(self, data_batch,  batch_i)

    def test_end(self, outputs):
        return callbacks.test_end(self, outputs)

    @data_loader
    def val_dataloader(self):
        return self._dataloader(train=False)

    @data_loader
    def test_dataloader(self):
        return [self._dataloader(train=False), self._dataloader(train=False)]

    def on_tng_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
