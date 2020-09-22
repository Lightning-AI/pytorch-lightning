"""Test deprecated functionality which will be removed in vX.Y.Z"""
import random
import sys

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GpuUsageLogger, LearningRateLogger
from tests.base import EvalModelTemplate


def _soft_unimport_module(str_module):
    # once the module is imported  e.g with parsing with pytest it lives in memory
    if str_module in sys.modules:
        del sys.modules[str_module]


def test_tbd_remove_in_v0_11_0_trainer():
    with pytest.deprecated_call(match='will be removed in v0.11.0'):
        lr_logger = LearningRateLogger()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_tbd_remove_in_v0_11_0_trainer_gpu():
    with pytest.deprecated_call(match='will be removed in v0.11.0'):
        gpu_usage = GpuUsageLogger()


class ModelVer0_6(EvalModelTemplate):

    # todo: this shall not be needed while evaluate asks for dataloader explicitly
    def val_dataloader(self):
        return self.dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return {'val_loss': torch.tensor(0.6)}

    def validation_end(self, outputs):
        return {'val_loss': torch.tensor(0.6)}

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_end(self, outputs):
        return {'test_loss': torch.tensor(0.6)}


class ModelVer0_7(EvalModelTemplate):

    # todo: this shall not be needed while evaluate asks for dataloader explicitly
    def val_dataloader(self):
        return self.dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return {'val_loss': torch.tensor(0.7)}

    def validation_end(self, outputs):
        return {'val_loss': torch.tensor(0.7)}

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_end(self, outputs):
        return {'test_loss': torch.tensor(0.7)}


# def test_tbd_remove_in_v1_0_0_model_hooks():
#
#     model = ModelVer0_6()
#
#     with pytest.deprecated_call(match='will be removed in v1.0. Use `test_epoch_end` instead'):
#         trainer = Trainer(logger=False)
#         trainer.test(model)
#     assert trainer.logger_connector.callback_metrics == {'test_loss': torch.tensor(0.6)}
#
#     with pytest.deprecated_call(match='will be removed in v1.0. Use `validation_epoch_end` instead'):
#         trainer = Trainer(logger=False)
#         # TODO: why `dataloder` is required if it is not used
#         result = trainer._evaluate(model, dataloaders=[[None]], max_batches=1)
#     assert result[0] == {'val_loss': torch.tensor(0.6)}
#
#     model = ModelVer0_7()
#
#     with pytest.deprecated_call(match='will be removed in v1.0. Use `test_epoch_end` instead'):
#         trainer = Trainer(logger=False)
#         trainer.test(model)
#     assert trainer.logger_connector.callback_metrics == {'test_loss': torch.tensor(0.7)}
#
#     with pytest.deprecated_call(match='will be removed in v1.0. Use `validation_epoch_end` instead'):
#         trainer = Trainer(logger=False)
#         # TODO: why `dataloder` is required if it is not used
#         result = trainer._evaluate(model, dataloaders=[[None]], max_batches=1)
#     assert result[0] == {'val_loss': torch.tensor(0.7)}
