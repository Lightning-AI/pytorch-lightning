"""Test deprecated functionality which will be removed in vX.Y.Z"""
import pytest
import sys

import torch

from tests.base import EvalModelTemplate
from pytorch_lightning.metrics.functional.classification import auc

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_tbd_remove_in_v1_2_0():
    with pytest.deprecated_call(match='will be removed in v1.2'):
        checkpoint_cb = ModelCheckpoint(filepath='.')

    with pytest.deprecated_call(match='will be removed in v1.2'):
        checkpoint_cb = ModelCheckpoint('.')

    with pytest.raises(MisconfigurationException, match='inputs which are not feasible'):
        checkpoint_cb = ModelCheckpoint(filepath='.', dirpath='.')


def _soft_unimport_module(str_module):
    # once the module is imported  e.g with parsing with pytest it lives in memory
    if str_module in sys.modules:
        del sys.modules[str_module]


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


def test_auc_reorder_remove_in_v1_1_0():
    with pytest.deprecated_call(match='The `reorder` parameter to `auc` has been deprecated'):
        _ = auc(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 2]), reorder=True)
