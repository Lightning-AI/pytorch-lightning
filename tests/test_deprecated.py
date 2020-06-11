"""Test deprecated functionality which will be removed in vX.Y.Z"""
import sys

import pytest

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


def _soft_unimport_module(str_module):
    # once the module is imported  e.g with parsing with pytest it lives in memory
    if str_module in sys.modules:
        del sys.modules[str_module]


def test_tbd_remove_in_v0_9_0_trainer():
    # test show_progress_bar set by progress_bar_refresh_rate
    with pytest.deprecated_call(match='v0.9.0'):
        trainer = Trainer(progress_bar_refresh_rate=0, show_progress_bar=True)
    assert not getattr(trainer, 'show_progress_bar')

    with pytest.deprecated_call(match='v0.9.0'):
        trainer = Trainer(progress_bar_refresh_rate=50, show_progress_bar=False)
    assert getattr(trainer, 'show_progress_bar')

    with pytest.deprecated_call(match='v0.9.0'):
        trainer = Trainer(num_tpu_cores=8)
        assert trainer.tpu_cores == 8


def test_tbd_remove_in_v0_9_0_module_imports():
    _soft_unimport_module("pytorch_lightning.core.decorators")
    with pytest.deprecated_call(match='v0.9.0'):
        from pytorch_lightning.core.decorators import data_loader  # noqa: F811
        data_loader(print)

    _soft_unimport_module("pytorch_lightning.logging.comet")
    with pytest.deprecated_call(match='v0.9.0'):
        from pytorch_lightning.logging.comet import CometLogger  # noqa: F402
    _soft_unimport_module("pytorch_lightning.logging.mlflow")
    with pytest.deprecated_call(match='v0.9.0'):
        from pytorch_lightning.logging.mlflow import MLFlowLogger  # noqa: F402
    _soft_unimport_module("pytorch_lightning.logging.neptune")
    with pytest.deprecated_call(match='v0.9.0'):
        from pytorch_lightning.logging.neptune import NeptuneLogger  # noqa: F402
    _soft_unimport_module("pytorch_lightning.logging.test_tube")
    with pytest.deprecated_call(match='v0.9.0'):
        from pytorch_lightning.logging.test_tube import TestTubeLogger  # noqa: F402
    _soft_unimport_module("pytorch_lightning.logging.wandb")
    with pytest.deprecated_call(match='v0.9.0'):
        from pytorch_lightning.logging.wandb import WandbLogger  # noqa: F402


class ModelVer0_6(EvalModelTemplate):

    # todo: this shall not be needed while evaluate asks for dataloader explicitly
    def val_dataloader(self):
        return self.dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return {'val_loss': 0.6}

    def validation_end(self, outputs):
        return {'val_loss': 0.6}

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_end(self, outputs):
        return {'test_loss': 0.6}


class ModelVer0_7(EvalModelTemplate):

    # todo: this shall not be needed while evaluate asks for dataloader explicitly
    def val_dataloader(self):
        return self.dataloader(train=False)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return {'val_loss': 0.7}

    def validation_end(self, outputs):
        return {'val_loss': 0.7}

    def test_dataloader(self):
        return self.dataloader(train=False)

    def test_end(self, outputs):
        return {'test_loss': 0.7}


def test_tbd_remove_in_v1_0_0_model_hooks():
    hparams = EvalModelTemplate.get_default_hparams()

    model = ModelVer0_6(hparams)

    with pytest.deprecated_call(match='v1.0'):
        trainer = Trainer(logger=False)
        trainer.test(model)
    assert trainer.callback_metrics == {'test_loss': 0.6}

    with pytest.deprecated_call(match='v1.0'):
        trainer = Trainer(logger=False)
        # TODO: why `dataloder` is required if it is not used
        result = trainer._evaluate(model, dataloaders=[[None]], max_batches=1)
    assert result == {'val_loss': 0.6}

    model = ModelVer0_7(hparams)

    with pytest.deprecated_call(match='v1.0'):
        trainer = Trainer(logger=False)
        trainer.test(model)
    assert trainer.callback_metrics == {'test_loss': 0.7}

    with pytest.deprecated_call(match='v1.0'):
        trainer = Trainer(logger=False)
        # TODO: why `dataloder` is required if it is not used
        result = trainer._evaluate(model, dataloaders=[[None]], max_batches=1)
    assert result == {'val_loss': 0.7}
