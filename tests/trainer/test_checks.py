import pytest

import tests.base.utils as tutils
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


# TODO: add matching messages


def test_wrong_train_setting(tmpdir):
    """
    * Test that an error is thrown when no `training_dataloader()` is defined
    * Test that an error is thrown when no `training_step()` is defined
    """
    tutils.reset_seed()
    hparams = EvalModelTemplate.get_default_hparams()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.train_dataloader = None
        trainer.fit(model)

    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.training_step = None
        trainer.fit(model)


def test_wrong_configure_optimizers(tmpdir):
    """ Test that an error is thrown when no `configure_optimizers()` is defined """
    tutils.reset_seed()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate()
        model.configure_optimizers = None
        trainer.fit(model)


def test_wrong_validation_settings(tmpdir):
    """ Test the following cases related to validation configuration of model:
        * error if `val_dataloader()` is overridden but `validation_step()` is not
        * if both `val_dataloader()` and `validation_step()` is overridden,
            throw warning if `val_epoch_end()` is not defined
        * error if `validation_step()` is overridden but `val_dataloader()` is not
    """
    tutils.reset_seed()
    hparams = EvalModelTemplate.get_default_hparams()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    # check val_dataloader -> val_step
    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.validation_step = None
        trainer.fit(model)

    # check val_dataloader + val_step -> val_epoch_end
    with pytest.warns(RuntimeWarning):
        model = EvalModelTemplate(hparams)
        model.validation_epoch_end = None
        trainer.fit(model)

    # check val_step -> val_dataloader
    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.val_dataloader = None
        trainer.fit(model)


def test_wrong_test_settigs(tmpdir):
    """ Test the following cases related to test configuration of model:
        * error if `test_dataloader()` is overridden but `test_step()` is not
        * if both `test_dataloader()` and `test_step()` is overridden,
            throw warning if `test_epoch_end()` is not defined
        * error if `test_step()` is overridden but `test_dataloader()` is not
    """
    hparams = EvalModelTemplate.get_default_hparams()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    # ----------------
    # if have test_dataloader should  have test_step
    # ----------------
    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.test_step = None
        trainer.fit(model)

    # ----------------
    # if have test_dataloader  and  test_step recommend test_epoch_end
    # ----------------
    with pytest.warns(RuntimeWarning):
        model = EvalModelTemplate(hparams)
        model.test_epoch_end = None
        trainer.test(model)

    # ----------------
    # if have test_step and NO test_dataloader passed in tell user to pass test_dataloader
    # ----------------
    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.test_dataloader = LightningModule.test_dataloader
        trainer.test(model)

    # ----------------
    # if have test_dataloader and NO test_step tell user to implement  test_step
    # ----------------
    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(hparams)
        model.test_dataloader = LightningModule.test_dataloader
        model.test_step = None
        trainer.test(model, test_dataloaders=model.dataloader(train=False))

    # ----------------
    # if have test_dataloader and test_step but no test_epoch_end warn user
    # ----------------
    with pytest.warns(RuntimeWarning):
        model = EvalModelTemplate(hparams)
        model.test_dataloader = LightningModule.test_dataloader
        model.test_epoch_end = None
        trainer.test(model, test_dataloaders=model.dataloader(train=False))

    # ----------------
    # if we are just testing, no need for train_dataloader, train_step, val_dataloader, and val_step
    # ----------------
    model = EvalModelTemplate(hparams)
    model.test_dataloader = LightningModule.test_dataloader
    model.train_dataloader = None
    model.train_step = None
    model.val_dataloader = None
    model.val_step = None
    trainer.test(model, test_dataloaders=model.dataloader(train=False))
