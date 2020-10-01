import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


# TODO: add matching messages


def test_wrong_train_setting(tmpdir):
    """
    * Test that an error is thrown when no `train_dataloader()` is defined
    * Test that an error is thrown when no `training_step()` is defined
    """
    tutils.reset_seed()
    hparams = EvalModelTemplate.get_default_hparams()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(**hparams)
        model.train_dataloader = None
        trainer.fit(model)

    with pytest.raises(MisconfigurationException):
        model = EvalModelTemplate(**hparams)
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


def test_val_loop_config(tmpdir):
    """"
    When either val loop or val data are missing raise warning
    """
    tutils.reset_seed()
    hparams = EvalModelTemplate.get_default_hparams()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    # no val data has val loop
    with pytest.warns(UserWarning):
        model = EvalModelTemplate(**hparams)
        model.validation_step = None
        trainer.fit(model)

    # has val loop but no val data
    with pytest.warns(UserWarning):
        model = EvalModelTemplate(**hparams)
        model.val_dataloader = None
        trainer.fit(model)


def test_test_loop_config(tmpdir):
    """"
    When either test loop or test data are missing
    """
    hparams = EvalModelTemplate.get_default_hparams()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    # has test loop but no test data
    with pytest.warns(UserWarning):
        model = EvalModelTemplate(**hparams)
        model.test_dataloader = None
        trainer.test(model)

    # has test data but no test loop
    with pytest.warns(UserWarning):
        model = EvalModelTemplate(**hparams)
        model.test_step = None
        trainer.test(model, test_dataloaders=model.dataloader(train=False))
