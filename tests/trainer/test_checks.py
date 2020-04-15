import pytest

import tests.base.utils as tutils
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import (
    TestModelBase,
    LightValidationDataloader,
    LightTestDataloader,
    LightValidationStepMixin,
    LightValStepFitSingleDataloaderMixin,
    LightTrainDataloader,
    LightTestStepMixin,
    LightTestFitMultipleTestDataloadersMixin,
)


def test_error_on_no_train_step(tmpdir):
    """ Test that an error is thrown when no `training_step()` is defined """
    tutils.reset_seed()

    class CurrentTestModel(LightningModule):
        def forward(self, x):
            pass

    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1)
    trainer = Trainer(**trainer_options)

    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel()
        trainer.fit(model)


def test_error_on_no_train_dataloader(tmpdir):
    """ Test that an error is thrown when no `training_dataloader()` is defined """
    tutils.reset_seed()
    hparams = tutils.get_default_hparams()

    class CurrentTestModel(TestModelBase):
        pass

    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1)
    trainer = Trainer(**trainer_options)

    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel(hparams)
        trainer.fit(model)


def test_error_on_no_configure_optimizers(tmpdir):
    """ Test that an error is thrown when no `configure_optimizers()` is defined """
    tutils.reset_seed()

    class CurrentTestModel(LightTrainDataloader, LightningModule):
        def forward(self, x):
            pass

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            pass

    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1)
    trainer = Trainer(**trainer_options)

    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel()
        trainer.fit(model)


def test_warning_on_wrong_validation_settings(tmpdir):
    """ Test the following cases related to validation configuration of model:
        * error if `val_dataloader()` is overriden but `validation_step()` is not
        * if both `val_dataloader()` and `validation_step()` is overriden,
            throw warning if `val_epoch_end()` is not defined
        * error if `validation_step()` is overriden but `val_dataloader()` is not
    """
    tutils.reset_seed()
    hparams = tutils.get_default_hparams()

    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1)
    trainer = Trainer(**trainer_options)

    class CurrentTestModel(LightTrainDataloader,
                           LightValidationDataloader,
                           TestModelBase):
        pass

    # check val_dataloader -> val_step
    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel(hparams)
        trainer.fit(model)

    class CurrentTestModel(LightTrainDataloader,
                           LightValidationStepMixin,
                           TestModelBase):
        pass

    # check val_dataloader + val_step -> val_epoch_end
    with pytest.warns(RuntimeWarning):
        model = CurrentTestModel(hparams)
        trainer.fit(model)

    class CurrentTestModel(LightTrainDataloader,
                           LightValStepFitSingleDataloaderMixin,
                           TestModelBase):
        pass

    # check val_step -> val_dataloader
    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel(hparams)
        trainer.fit(model)


def test_warning_on_wrong_test_settigs(tmpdir):
    """ Test the following cases related to test configuration of model:
        * error if `test_dataloader()` is overriden but `test_step()` is not
        * if both `test_dataloader()` and `test_step()` is overriden,
            throw warning if `test_epoch_end()` is not defined
        * error if `test_step()` is overriden but `test_dataloader()` is not
    """
    tutils.reset_seed()
    hparams = tutils.get_default_hparams()

    trainer_options = dict(default_root_dir=tmpdir, max_epochs=1)
    trainer = Trainer(**trainer_options)

    class CurrentTestModel(LightTrainDataloader,
                           LightTestDataloader,
                           TestModelBase):
        pass

    # check test_dataloader -> test_step
    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel(hparams)
        trainer.fit(model)

    class CurrentTestModel(LightTrainDataloader,
                           LightTestStepMixin,
                           TestModelBase):
        pass

    # check test_dataloader + test_step -> test_epoch_end
    with pytest.warns(RuntimeWarning):
        model = CurrentTestModel(hparams)
        trainer.fit(model)

    class CurrentTestModel(LightTrainDataloader,
                           LightTestFitMultipleTestDataloadersMixin,
                           TestModelBase):
        pass

    # check test_step -> test_dataloader
    with pytest.raises(MisconfigurationException):
        model = CurrentTestModel(hparams)
        trainer.fit(model)
