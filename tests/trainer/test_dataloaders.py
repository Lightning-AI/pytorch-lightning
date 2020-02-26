import pytest

import tests.models.utils as tutils
from pytorch_lightning import Trainer
from tests.models import (
    TestModelBase,
    LightningTestModel,
    LightEmptyTestStep,
    LightValidationMultipleDataloadersMixin,
    LightTestMultipleDataloadersMixin,
    LightTestFitSingleTestDataloadersMixin,
    LightTestFitMultipleTestDataloadersMixin,
    LightValStepFitMultipleDataloadersMixin,
    LightValStepFitSingleDataloaderMixin,
    LightTrainDataloader,
)
from pytorch_lightning.utilities.debugging import MisconfigurationException


def test_multiple_val_dataloader(tmpdir):
    """Verify multiple val_dataloader."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValidationMultipleDataloadersMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=1.0,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    # verify there are 2 val loaders
    assert len(trainer.val_dataloaders) == 2, \
        'Multiple val_dataloaders not initiated properly'

    # make sure predictions are good for each val set
    for dataloader in trainer.val_dataloaders:
        tutils.run_prediction(dataloader, trainer.model)


def test_multiple_test_dataloader(tmpdir):
    """Verify multiple test_dataloader."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightTestMultipleDataloadersMixin,
        LightEmptyTestStep,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model)
    trainer.test()

    # verify there are 2 val loaders
    assert len(trainer.test_dataloaders) == 2, \
        'Multiple test_dataloaders not initiated properly'

    # make sure predictions are good for each test set
    for dataloader in trainer.test_dataloaders:
        tutils.run_prediction(dataloader, trainer.model)

    # run the test method
    trainer.test()


def test_train_dataloaders_passed_to_fit(tmpdir):
    """ Verify that train dataloader can be passed to fit """
    tutils.reset_seed()

    class CurrentTestModel(LightTrainDataloader, TestModelBase):
        pass

    hparams = tutils.get_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # only train passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True))
    results = trainer.fit(model, **fit_options)


def test_train_val_dataloaders_passed_to_fit(tmpdir):
    """ Verify that train & val dataloader can be passed to fit """
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValStepFitSingleDataloaderMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, val passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False))

    results = trainer.fit(model, **fit_options)
    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'


def test_all_dataloaders_passed_to_fit(tmpdir):
    """ Verify train, val & test dataloader can be passed to fit """
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValStepFitSingleDataloaderMixin,
        LightTestFitSingleTestDataloadersMixin,
        LightEmptyTestStep,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, val and test passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False),
                       test_dataloaders=model._dataloader(train=False))

    results = trainer.fit(model, **fit_options)

    trainer.test()

    assert len(trainer.val_dataloaders) == 1, \
        f'"val_dataloaders` not initiated properly, got {trainer.val_dataloaders}"
    assert len(trainer.test_dataloaders) == 1, \
        f'"test_dataloaders` not initiated properly, got {trainer.test_dataloaders}"


def test_multiple_dataloaders_passed_to_fit(tmpdir):
    """Verify that multiple val & test dataloaders can be passed to fit."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightningTestModel,
        LightValStepFitMultipleDataloadersMixin,
        LightTestFitMultipleTestDataloadersMixin,
    ):
        pass

    hparams = tutils.get_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, multiple val and multiple test passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=[model._dataloader(train=False),
                                        model._dataloader(train=False)],
                       test_dataloaders=[model._dataloader(train=False),
                                         model._dataloader(train=False)])
    results = trainer.fit(model, **fit_options)
    trainer.test()

    assert len(trainer.val_dataloaders) == 2, \
        f'Multiple `val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 2, \
        f'Multiple `test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


def test_mixing_of_dataloader_options(tmpdir):
    """Verify that dataloaders can be passed to fit"""
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValStepFitSingleDataloaderMixin,
        LightTestFitSingleTestDataloadersMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # fit model
    trainer = Trainer(**trainer_options)
    fit_options = dict(val_dataloaders=model._dataloader(train=False))
    results = trainer.fit(model, **fit_options)

    # fit model
    trainer = Trainer(**trainer_options)
    fit_options = dict(val_dataloaders=model._dataloader(train=False),
                       test_dataloaders=model._dataloader(train=False))
    _ = trainer.fit(model, **fit_options)
    trainer.test()

    assert len(trainer.val_dataloaders) == 1, \
        f"`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}"
    assert len(trainer.test_dataloaders) == 1, \
        f'"test_dataloaders` not initiated properly, got {trainer.test_dataloaders}"


def test_inf_train_dataloader(tmpdir):
    """Test inf train data loader (e.g. IterableDataset)"""
    tutils.reset_seed()

    class CurrentTestModel(LightningTestModel):
        def train_dataloader(self):
            dataloader = self._dataloader(train=True)

            class CustomInfDataLoader:
                def __init__(self, dataloader):
                    self.dataloader = dataloader
                    self.iter = iter(dataloader)
                    self.count = 0

                def __iter__(self):
                    self.count = 0
                    return self

                def __next__(self):
                    if self.count >= 5:
                        raise StopIteration
                    self.count = self.count + 1
                    try:
                        return next(self.iter)
                    except StopIteration:
                        self.iter = iter(self.dataloader)
                        return next(self.iter)

            return CustomInfDataLoader(dataloader)

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # fit model
    with pytest.raises(MisconfigurationException):
        trainer = Trainer(
            default_save_path=tmpdir,
            max_epochs=1,
            val_check_interval=0.5
        )
        trainer.fit(model)

    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1,
        val_check_interval=50,
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1
