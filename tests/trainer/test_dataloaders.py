import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import (
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
    LightValidationDataloader,
    LightInfTrainDataloader,
    LightInfValDataloader,
    LightInfTestDataloader,
    LightZeroLenDataloader
)


def test_dataloader_config_errors(tmpdir):
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # percent check < 0

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=-0.1,
    )

    # fit model
    trainer = Trainer(**trainer_options)

    with pytest.raises(ValueError):
        trainer.fit(model)

    # percent check > 1

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=1.1,
    )

    # fit model
    trainer = Trainer(**trainer_options)

    with pytest.raises(ValueError):
        trainer.fit(model)

    # int val_check_interval > num batches

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=10000
    )

    # fit model
    trainer = Trainer(**trainer_options)

    with pytest.raises(ValueError):
        trainer.fit(model)

    # float val_check_interval > 1

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=1.1
    )

    # fit model
    trainer = Trainer(**trainer_options)

    with pytest.raises(ValueError):
        trainer.fit(model)


def test_multiple_val_dataloader(tmpdir):
    """Verify multiple val_dataloader."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValidationMultipleDataloadersMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
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

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
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
    """Verify that train dataloader can be passed to fit """
    tutils.reset_seed()

    class CurrentTestModel(LightTrainDataloader, TestModelBase):
        pass

    hparams = tutils.get_default_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # only train passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True))
    result = trainer.fit(model, **fit_options)

    assert result == 1


def test_train_val_dataloaders_passed_to_fit(tmpdir):
    """ Verify that train & val dataloader can be passed to fit """
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValStepFitSingleDataloaderMixin,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, val passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False))

    result = trainer.fit(model, **fit_options)
    assert result == 1
    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'


def test_all_dataloaders_passed_to_fit(tmpdir):
    """Verify train, val & test dataloader can be passed to fit """
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValStepFitSingleDataloaderMixin,
        LightTestFitSingleTestDataloadersMixin,
        LightEmptyTestStep,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, val and test passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False))
    test_options = dict(test_dataloaders=model._dataloader(train=False))

    result = trainer.fit(model, **fit_options)

    trainer.test(**test_options)

    assert result == 1
    assert len(trainer.val_dataloaders) == 1, \
        f'val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 1, \
        f'test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


def test_multiple_dataloaders_passed_to_fit(tmpdir):
    """Verify that multiple val & test dataloaders can be passed to fit."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightningTestModel,
        LightValStepFitMultipleDataloadersMixin,
        LightTestFitMultipleTestDataloadersMixin,
    ):
        pass

    hparams = tutils.get_default_hparams()

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    # train, multiple val and multiple test passed to fit
    model = CurrentTestModel(hparams)
    trainer = Trainer(**trainer_options)
    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=[model._dataloader(train=False),
                                        model._dataloader(train=False)])
    test_options = dict(test_dataloaders=[model._dataloader(train=False),
                                          model._dataloader(train=False)])

    results = trainer.fit(model, **fit_options)
    trainer.test(**test_options)

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

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
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
    fit_options = dict(val_dataloaders=model._dataloader(train=False))
    test_options = dict(test_dataloaders=model._dataloader(train=False))

    _ = trainer.fit(model, **fit_options)
    trainer.test(**test_options)

    assert len(trainer.val_dataloaders) == 1, \
        f'`val_dataloaders` not initiated properly, got {trainer.val_dataloaders}'
    assert len(trainer.test_dataloaders) == 1, \
        f'`test_dataloaders` not initiated properly, got {trainer.test_dataloaders}'


def test_inf_train_dataloader(tmpdir):
    """Test inf train data loader (e.g. IterableDataset)"""
    tutils.reset_seed()

    class CurrentTestModel(
        LightInfTrainDataloader,
        LightningTestModel
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # fit model
    with pytest.raises(MisconfigurationException):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            val_check_interval=0.5
        )
        trainer.fit(model)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_check_interval=50
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1


def test_inf_val_dataloader(tmpdir):
    """Test inf val data loader (e.g. IterableDataset)"""
    tutils.reset_seed()

    class CurrentTestModel(
        LightInfValDataloader,
        LightningTestModel
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # fit model
    with pytest.raises(MisconfigurationException):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            val_percent_check=0.5
        )
        trainer.fit(model)

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    result = trainer.fit(model)

    # verify training completed
    assert result == 1


def test_inf_test_dataloader(tmpdir):
    """Test inf test data loader (e.g. IterableDataset)"""
    tutils.reset_seed()

    class CurrentTestModel(
        LightInfTestDataloader,
        LightningTestModel,
        LightTestFitSingleTestDataloadersMixin
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # fit model
    with pytest.raises(MisconfigurationException):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            test_percent_check=0.5
        )
        trainer.test(model)

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    result = trainer.fit(model)
    trainer.test(model)

    # verify training completed
    assert result == 1


def test_error_on_zero_len_dataloader(tmpdir):
    """ Test that error is raised if a zero-length dataloader is defined """
    tutils.reset_seed()

    class CurrentTestModel(
        LightZeroLenDataloader,
        LightningTestModel
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # fit model
    with pytest.raises(ValueError):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            test_percent_check=0.5
        )
        trainer.fit(model)


def test_warning_with_few_workers(tmpdir):
    """ Test that error is raised if dataloader with only a few workers is used """
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        LightValStepFitSingleDataloaderMixin,
        LightTestFitSingleTestDataloadersMixin,
        LightEmptyTestStep,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2
    )

    fit_options = dict(train_dataloader=model._dataloader(train=True),
                       val_dataloaders=model._dataloader(train=False))
    test_options = dict(test_dataloaders=model._dataloader(train=False))

    trainer = Trainer(**trainer_options)

    # fit model
    with pytest.warns(UserWarning, match='train'):
        trainer.fit(model, **fit_options)

    with pytest.warns(UserWarning, match='val'):
        trainer.fit(model, **fit_options)

    with pytest.warns(UserWarning, match='test'):
        trainer.test(**test_options)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Test requires multiple GPUs')
def test_dataloader_reinit_for_subclass():

    class CustomDataLoader(torch.utils.data.DataLoader):
        def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                     batch_sampler=None, num_workers=0, collate_fn=None,
                     pin_memory=False, drop_last=False, timeout=0,
                     worker_init_fn=None, dummy_kwarg=None):
            super().__init__(dataset,
                             batch_size,
                             shuffle,
                             sampler,
                             batch_sampler,
                             num_workers,
                             collate_fn,
                             pin_memory,
                             drop_last,
                             timeout,
                             worker_init_fn)

            self.dummy_kwarg = dummy_kwarg

    trainer = Trainer(gpus=[0, 1],
                      num_nodes=1,
                      distributed_backend='ddp')

    class CustomDummyObj:
        sampler = None

    result = trainer.auto_add_sampler(CustomDummyObj(), train=True)
    assert isinstance(result, CustomDummyObj), "Wrongly reinstantiated data loader"

    result = trainer.auto_add_sampler(CustomDataLoader(list(range(1000))), train=True)
    assert isinstance(result, torch.utils.data.DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert hasattr(result, 'dummy_kwarg')
