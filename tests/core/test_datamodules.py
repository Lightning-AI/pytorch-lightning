from pytorch_lightning import Trainer
from tests.base.datamodules import MNISTDataModule
from tests.base import EvalModelTemplate


def test_train_loop_only(tmpdir):
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = EvalModelTemplate()
    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1


def test_train_val_loop_only(tmpdir):
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = EvalModelTemplate()
    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)
    assert result == 1


def test_full_loop(tmpdir):
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        weights_summary=None,
    )
    trainer.fit(model, dm)

    # fit model
    result = trainer.fit(model)

    # test
    result = trainer.test()
    result = result[0]
    assert result['test_acc'] > 0.8

