import pytest
import os
from tests.base import SimpleModule
from pytorch_lightning.trainer import Trainer


os.environ['PL_DEV_DEBUG'] = '1'

@pytest.mark.parametrize('max_epochs', [1, 2, 3])
def test_val_check_interval_1(tmpdir, max_epochs):

    class TestModel(SimpleModule):
        def __init__(self):
            super().__init__()
            self.train_epoch_calls = 0
            self.val_epoch_calls = 0

        def on_train_epoch_start(self) -> None:
            self.train_epoch_calls += 1

        def on_validation_epoch_start(self) -> None:
            if not self.trainer.running_sanity_check:
                self.val_epoch_calls += 1

    model = TestModel()
    trainer = Trainer(
        max_epochs=max_epochs,
        val_check_interval=1.0,
        logger=False,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
def test_val_check_interval_quarter(tmpdir, max_epochs):

    class TestModel(SimpleModule):
        def __init__(self):
            super().__init__()
            self.train_epoch_calls = 0
            self.val_epoch_calls = 0

        def on_train_epoch_start(self) -> None:
            self.train_epoch_calls += 1

        def on_validation_epoch_start(self) -> None:
            if not self.trainer.running_sanity_check:
                self.val_epoch_calls += 1

    model = TestModel()
    trainer = Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.25,
        logger=False,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs * 4


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
def test_val_check_interval_third(tmpdir, max_epochs):

    class TestModel(SimpleModule):
        def __init__(self):
            super().__init__()
            self.train_epoch_calls = 0
            self.val_epoch_calls = 0

        def on_train_epoch_start(self) -> None:
            self.train_epoch_calls += 1

        def on_validation_epoch_start(self) -> None:
            if not self.trainer.running_sanity_check:
                self.val_epoch_calls += 1

    model = TestModel()
    trainer = Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.33,
        logger=False,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs * 3
