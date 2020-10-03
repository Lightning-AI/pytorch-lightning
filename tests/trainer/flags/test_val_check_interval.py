import pytest
from tests.base import SimpleModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import callbacks


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
def test_val_check_interval(tmpdir, max_epochs):

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
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="avg_val_loss")
    trainer = Trainer(
        max_epochs=max_epochs,
        val_check_interval=1.0,
        logger=False,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
def test_val_check_interval(tmpdir, max_epochs):

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
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="avg_val_loss")
    trainer = Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.25,
        logger=False,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs * 4


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
def test_val_check_interval(tmpdir, max_epochs):

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
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="avg_val_loss")
    trainer = Trainer(
        max_epochs=max_epochs,
        val_check_interval=0.33,
        logger=False,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs * 3
