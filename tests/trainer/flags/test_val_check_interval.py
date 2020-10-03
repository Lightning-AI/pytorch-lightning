import pytest
from tests.base import SimpleModule
from pytorch_lightning.trainer import Trainer


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
@pytest.mark.parametrize('interval', [1.0, 0.25, 0.33])
def test_val_check_interval_(tmpdir, max_epochs, interval):

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
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        val_check_interval=interval,
        logger=False,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs * round(1 / interval)
