import pytest
from tests.base import SimpleModule
from pytorch_lightning.trainer import Trainer


@pytest.mark.parametrize('steps', [1, 5])
def test_trainer_max_steps(tmpdir, steps):

    model = SimpleModule()
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=False,
        limit_train_batches=1,
        limit_val_batches=1,
        max_steps=steps,
    )
    trainer.fit(model)

    assert trainer.global_step == steps


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
@pytest.mark.parametrize('freq', [0.25, 0.33, 1.0])
def test_val_check_interval(tmpdir, max_epochs, freq):

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
        val_check_interval=freq,
        logger=False,
    )
    trainer.fit(model)

    assert model.val_epoch_calls == max_epochs * round(1 / freq)
