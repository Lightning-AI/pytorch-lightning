import pytest
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning_utilities.test.warning import no_warning_call


@pytest.mark.parametrize(
    ("min_epochs", "max_epochs", "min_steps", "max_steps"),
    [
        (None, 3, None, -1),
        (None, None, None, 20),
        (None, 3, None, 20),
        (None, None, 10, 20),
        (1, 3, None, -1),
        (1, None, None, 20),
        (None, 3, 10, -1),
    ],
)
def test_min_max_steps_epochs(tmp_path, min_epochs, max_epochs, min_steps, max_steps):
    """Tests that max_steps can be used without max_epochs."""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        min_steps=min_steps,
        max_steps=max_steps,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # check training stopped at max_epochs or max_steps
    if trainer.max_steps and not trainer.max_epochs:
        assert trainer.global_step == trainer.max_steps


def test_max_epochs_not_set_warning(tmp_path):
    """Test that a warning is only emitted when `max_epochs` was not set by the user."""

    class CustomModel(BoringModel):
        def training_step(self, *args, **kwargs):
            self.trainer.should_stop = True

    match = "`max_epochs` was not set. Setting it to 1000 epochs."

    model = CustomModel()
    trainer = Trainer(logger=False, enable_checkpointing=False, max_epochs=None, limit_train_batches=1)
    with pytest.warns(PossibleUserWarning, match=match):
        trainer.fit(model)

    assert trainer.max_epochs == 1000
    assert trainer.current_epoch == 1

    with no_warning_call(expected_warning=PossibleUserWarning, match=match):
        Trainer(fast_dev_run=True)
        Trainer(fast_dev_run=1)
