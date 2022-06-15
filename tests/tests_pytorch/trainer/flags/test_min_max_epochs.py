import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.utilities.warnings import PossibleUserWarning


@pytest.mark.parametrize(
    ["min_epochs", "max_epochs", "min_steps", "max_steps"],
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
def test_min_max_steps_epochs(tmpdir, min_epochs, max_epochs, min_steps, max_steps):
    """Tests that max_steps can be used without max_epochs."""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
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


def test_max_epochs_not_set_warning():
    """Test that a warning is emitted when `max_epochs` was not set by the user."""
    with pytest.warns(PossibleUserWarning, match="`max_epochs` was not set. Setting it to 1000 epochs."):
        trainer = Trainer(max_epochs=None)
        assert trainer.max_epochs == 1000
