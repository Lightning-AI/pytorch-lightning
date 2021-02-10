import pytest

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


@pytest.mark.parametrize(
    ["min_epochs", "max_epochs", "min_steps", "max_steps"],
    [
        (None, 3, None, None),
        (None, None, None, 20),
        (None, 3, None, 20),
        (None, None, 10, 20),
        (1, 3, None, None),
        (1, None, None, 20),
        (None, 3, 10, None),
    ],
)
def test_min_max_steps_epochs(tmpdir, min_epochs, max_epochs, min_steps, max_steps):
    """
    Tests that max_steps can be used without max_epochs
    """
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        min_steps=min_steps,
        max_steps=max_steps,
        weights_summary=None,
    )

    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check training stopped at max_epochs or max_steps
    if trainer.max_steps and not trainer.max_epochs:
        assert trainer.global_step == trainer.max_steps
