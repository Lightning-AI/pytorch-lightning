import pytest

from tests.base.boring_model import BoringModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


# @pytest.mark.parametrize("min_epochs", [None, 2])
# @pytest.mark.parametrize("max_epochs", [None, 3])
# @pytest.mark.parametrize("min_steps", [None, 20])
# @pytest.mark.parametrize("max_steps", [None, 100])
@pytest.mark.parametrize(
    ["min_epochs", "max_epochs", "min_steps", "max_steps"],
    [
        pytest.param(None, 5, None, None),
        pytest.param(None, None, None, 100),
        pytest.param(None, 5, None, 100),
        pytest.param(None, None, 10, 100),
        pytest.param(1, 5, None, None),
        pytest.param(1, None, None, 100),
        pytest.param(None, 5, 10, None),
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

    if trainer.max_steps and not trainer.max_epochs:
        assert trainer.global_step == trainer.max_steps
