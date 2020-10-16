from tests.base.boring_model import BoringModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


def test_max_steps_only(tmpdir):
    """
    Tests that max_steps can be used without max_epochs
    """
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        min_epochs=0,
        max_steps=3,
        min_steps=0,
        weights_summary=None,
    )

    result = trainer.fit(model)
    assert result == 1, "Training did not complete"

    # check training stopped at max_steps
    assert trainer.global_step == trainer.max_steps


def test_min_steps_only(tmpdir):
    """
    Tests that min_steps can be used without min_epochs
    """

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        min_steps=3,
        max_epochs=2,
        weights_summary=None,
    )

    trainer.fit(model)
