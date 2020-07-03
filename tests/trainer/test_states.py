from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import TrainerState
from tests.base import EvalModelTemplate


def test_initialize_state(tmpdir):
    """
    Tests that state is INITIALIZE after Trainer creation
    """
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    assert trainer.state == TrainerState.INITIALIZE


def test_finished_state_after_fit(tmpdir):
    """
        Tests that state is FINISHED after fit
    """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)

    assert trainer.state == TrainerState.FINISHED


def test_finished_state_after_test(tmpdir):
    """
        Tests that state is FINISHED after fit
    """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.test(model)

    assert trainer.state == TrainerState.FINISHED
