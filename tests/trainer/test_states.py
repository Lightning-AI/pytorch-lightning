from pytorch_lightning import Trainer, Callback
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


def test_running_state_during_fit(tmpdir):
    """
        Tests that state is set to RUNNING during fit
    """

    class StateSnapshotCallback(Callback):
        def __init__(self):
            super().__init__()
            self.trainer_state = None

        def on_batch_start(self, trainer, pl_module):
            self.trainer_state = trainer.state

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    snapshot_callback = StateSnapshotCallback()

    trainer = Trainer(
        callbacks=[snapshot_callback],
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)

    assert snapshot_callback.trainer_state == TrainerState.RUNNING


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


def test_running_state_during_test(tmpdir):
    """
        Tests that state is set to RUNNING during test
    """

    class StateSnapshotCallback(Callback):
        def __init__(self):
            super().__init__()
            self.trainer_state = None

        def on_test_batch_start(self, trainer, pl_module):
            self.trainer_state = trainer.state

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    snapshot_callback = StateSnapshotCallback()

    trainer = Trainer(
        callbacks=[snapshot_callback],
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.test(model)

    assert snapshot_callback.trainer_state == TrainerState.RUNNING


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


def test_interrupt_state_on_keyboard_interrupt(tmpdir):
    """
        Tests that state is set to INTERRUPTED on KeyboardInterrupt
    """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    class InterruptCallback(Callback):
        def __init__(self):
            super().__init__()

        def on_batch_start(self, trainer, pl_module):
            raise KeyboardInterrupt

    trainer = Trainer(
        callbacks=[InterruptCallback()],
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)

    assert trainer.state == TrainerState.INTERRUPTED
