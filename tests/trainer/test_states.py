import pytest

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.trainer.states import TrainerState, trainer_state
from tests.base import EvalModelTemplate


class StateSnapshotCallback(Callback):
    """ Allows to shapshot the state inside a particular trainer method. """

    def __init__(self, snapshot_method: str):
        super().__init__()
        assert snapshot_method in ['on_batch_start', 'on_test_batch_start']
        self.snapshot_method = snapshot_method
        self.trainer_state = None

    def on_batch_start(self, trainer, pl_module):
        if self.snapshot_method == 'on_batch_start':
            self.trainer_state = trainer.state

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self.snapshot_method == 'on_test_batch_start':
            self.trainer_state = trainer.state


def test_state_decorator_nothing_passed(tmpdir):
    """ Test that state is not changed if nothing is passed to a decorator"""

    @trainer_state()
    def test_method(self):
        return self.state

    trainer = Trainer(default_root_dir=tmpdir)
    trainer.state = TrainerState.INITIALIZING

    snapshot_state = test_method(trainer)

    assert snapshot_state == TrainerState.INITIALIZING
    assert trainer.state == TrainerState.INITIALIZING


def test_state_decorator_entering_only(tmpdir):
    """ Tests that state is set to entering inside a run function and restored to the previous value after. """

    @trainer_state(entering=TrainerState.RUNNING)
    def test_method(self):
        return self.state

    trainer = Trainer(default_root_dir=tmpdir)
    trainer.state = TrainerState.INITIALIZING

    snapshot_state = test_method(trainer)

    assert snapshot_state == TrainerState.RUNNING
    assert trainer.state == TrainerState.INITIALIZING


def test_state_decorator_exiting_only(tmpdir):
    """ Tests that state is not changed inside a run function and set to `exiting` after. """

    @trainer_state(exiting=TrainerState.FINISHED)
    def test_method(self):
        return self.state

    trainer = Trainer(default_root_dir=tmpdir)
    trainer.state = TrainerState.INITIALIZING

    snapshot_state = test_method(trainer)

    assert snapshot_state == TrainerState.INITIALIZING
    assert trainer.state == TrainerState.FINISHED


def test_state_decorator_entering_and_exiting(tmpdir):
    """ Tests that state is set to `entering` inside a run function and set ot `exiting` after. """

    @trainer_state(entering=TrainerState.RUNNING, exiting=TrainerState.FINISHED)
    def test_method(self):
        return self.state

    trainer = Trainer(default_root_dir=tmpdir)
    trainer.state = TrainerState.INITIALIZING

    snapshot_state = test_method(trainer)

    assert snapshot_state == TrainerState.RUNNING
    assert trainer.state == TrainerState.FINISHED


def test_state_decorator_interrupt(tmpdir):
    """ Tests that state remains `INTERRUPTED` is its set in run function. """

    @trainer_state(exiting=TrainerState.FINISHED)
    def test_method(self):
        self.state = TrainerState.INTERRUPTED

    trainer = Trainer(default_root_dir=tmpdir)
    trainer.state = TrainerState.INITIALIZING

    test_method(trainer)
    assert trainer.state == TrainerState.INTERRUPTED


def test_initialize_state(tmpdir):
    """ Tests that state is INITIALIZE after Trainer creation """
    trainer = Trainer(default_root_dir=tmpdir)
    assert trainer.state == TrainerState.INITIALIZING


@pytest.mark.parametrize("extra_params", [
    pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
    pytest.param(dict(max_steps=1), id='Single-Step'),
])
def test_running_state_during_fit(tmpdir, extra_params):
    """ Tests that state is set to RUNNING during fit """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    snapshot_callback = StateSnapshotCallback(snapshot_method='on_batch_start')

    trainer = Trainer(
        callbacks=[snapshot_callback],
        default_root_dir=tmpdir,
        **extra_params
    )

    trainer.fit(model)

    assert snapshot_callback.trainer_state == TrainerState.RUNNING


@pytest.mark.parametrize("extra_params", [
    pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
    pytest.param(dict(max_steps=1), id='Single-Step'),
])
def test_finished_state_after_fit(tmpdir, extra_params):
    """ Tests that state is FINISHED after fit """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer = Trainer(
        default_root_dir=tmpdir,
        **extra_params
    )

    trainer.fit(model)

    assert trainer.state == TrainerState.FINISHED


def test_running_state_during_test(tmpdir):
    """ Tests that state is set to RUNNING during test """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    snapshot_callback = StateSnapshotCallback(snapshot_method='on_test_batch_start')

    trainer = Trainer(
        callbacks=[snapshot_callback],
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.test(model)

    assert snapshot_callback.trainer_state == TrainerState.RUNNING


def test_finished_state_after_test(tmpdir):
    """ Tests that state is FINISHED after fit """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.test(model)

    assert trainer.state == TrainerState.FINISHED


@pytest.mark.parametrize("extra_params", [
    pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
    pytest.param(dict(max_steps=1), id='Single-Step'),
])
def test_interrupt_state_on_keyboard_interrupt(tmpdir, extra_params):
    """ Tests that state is set to INTERRUPTED on KeyboardInterrupt """
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
        **extra_params
    )

    trainer.fit(model)

    assert trainer.state == TrainerState.INTERRUPTED
