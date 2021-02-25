# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.trainer.states import TrainerState
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


def test_initialize_state(tmpdir):
    """ Tests that state is INITIALIZE after Trainer creation """
    trainer = Trainer(default_root_dir=tmpdir)
    assert trainer.state == TrainerState.INITIALIZING


@pytest.mark.parametrize(
    "extra_params", [
        pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
        pytest.param(dict(max_steps=1), id='Single-Step'),
    ]
)
def test_running_state_during_fit(tmpdir, extra_params):
    """ Tests that state is set to RUNNING during fit """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    snapshot_callback = StateSnapshotCallback(snapshot_method='on_batch_start')

    trainer = Trainer(callbacks=[snapshot_callback], default_root_dir=tmpdir, **extra_params)

    trainer.fit(model)

    assert snapshot_callback.trainer_state.running()


@pytest.mark.parametrize(
    "extra_params", [
        pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
        pytest.param(dict(max_steps=1), id='Single-Step'),
    ]
)
def test_finished_state_after_fit(tmpdir, extra_params):
    """ Tests that state is FINISHED after fit """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer = Trainer(default_root_dir=tmpdir, **extra_params)

    trainer.fit(model)

    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"


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

    assert snapshot_callback.trainer_state.running()


def test_finished_state_after_test(tmpdir):
    """ Tests that state is FINISHED after fit """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.test(model)

    assert trainer.state == TrainerState.FINISHED, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    "extra_params", [
        pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
        pytest.param(dict(max_steps=1), id='Single-Step'),
    ]
)
def test_interrupt_state_on_keyboard_interrupt(tmpdir, extra_params):
    """ Tests that state is set to INTERRUPTED on KeyboardInterrupt """
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    class InterruptCallback(Callback):

        def __init__(self):
            super().__init__()

        def on_batch_start(self, trainer, pl_module):
            raise KeyboardInterrupt

    trainer = Trainer(callbacks=[InterruptCallback()], default_root_dir=tmpdir, **extra_params)

    trainer.fit(model)

    assert trainer.state == TrainerState.INTERRUPTED
