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
from tests.helpers import BoringModel


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
def test_trainer_state_while_running(tmpdir, extra_params):
    trainer = Trainer(default_root_dir=tmpdir, **extra_params, auto_lr_find=True)

    class TestModel(BoringModel):

        def __init__(self, expected_state):
            super().__init__()
            self.expected_state = expected_state
            self.lr = 0.1

        def on_batch_start(self, *_):
            assert self.trainer.state == self.expected_state

        def on_train_batch_start(self, *_):
            assert self.trainer.training

        def on_sanity_check_start(self, *_):
            assert self.trainer.sanity_checking

        def on_validation_batch_start(self, *_):
            assert self.trainer.validating or self.trainer.sanity_checking

        def on_test_batch_start(self, *_):
            assert self.trainer.testing

    model = TestModel(TrainerState.TUNING)
    trainer.tune(model)
    assert trainer.state == TrainerState.FINISHED

    model = TestModel(TrainerState.FITTING)
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED

    model = TestModel(TrainerState.TESTING)
    trainer.test(model)
    assert trainer.state == TrainerState.FINISHED


@pytest.mark.parametrize(
    "extra_params", [
        pytest.param(dict(fast_dev_run=True), id='Fast-Run'),
        pytest.param(dict(max_steps=1), id='Single-Step'),
    ]
)
def test_interrupt_state_on_keyboard_interrupt(tmpdir, extra_params):
    """ Tests that state is set to INTERRUPTED on KeyboardInterrupt """
    model = BoringModel()

    class InterruptCallback(Callback):

        def on_batch_start(self, trainer, pl_module):
            raise KeyboardInterrupt

    trainer = Trainer(callbacks=[InterruptCallback()], default_root_dir=tmpdir, **extra_params)

    trainer.fit(model)
    assert trainer.state == TrainerState.INTERRUPTED
