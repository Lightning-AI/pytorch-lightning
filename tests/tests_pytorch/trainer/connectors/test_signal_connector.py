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
import concurrent.futures
import os
import signal
from time import sleep
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.utilities.exceptions import ExitGracefullyException
from tests_pytorch.helpers.runif import RunIf


@RunIf(skip_windows=True)
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_signal_handlers_restored_in_teardown():
    """Test that the SignalConnector restores the previously configured handler on teardown."""
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    trainer = Trainer(plugins=SLURMEnvironment())
    connector = SignalConnector(trainer)
    connector.register_signal_handlers()

    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    connector.teardown()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


@pytest.mark.parametrize("register_handler", [False, True])
@pytest.mark.parametrize("terminate_gracefully", [False, True])
@RunIf(skip_windows=True)
def test_fault_tolerant_sig_handler(register_handler, terminate_gracefully, tmpdir):

    if register_handler:

        def handler(*_):
            pass

        signal.signal(signal.SIGTERM, handler)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            if terminate_gracefully or register_handler:
                os.kill(os.getpid(), signal.SIGTERM)
                sleep(0.1)
            return super().training_step(batch, batch_idx)

    model = TestModel()

    with mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": str(int(terminate_gracefully))}):
        trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, limit_val_batches=0)
        if terminate_gracefully and not register_handler:
            with pytest.raises(ExitGracefullyException):
                trainer.fit(model)
        else:
            trainer.fit(model)
        assert trainer._terminate_gracefully == (False if register_handler else terminate_gracefully)

    # reset the signal to system defaults
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


@RunIf(skip_windows=True)
@pytest.mark.parametrize("auto_requeue", (True, False))
def test_auto_requeue_flag(auto_requeue):
    trainer = Trainer(plugins=[SLURMEnvironment(auto_requeue=auto_requeue)])
    connector = SignalConnector(trainer)
    connector.register_signal_handlers()

    if auto_requeue:
        sigterm_handlers = signal.getsignal(signal.SIGTERM).signal_handlers
        assert len(sigterm_handlers) == 1
        assert sigterm_handlers[0].__qualname__ == "SignalConnector.sigterm_handler_fn"

        sigusr1_handlers = signal.getsignal(signal.SIGUSR1).signal_handlers
        assert len(sigusr1_handlers) == 1
        assert sigusr1_handlers[0].__qualname__ == "SignalConnector.slurm_sigusr1_handler_fn"
    else:
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
        assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL

    connector.teardown()


def _registering_signals():
    trainer = Trainer()
    trainer._signal_connector.register_signal_handlers()


@RunIf(skip_windows=True)
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_signal_connector_in_thread():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for future in concurrent.futures.as_completed([executor.submit(_registering_signals)]):
            assert future.exception() is None


def signal_handler():
    pass


class SignalHandlers:
    def signal_handler(self):
        pass


@pytest.mark.parametrize(
    ["handler", "expected_return"],
    [
        (None, False),
        (signal.Handlers.SIG_IGN, True),
        (signal.Handlers.SIG_DFL, False),
        (signal_handler, True),
        (SignalHandlers().signal_handler, True),
    ],
)
def test_has_already_handler(handler, expected_return):
    """Test that the SignalConnector detects whether a signal handler is already attached."""
    with mock.patch("pytorch_lightning.trainer.connectors.signal_connector.signal.getsignal", return_value=handler):
        assert SignalConnector._has_already_handler(signal.SIGTERM) is expected_return
