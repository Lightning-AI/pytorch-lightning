# Copyright The Lightning AI team.
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
from unittest import mock
from unittest.mock import Mock

import pytest

from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.connectors.signal_connector import _HandlersCompose, _SignalConnector, _SignalFlag
from lightning.pytorch.utilities.exceptions import SIGTERMException
from tests_pytorch.helpers.runif import RunIf


@RunIf(skip_windows=True)
def test_signal_handlers_restored_in_teardown():
    """Test that the SignalConnector restores the previously configured handler on teardown."""
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    trainer = Trainer(plugins=SLURMEnvironment())
    connector = _SignalConnector(trainer)
    connector.register_signal_handlers()

    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    connector.teardown()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


@RunIf(skip_windows=True)
def test_sigterm_handler_can_be_added(tmp_path):
    handler_ran = False

    def handler(*_):
        nonlocal handler_ran
        handler_ran = True

    signal.signal(signal.SIGTERM, handler)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            os.kill(os.getpid(), signal.SIGTERM)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, limit_train_batches=2, limit_val_batches=0)

    assert not trainer.received_sigterm
    assert not handler_ran
    with pytest.raises(SIGTERMException):
        trainer.fit(model)
    assert trainer.received_sigterm
    assert handler_ran

    # reset the signal to system defaults
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


@RunIf(skip_windows=True)
@pytest.mark.parametrize("auto_requeue", [True, False])
@pytest.mark.parametrize("requeue_signal", [signal.SIGUSR1, signal.SIGUSR2, signal.SIGHUP] if not _IS_WINDOWS else [])
def test_auto_requeue_signal_handlers(auto_requeue, requeue_signal):
    trainer = Trainer(plugins=[SLURMEnvironment(auto_requeue=auto_requeue, requeue_signal=requeue_signal)])
    connector = _SignalConnector(trainer)
    connector.register_signal_handlers()

    sigterm_handler = signal.getsignal(signal.SIGTERM)
    assert isinstance(sigterm_handler, _HandlersCompose)
    assert len(sigterm_handler.signal_handlers) == 2
    assert sigterm_handler.signal_handlers[0] is signal.SIG_DFL
    assert isinstance(sigterm_handler.signal_handlers[1], _SignalFlag)

    if auto_requeue:
        sigusr_handler = signal.getsignal(requeue_signal)
        assert isinstance(sigusr_handler, _HandlersCompose)
        assert len(sigusr_handler.signal_handlers) == 2
        assert sigusr_handler.signal_handlers[0] is signal.SIG_DFL
        assert isinstance(sigusr_handler.signal_handlers[1], _SignalFlag)
    else:
        assert signal.getsignal(requeue_signal) is signal.SIG_DFL

    connector.teardown()


@RunIf(skip_windows=True)
@mock.patch("subprocess.run", return_value=Mock(returncode=0))
@mock.patch("lightning.pytorch.trainer.Trainer.save_checkpoint")
@mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345"})
def test_auto_requeue_job(ckpt_mock, run_mock):
    trainer = Trainer(plugins=[SLURMEnvironment()])
    connector = _SignalConnector(trainer)
    connector.requeue_flag.set()
    connector._process_signals()

    ckpt_mock.assert_called_once()
    run_mock.assert_called_once()
    assert run_mock.call_args[0][0] == ["scontrol", "requeue", "12345"]


@RunIf(skip_windows=True)
@mock.patch("subprocess.run", return_value=Mock(returncode=0))
@mock.patch("lightning.pytorch.trainer.Trainer.save_checkpoint")
@mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12346", "SLURM_ARRAY_JOB_ID": "12345", "SLURM_ARRAY_TASK_ID": "2"})
def test_auto_requeue_array_job(ckpt_mock, run_mock):
    trainer = Trainer(plugins=[SLURMEnvironment()])
    connector = _SignalConnector(trainer)
    connector.requeue_flag.set()
    connector._process_signals()

    ckpt_mock.assert_called_once()
    run_mock.assert_called_once()
    assert run_mock.call_args[0][0] == ["scontrol", "requeue", "12345_2"]


def _registering_signals():
    trainer = Trainer()
    trainer._signal_connector.register_signal_handlers()
    trainer._signal_connector.teardown()


@RunIf(skip_windows=True)
def test_no_signal_handling_in_non_main_thread():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for future in concurrent.futures.as_completed([executor.submit(_registering_signals)]):
            assert future.exception() is None


def test_sigterm_sets_flag_and_kills_subprocesses():
    trainer = Mock()
    launcher = Mock()
    trainer.strategy.launcher = launcher
    connector = _SignalConnector(trainer)

    assert not connector.received_sigterm
    connector.sigterm_flag.set()
    connector._process_signals()
    launcher.kill.assert_called_once_with(signal.SIGTERM)
    assert connector.received_sigterm

    launcher.reset_mock()
    connector.sigterm_flag.set()
    connector._process_signals()
    launcher.kill.assert_not_called()
