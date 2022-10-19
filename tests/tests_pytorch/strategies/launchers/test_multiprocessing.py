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
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch

from pytorch_lightning.strategies.launchers.multiprocessing import _GlobalStateSnapshot, _MultiProcessingLauncher
from tests_pytorch.helpers.runif import RunIf


@mock.patch("pytorch_lightning.strategies.launchers.multiprocessing.mp.get_all_start_methods", return_value=[])
def test_multiprocessing_launcher_forking_on_unsupported_platform(_):
    with pytest.raises(ValueError, match="The start method 'fork' is not available on this platform"):
        _MultiProcessingLauncher(strategy=Mock(), start_method="fork")


@pytest.mark.parametrize("start_method", ["spawn", pytest.param("fork", marks=RunIf(standalone=True))])
@mock.patch("pytorch_lightning.strategies.launchers.multiprocessing.mp")
def test_multiprocessing_launcher_start_method(mp_mock, start_method):
    mp_mock.get_all_start_methods.return_value = [start_method]
    launcher = _MultiProcessingLauncher(strategy=Mock(), start_method=start_method)
    launcher.launch(function=Mock())
    mp_mock.get_context.assert_called_with(start_method)
    mp_mock.start_processes.assert_called_with(
        ANY,
        args=ANY,
        nprocs=ANY,
        start_method=start_method,
    )


@pytest.mark.parametrize("start_method", ["spawn", pytest.param("fork", marks=RunIf(standalone=True))])
@mock.patch("pytorch_lightning.strategies.launchers.multiprocessing.mp")
def test_multiprocessing_launcher_restore_globals(mp_mock, start_method):
    """Test that we pass the global state snapshot to the worker function only if we are starting with 'spawn'."""
    mp_mock.get_all_start_methods.return_value = [start_method]
    launcher = _MultiProcessingLauncher(strategy=Mock(), start_method=start_method)
    launcher.launch(function=Mock())
    function_args = mp_mock.start_processes.call_args[1]["args"]
    if start_method == "spawn":
        assert len(function_args) == 6
        assert isinstance(function_args[5], _GlobalStateSnapshot)
    else:
        assert len(function_args) == 5


def test_global_state_snapshot():
    """Test the capture() and restore() methods for the global state snapshot."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(123)

    # capture the state of globals
    snapshot = _GlobalStateSnapshot.capture()

    # simulate there is a process boundary and flags get reset here
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(321)

    # restore the state of globals
    snapshot.restore()
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.initial_seed() == 123
