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
import os
import signal
import sys
from unittest import mock
from unittest.mock import ANY, Mock

import lightning.fabric
import pytest
from lightning.fabric.strategies.launchers.subprocess_script import (
    _HYDRA_AVAILABLE,
    _ChildProcessObserver,
    _SubprocessScriptLauncher,
)


def test_subprocess_script_launcher_interactive_compatible():
    launcher = _SubprocessScriptLauncher(Mock(), num_processes=2, num_nodes=1)
    assert not launcher.is_interactive_compatible


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.subprocess.Popen")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_subprocess_script_launcher_can_launch(*_):
    cluster_env = Mock()
    cluster_env.creates_processes_externally = False
    cluster_env.local_rank.return_value = 1
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=2, num_nodes=1)

    with pytest.raises(RuntimeError, match="attempted to launch new distributed processes with `local_rank > 0`"):
        launcher.launch(Mock())

    launcher.procs = [Mock()]  # there are already processes running
    with pytest.raises(RuntimeError, match="The launcher can only create subprocesses once"):
        launcher.launch(Mock())


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.subprocess.Popen")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_subprocess_script_launcher_external_processes(_, popen_mock):
    cluster_env = Mock()
    cluster_env.creates_processes_externally = True
    function = Mock()
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=4, num_nodes=2)
    launcher.launch(function, "positional-arg", keyword_arg=0)
    function.assert_called_with("positional-arg", keyword_arg=0)
    popen_mock.assert_not_called()


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.subprocess.Popen")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_subprocess_script_launcher_launch_processes(_, popen_mock):
    cluster_env = Mock()
    cluster_env.creates_processes_externally = False
    cluster_env.local_rank.return_value = 0
    cluster_env.main_address = "address"
    cluster_env.main_port = 1234

    function = Mock()
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=4, num_nodes=2)
    num_new_processes = launcher.num_processes - 1

    # launches n-1 new processes, the current one will participate too
    launcher.launch(function, "positional-arg", keyword_arg=0)

    calls = popen_mock.call_args_list
    assert len(calls) == num_new_processes

    # world size in child processes
    world_sizes = [int(calls[i][1]["env"]["WORLD_SIZE"]) for i in range(num_new_processes)]
    assert world_sizes == [launcher.num_processes * launcher.num_nodes] * num_new_processes

    # local rank in child processes
    local_ranks = [int(calls[i][1]["env"]["LOCAL_RANK"]) for i in range(num_new_processes)]
    assert local_ranks == list(range(1, num_new_processes + 1))

    # the current process
    assert int(os.environ["WORLD_SIZE"]) == launcher.num_processes * launcher.num_nodes
    assert int(os.environ["LOCAL_RANK"]) == 0


@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="hydra-core is required")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.subprocess.Popen")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_subprocess_script_launcher_hydra_in_use(_, popen_mock, monkeypatch):
    basic_command = Mock(return_value="basic_command")
    hydra_command = Mock(return_value=("hydra_command", "hydra_cwd"))
    monkeypatch.setattr(lightning.fabric.strategies.launchers.subprocess_script, "_basic_subprocess_cmd", basic_command)
    monkeypatch.setattr(lightning.fabric.strategies.launchers.subprocess_script, "_hydra_subprocess_cmd", hydra_command)

    def simulate_launch():
        cluster_env = Mock()
        cluster_env.creates_processes_externally = False
        cluster_env.local_rank.return_value = 0
        cluster_env.main_address = "address"
        cluster_env.main_port = 1234
        function = Mock()
        launcher = _SubprocessScriptLauncher(cluster_env, num_processes=4, num_nodes=2)
        launcher.launch(function)

    # when hydra not available
    monkeypatch.setattr(lightning.fabric.strategies.launchers.subprocess_script, "_HYDRA_AVAILABLE", False)
    simulate_launch()
    popen_mock.assert_called_with("basic_command", env=ANY, cwd=None)
    popen_mock.reset_mock()

    import hydra

    # when hydra available but not initialized
    monkeypatch.setattr(lightning.fabric.strategies.launchers.subprocess_script, "_HYDRA_AVAILABLE", True)
    HydraConfigMock = Mock()
    HydraConfigMock.initialized.return_value = False
    monkeypatch.setattr(hydra.core.hydra_config, "HydraConfig", HydraConfigMock)
    simulate_launch()
    popen_mock.assert_called_with("basic_command", env=ANY, cwd=None)
    popen_mock.reset_mock()

    # when hydra available and initialized
    monkeypatch.setattr(lightning.fabric.strategies.launchers.subprocess_script, "_HYDRA_AVAILABLE", True)
    HydraConfigMock = Mock()
    HydraConfigMock.initialized.return_value = True
    monkeypatch.setattr(hydra.core.hydra_config, "HydraConfig", HydraConfigMock)
    simulate_launch()
    popen_mock.assert_called_with("hydra_command", env=ANY, cwd="hydra_cwd")
    popen_mock.reset_mock()


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.os.kill")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.time.sleep")
def test_child_process_observer(sleep_mock, os_kill_mock):
    # Case 1: All processes are running and did not exit yet
    processes = [Mock(returncode=None), Mock(returncode=None)]
    observer = _ChildProcessObserver(main_pid=1234, child_processes=processes)
    finished = observer._run()  # call _run() directly to simulate while loop
    assert not finished

    # Case 2: All processes have finished with exit code 0 (success)
    processes = [Mock(returncode=0), Mock(returncode=0)]
    observer = _ChildProcessObserver(main_pid=1234, child_processes=processes)
    finished = observer._run()  # call _run() directly to simulate while loop
    assert finished

    # Case 3: One process has finished with exit code 1 (failure)
    processes = [Mock(returncode=0), Mock(returncode=1)]
    observer = _ChildProcessObserver(main_pid=1234, child_processes=processes)
    finished = observer._run()  # call _run() directly to simulate while loop
    assert finished
    expected_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
    processes[0].send_signal.assert_called_once_with(expected_signal)
    processes[1].send_signal.assert_called_once_with(expected_signal)
    os_kill_mock.assert_called_once_with(1234, expected_signal)

    # The main routine stops
    observer = _ChildProcessObserver(main_pid=1234, child_processes=[Mock(), Mock()])
    observer._run = Mock()
    assert not observer._finished
    observer.run()
    assert observer._finished
    sleep_mock.assert_called_once_with(5)


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.subprocess.Popen")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_validate_cluster_environment_user_settings(*_):
    """Test that the launcher calls into the cluster environment to validate the user settings."""
    cluster_env = Mock(validate_settings=Mock(side_effect=RuntimeError("test")))
    cluster_env.creates_processes_externally = True
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=2, num_nodes=1)

    with pytest.raises(RuntimeError, match="test"):
        launcher.launch(Mock())
