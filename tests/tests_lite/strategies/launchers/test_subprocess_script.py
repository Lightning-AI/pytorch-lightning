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
import os
from unittest import mock
from unittest.mock import ANY, Mock

import pytest

import lightning_lite
from lightning_lite.strategies.launchers.subprocess_script import _HYDRA_AVAILABLE, _SubprocessScriptLauncher


def test_subprocess_script_launcher_interactive_compatible():
    launcher = _SubprocessScriptLauncher(Mock(), num_processes=2, num_nodes=1)
    assert not launcher.is_interactive_compatible


@mock.patch("lightning_lite.strategies.launchers.subprocess_script.subprocess.Popen")
def test_subprocess_script_launcher_error_launching_on_non_zero_rank(popen_mock):
    cluster_env = Mock()
    cluster_env.creates_processes_externally = False
    cluster_env.local_rank.return_value = 1
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=2, num_nodes=1)
    with pytest.raises(RuntimeError, match="attempted to launch new distributed processes with `local_rank > 0`"):
        launcher.launch(Mock())


@mock.patch("lightning_lite.strategies.launchers.subprocess_script.subprocess.Popen")
def test_subprocess_script_launcher_external_processes(popen_mock):
    cluster_env = Mock()
    cluster_env.creates_processes_externally = True
    function = Mock()
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=4, num_nodes=2)
    launcher.launch(function, "positional-arg", keyword_arg=0)
    function.assert_called_with("positional-arg", keyword_arg=0)
    popen_mock.assert_not_called()


@mock.patch("lightning_lite.strategies.launchers.subprocess_script.sleep")
@mock.patch("lightning_lite.strategies.launchers.subprocess_script.subprocess.Popen")
def test_subprocess_script_launcher_launch_processes(popen_mock, _):
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
@mock.patch("lightning_lite.strategies.launchers.subprocess_script.sleep")
@mock.patch("lightning_lite.strategies.launchers.subprocess_script.subprocess.Popen")
def test_subprocess_script_launcher_hydra_in_use(popen_mock, _, monkeypatch):
    basic_command = Mock(return_value="basic_command")
    hydra_command = Mock(return_value="hydra_command")
    monkeypatch.setattr(lightning_lite.strategies.launchers.subprocess_script, "_basic_subprocess_cmd", basic_command)
    monkeypatch.setattr(lightning_lite.strategies.launchers.subprocess_script, "_hydra_subprocess_cmd", hydra_command)

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
    monkeypatch.setattr(lightning_lite.strategies.launchers.subprocess_script, "_HYDRA_AVAILABLE", False)
    simulate_launch()
    popen_mock.assert_called_with("basic_command", env=ANY)
    popen_mock.reset_mock()

    import hydra

    # when hydra available but not initialized
    monkeypatch.setattr(lightning_lite.strategies.launchers.subprocess_script, "_HYDRA_AVAILABLE", True)
    HydraConfigMock = Mock()
    HydraConfigMock.initialized.return_value = False
    monkeypatch.setattr(hydra.core.hydra_config, "HydraConfig", HydraConfigMock)
    simulate_launch()
    popen_mock.assert_called_with("basic_command", env=ANY)
    popen_mock.reset_mock()

    # when hydra available and initialized
    monkeypatch.setattr(lightning_lite.strategies.launchers.subprocess_script, "_HYDRA_AVAILABLE", True)
    HydraConfigMock = Mock()
    HydraConfigMock.initialized.return_value = True
    monkeypatch.setattr(hydra.core.hydra_config, "HydraConfig", HydraConfigMock)
    simulate_launch()
    popen_mock.assert_called_with("hydra_command", env=ANY)
    popen_mock.reset_mock()
