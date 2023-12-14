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
import logging
import os
from unittest import mock
from unittest.mock import MagicMock

import lightning.fabric.plugins.environments.mpi
import pytest
from lightning.fabric.plugins.environments import MPIEnvironment


def test_dependencies(monkeypatch):
    """Test that the MPI environment requires the `mpi4py` package."""
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError):
        MPIEnvironment()

    # pretend mpi4py is available
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", True)
    with mock.patch.dict("sys.modules", {"mpi4py": MagicMock()}):
        MPIEnvironment()


def test_detect(monkeypatch):
    """Test the detection of an MPI environment configuration."""
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", False)
    assert not MPIEnvironment.detect()

    # pretend mpi4py is available
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", True)
    mpi4py_mock = MagicMock()

    with mock.patch.dict("sys.modules", {"mpi4py": mpi4py_mock}):
        mpi4py_mock.MPI.COMM_WORLD.Get_size.return_value = 0
        assert not MPIEnvironment.detect()

        mpi4py_mock.MPI.COMM_WORLD.Get_size.return_value = 1
        assert not MPIEnvironment.detect()

        mpi4py_mock.MPI.COMM_WORLD.Get_size.return_value = 2
        assert MPIEnvironment.detect()


@mock.patch.dict(os.environ, {}, clear=True)
def test_default_attributes(monkeypatch):
    """Test the default attributes when no environment variables are set."""
    # pretend mpi4py is available
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", True)
    mpi4py_mock = MagicMock()
    with mock.patch.dict("sys.modules", {"mpi4py": mpi4py_mock}):
        env = MPIEnvironment()

    assert env._node_rank is None
    assert env._main_address is None
    assert env._main_port is None
    assert env.creates_processes_externally


def test_init_local_comm(monkeypatch):
    """Test that it can determine the node rank and local rank based on the hostnames of all participating nodes."""
    # pretend mpi4py is available
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", True)
    mpi4py_mock = MagicMock()
    hostname_mock = MagicMock()

    mpi4py_mock.MPI.COMM_WORLD.Get_size.return_value = 4
    with mock.patch.dict("sys.modules", {"mpi4py": mpi4py_mock}), mock.patch("socket.gethostname", hostname_mock):
        env = MPIEnvironment()

        hostname_mock.return_value = "host1"
        env._comm_world.gather.return_value = ["host1", "host2"]
        env._comm_world.bcast.return_value = ["host1", "host2"]
        assert env.node_rank() == 0

        env._node_rank = None
        hostname_mock.return_value = "host2"
        env._comm_world.gather.return_value = None
        env._comm_world.bcast.return_value = ["host1", "host2"]
        assert env.node_rank() == 1

        assert env._comm_local is not None
        env._comm_local.Get_rank.return_value = 33
        assert env.local_rank() == 33


def test_world_comm(monkeypatch):
    # pretend mpi4py is available
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", True)
    mpi4py_mock = MagicMock()

    with mock.patch.dict("sys.modules", {"mpi4py": mpi4py_mock}):
        env = MPIEnvironment()

        env._comm_world.Get_size.return_value = 8
        assert env.world_size() == 8
        env._comm_world.Get_rank.return_value = 3
        assert env.global_rank() == 3


def test_setters(monkeypatch, caplog):
    # pretend mpi4py is available
    monkeypatch.setattr(lightning.fabric.plugins.environments.mpi, "_MPI4PY_AVAILABLE", True)
    mpi4py_mock = MagicMock()

    with mock.patch.dict("sys.modules", {"mpi4py": mpi4py_mock}):
        env = MPIEnvironment()

    # setter should be no-op
    with caplog.at_level(logging.DEBUG, logger="lightning.fabric.plugins.environments"):
        env.set_global_rank(100)
    assert "setting global rank is not allowed" in caplog.text

    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="lightning.fabric.plugins.environments"):
        env.set_world_size(100)
    assert "setting world size is not allowed" in caplog.text
