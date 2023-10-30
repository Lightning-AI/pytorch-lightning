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
from unittest import mock

import pytest
from lightning.fabric.plugins.environments import LightningEnvironment


@mock.patch.dict(os.environ, {}, clear=True)
def test_default_attributes():
    """Test the default attributes when no environment variables are set."""
    env = LightningEnvironment()
    assert not env.creates_processes_externally
    assert env.main_address == "127.0.0.1"
    assert isinstance(env.main_port, int)
    assert env.world_size() == 1
    assert env.local_rank() == 0
    assert env.node_rank() == 0


@mock.patch.dict(os.environ, {"MASTER_ADDR": "1.2.3.4", "MASTER_PORT": "500", "LOCAL_RANK": "2", "NODE_RANK": "3"})
def test_attributes_from_environment_variables():
    """Test that the default cluster environment takes the attributes from the environment variables."""
    env = LightningEnvironment()
    assert env.main_address == "1.2.3.4"
    assert env.main_port == 500
    assert env.world_size() == 1
    assert env.global_rank() == 0
    assert env.local_rank() == 2
    assert env.node_rank() == 3
    env.set_global_rank(100)
    assert env.global_rank() == 100
    env.set_world_size(100)
    assert env.world_size() == 100


@pytest.mark.parametrize(
    ("environ", "creates_processes_externally"), [({}, False), ({"LOCAL_RANK": "2"}, True), ({"NODE_RANK": "1"}, False)]
)
def test_manual_user_launch(environ, creates_processes_externally):
    """Test that the environment switches to manual user mode when LOCAL_RANK env variable detected."""
    with mock.patch.dict(os.environ, environ):
        env = LightningEnvironment()
        assert env.creates_processes_externally == creates_processes_externally


@mock.patch.dict(os.environ, {"GROUP_RANK": "1"})
def test_node_rank_from_group_rank():
    """Test that the GROUP_RANK substitutes NODE_RANK."""
    env = LightningEnvironment()
    assert "NODE_RANK" not in os.environ
    assert env.node_rank() == 1


@mock.patch.dict(os.environ, {}, clear=True)
def test_random_main_port():
    """Test randomly chosen main port when no main port was given by user."""
    env = LightningEnvironment()
    port = env.main_port
    assert isinstance(port, int)
    # repeated calls do not generate a new port number
    assert env.main_port == port


@mock.patch.dict(os.environ, {"WORLD_SIZE": "1"})
def test_teardown():
    """Test that the GROUP_RANK substitutes NODE_RANK."""
    env = LightningEnvironment()
    assert "WORLD_SIZE" in os.environ
    env.teardown()
    assert "WORLD_SIZE" not in os.environ


def test_detect():
    assert LightningEnvironment.detect()
