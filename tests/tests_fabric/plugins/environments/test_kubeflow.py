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

import pytest
from lightning.fabric.plugins.environments import KubeflowEnvironment


@mock.patch.dict(os.environ, {}, clear=True)
def test_default_attributes():
    """Test the default attributes when no environment variables are set."""
    env = KubeflowEnvironment()
    assert env.creates_processes_externally

    with pytest.raises(KeyError):
        # MASTER_ADDR is required
        env.main_address
    with pytest.raises(KeyError):
        # MASTER_PORT is required
        env.main_port
    with pytest.raises(KeyError):
        # WORLD_SIZE is required
        env.world_size()
    with pytest.raises(KeyError):
        # RANK is required
        env.global_rank()
    assert env.local_rank() == 0


@mock.patch.dict(
    os.environ,
    {
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    },
)
def test_attributes_from_environment_variables(caplog):
    """Test that the torchelastic cluster environment takes the attributes from the environment variables."""
    env = KubeflowEnvironment()
    assert env.main_address == "1.2.3.4"
    assert env.main_port == 500
    assert env.world_size() == 20
    assert env.global_rank() == 1
    assert env.local_rank() == 0
    assert env.node_rank() == 1
    # setter should be no-op
    with caplog.at_level(logging.DEBUG, logger="lightning.fabric.plugins.environments"):
        env.set_global_rank(100)
    assert env.global_rank() == 1
    assert "setting global rank is not allowed" in caplog.text

    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="lightning.fabric.plugins.environments"):
        env.set_world_size(100)
    assert env.world_size() == 20
    assert "setting world size is not allowed" in caplog.text


def test_detect_kubeflow():
    """Test that the KubeflowEnvironment does not support auto-detection."""
    with pytest.raises(NotImplementedError, match="can't be detected automatically"):
        KubeflowEnvironment.detect()
