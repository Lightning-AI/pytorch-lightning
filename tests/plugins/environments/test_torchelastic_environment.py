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
import logging
import os
from unittest import mock

import pytest

from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from tests.helpers.runif import RunIf


@mock.patch.dict(os.environ, {}, clear=True)
def test_default_attributes():
    """Test the default attributes when no environment variables are set."""
    env = TorchElasticEnvironment()
    assert env.creates_processes_externally
    assert env.main_address == "127.0.0.1"
    assert env.main_port == 12910
    with pytest.raises(KeyError):
        # world size is required to be passed as env variable
        env.world_size()
    with pytest.raises(KeyError):
        # local rank is required to be passed as env variable
        env.local_rank()
    assert env.node_rank() == 0


@mock.patch.dict(
    os.environ,
    {
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
        "LOCAL_RANK": "2",
        "GROUP_RANK": "3",
    },
)
def test_attributes_from_environment_variables(caplog):
    """Test that the torchelastic cluster environment takes the attributes from the environment variables."""
    env = TorchElasticEnvironment()
    assert env.main_address == "1.2.3.4"
    assert env.main_port == 500
    assert env.world_size() == 20
    assert env.global_rank() == 1
    assert env.local_rank() == 2
    assert env.node_rank() == 3
    # setter should be no-op
    with caplog.at_level(logging.DEBUG, logger="pytorch_lightning.plugins.environments"):
        env.set_global_rank(100)
    assert env.global_rank() == 1
    assert "setting global rank is not allowed" in caplog.text

    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger="pytorch_lightning.plugins.environments"):
        env.set_world_size(100)
    assert env.world_size() == 20
    assert "setting world size is not allowed" in caplog.text


@RunIf(max_torch="1.9.0")
def test_detect_before_1_9_1():
    """Test the detection of a torchelastic environment configuration before 1.9.1."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not TorchElasticEnvironment.detect()

    with mock.patch.dict(
        os.environ,
        {
            "RANK": "",
            "GROUP_RANK": "",
            "LOCAL_RANK": "",
            "LOCAL_WORLD_SIZE": "",
        },
    ):
        assert TorchElasticEnvironment.detect()


@RunIf(min_torch="1.9.1")
def test_detect_after_1_9_1():
    """Test the detection of a torchelastic environment configuration after 1.9.1."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not TorchElasticEnvironment.detect()

    with mock.patch.dict(
        os.environ,
        {
            "TORCHELASTIC_RUN_ID": "",
        },
    ):
        assert TorchElasticEnvironment.detect()
