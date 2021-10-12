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

from pytorch_lightning.plugins.environments import SLURMEnvironment


@mock.patch.dict(os.environ, {})
def test_default_attributes():
    """Test the default attributes when no environment variables are set."""
    env = SLURMEnvironment()
    assert env.creates_children()
    assert env.master_address() == "127.0.0.1"
    assert env.master_port() == 12910
    with pytest.raises(KeyError):
        # world size is required to be passed as env variable
        env.world_size()
    with pytest.raises(KeyError):
        # local rank is required to be passed as env variable
        env.local_rank()
    with pytest.raises(KeyError):
        # node_rank is required to be passed as env variable
        env.node_rank()


@mock.patch.dict(
    os.environ,
    {
        "SLURM_NODELIST": "1.1.1.1, 1.1.1.2",
        "SLURM_JOB_ID": "0001234",
        "SLURM_NTASKS": "20",
        "SLURM_LOCALID": "2",
        "SLURM_PROCID": "1",
        "SLURM_NODEID": "3",
    },
)
def test_attributes_from_environment_variables(caplog):
    """Test that the SLURM cluster environment takes the attributes from the environment variables."""
    env = SLURMEnvironment()
    assert env.master_address() == "1.1.1.1"
    assert env.master_port() == 15000 + 1234
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


@pytest.mark.parametrize(
    "slurm_node_list,expected",
    [("alpha,beta,gamma", "alpha"), ("alpha beta gamma", "alpha"), ("1.2.3.[100-110]", "1.2.3.100")],
)
def test_master_address_from_slurm_node_list(slurm_node_list, expected):
    """Test extracting the master node from different formats for the SLURM_NODELIST."""
    with mock.patch.dict(os.environ, {"SLURM_NODELIST": slurm_node_list}):
        env = SLURMEnvironment()
        assert env.master_address() == expected
