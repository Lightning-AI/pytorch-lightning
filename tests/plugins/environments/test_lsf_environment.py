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

import pytest

from pytorch_lightning.plugins.environments import LSFEnvironment


@mock.patch.dict(os.environ, {"LSB_MCPU_HOSTS": "10.10.10.0 1 10.10.10.1 1", "LSB_JOBID": "1234"})
def test_missing_lsb_hosts():
    """Test an error when the lsb hosts list cannot be found."""
    del os.environ["LSB_MCPU_HOSTS"]
    with pytest.raises(ValueError, match="Could not find hosts in environment variable LSB_MCPU_HOSTS"):
        LSFEnvironment()


@mock.patch.dict(os.environ, {"LSB_MCPU_HOSTS": "10.10.10.0 1 10.10.10.1 1", "LSB_JOBID": "1234"})
def test_missing_lsb_job_id():
    """Test an error when the job id cannot be found."""
    del os.environ["LSB_JOBID"]
    with pytest.raises(ValueError, match="Could not find job id in environment variable LSB_JOBID"):
        LSFEnvironment()


@mock.patch.dict(
    os.environ, {"MASTER_PORT": "4321", "LSB_JOBID": "1234", "LSB_MCPU_HOSTS": "10.10.10.0 1 10.10.10.1 1"}
)
def test_manual_main_port_and_address():
    """Test a user can set the port manually through the MASTER_PORT env variable."""
    env = LSFEnvironment()
    assert env.main_port == 4321


@mock.patch.dict(
    os.environ,
    {
        "LSB_MCPU_HOSTS": "10.10.10.0 1 10.10.10.1 1 10.10.10.2 1 10.10.10.3 1",
        "LSB_JOBID": "1234",
        "JSM_NAMESPACE_SIZE": "4",
        "JSM_NAMESPACE_RANK": "3",
        "JSM_NAMESPACE_LOCAL_RANK": "1",
    },
)
def test_attributes_from_environment_variables():
    """Test that the LSF environment takes the attributes from the environment variables."""
    env = LSFEnvironment()
    assert env.creates_processes_externally
    assert env.main_address == "10.10.10.0"
    assert env.main_port == 10234
    assert env.world_size() == 4
    assert env.global_rank() == 3
    assert env.local_rank() == 1
    env.set_global_rank(100)
    assert env.global_rank() == 3
    env.set_world_size(100)
    assert env.world_size() == 4
    assert LSFEnvironment.is_using_lsf()


@mock.patch("socket.gethostname", return_value="host2")
@mock.patch.dict(os.environ, {"LSB_MCPU_HOSTS": "host0 1 host1 1 host2 1 host3 1", "LSB_JOBID": "1234"})
def test_node_rank(_):
    env = LSFEnvironment()
    assert env.node_rank() == 2
