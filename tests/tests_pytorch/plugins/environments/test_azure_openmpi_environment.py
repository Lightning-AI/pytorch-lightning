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

# import pytest
from pytorch_lightning.plugins.environments import AzureOpenMPIEnvironment


# Multinode setup where AZ_BATCH_MASTER_NODE is defined
# This is the 5th process on a 2 node, 3 device per node setup
@mock.patch.dict(
    os.environ,
    {
        "AZ_BATCH_MASTER_NODE": "1.2.3.4:500",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        "OMPI_COMM_WORLD_RANK": "5",
        "OMPI_COMM_WORLD_SIZE": "6",
    },
)
def test_attributes_from_environment_variables1():
    """Test that the cluster environment takes the attributes from the environment variables."""
    env = AzureOpenMPIEnvironment(devices=3)
    assert env.main_address == "1.2.3.4"
    # assert env.main_port == 500
    assert env.world_size() == 6
    assert env.global_rank() == 5
    assert env.local_rank() == 1
    assert env.node_rank() == 1
    # env.set_global_rank(100)
    # assert env.global_rank() == 100
    # env.set_world_size(100)
    # assert env.world_size() == 100


# Multinode setup where AZ_BATCHAI_MPI_MASTER_NODE is defined, but AZ_BATCH_MASTER_NODE is not
# This is the 5th process on a 2 node, 3 device per node setup (6 processes total)
@mock.patch.dict(
    os.environ,
    {
        "AZ_BATCHAI_MPI_MASTER_NODE": "1.2.3.4",
        "OMPI_COMM_WORLD_LOCAL_RANK": "1",
        "OMPI_COMM_WORLD_RANK": "5",
        "OMPI_COMM_WORLD_SIZE": "6",
    },
)
def test_attributes_from_environment_variables2():
    """Test that the cluster environment takes the attributes from the environment variables."""
    env = AzureOpenMPIEnvironment(devices=3)
    assert env.main_address == "1.2.3.4"
    # assert env.main_port == 500
    assert env.world_size() == 6
    assert env.global_rank() == 5
    assert env.local_rank() == 1
    assert env.node_rank() == 1
    # env.set_global_rank(100)
    # assert env.global_rank() == 100
    # env.set_world_size(100)
    # assert env.world_size() == 100


def test_detect():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert not AzureOpenMPIEnvironment.detect()
    with mock.patch.dict(
        os.environ,
        {
            "AZ_BATCHAI_MPI_MASTER_NODE": "1.2.3.4",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "OMPI_COMM_WORLD_RANK": "2",
            "OMPI_COMM_WORLD_SIZE": "6",
        },
        clear=True,
    ):
        assert AzureOpenMPIEnvironment.detect()
