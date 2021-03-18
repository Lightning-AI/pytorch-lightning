import os
from unittest import mock

import pytest

from pytorch_lightning.plugins.environments import TorchElasticEnvironment


@mock.patch.dict(os.environ, {})
def test_default_attributes():
    """ Test the default attributes when no environment variables are set. """
    env = TorchElasticEnvironment()
    assert env.creates_children()
    assert env.master_address() == "127.0.0.1"
    assert env.master_port() == 12910
    assert env.world_size() is None
    with pytest.raises(KeyError):
        # local rank is required to be passed as env variable
        env.local_rank()
    assert env.node_rank() == 0


@mock.patch.dict(
    os.environ, {
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "LOCAL_RANK": "2",
        "GROUP_RANK": "3",
    }
)
def test_attributes_from_environment_variables():
    """ Test that the torchelastic cluster environment takes the attributes from the environment variables. """
    env = TorchElasticEnvironment()
    assert env.master_address() == "1.2.3.4"
    assert env.master_port() == 500
    assert env.world_size() == 20
    assert env.local_rank() == 2
    assert env.node_rank() == 3
