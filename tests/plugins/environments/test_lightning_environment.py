import os
from unittest import mock

from pytorch_lightning.plugins.environments import LightningEnvironment


@mock.patch.dict(os.environ, {})
def test_default_attributes():
    """ Test the default attributes when no environment variables are set. """
    env = LightningEnvironment()
    assert not env.creates_children()
    assert env.master_address() == "127.0.0.1"
    assert isinstance(env.master_port(), int)
    assert env.world_size() is None
    assert env.local_rank() == 0
    assert env.node_rank() == 0


@mock.patch.dict(os.environ, {
    "MASTER_ADDR": "1.2.3.4",
    "MASTER_PORT": "500",
    "LOCAL_RANK": "2",
    "NODE_RANK": "3",
})
def test_attributes_from_environment_variables():
    """ Test that the default cluster environment takes the attributes from the environment variables. """
    env = LightningEnvironment()
    assert env.master_address() == "1.2.3.4"
    assert env.master_port() == 500
    assert env.world_size() is None
    assert env.local_rank() == 2
    assert env.node_rank() == 3


@mock.patch.dict(os.environ, {
    "GROUP_RANK": "1",
})
def test_node_rank_from_group_rank():
    """ Test that the GROUP_RANK substitutes NODE_RANK. """
    env = LightningEnvironment()
    assert "NODE_RANK" not in os.environ
    assert env.node_rank() == 1


@mock.patch.dict(os.environ, {})
def test_random_master_port():
    """ Test randomly chosen master port when no master port was given by user. """
    env = LightningEnvironment()
    port = env.master_port()
    assert isinstance(port, int)
    # repeated calls do not generate a new port number
    assert env.master_port() == port
