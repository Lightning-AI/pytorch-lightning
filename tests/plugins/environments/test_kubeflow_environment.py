import logging
import os
from unittest import mock

import pytest

from pytorch_lightning.plugins.environments import KubeflowEnvironment


@mock.patch.dict(os.environ, {})
def test_default_attributes():
    """ Test the default attributes when no environment variables are set. """
    env = KubeflowEnvironment()
    assert env.creates_children()

    with pytest.raises(KeyError):
        # MASTER_ADDR is required
        env.master_address()
    with pytest.raises(KeyError):
        # MASTER_PORT is required
        env.master_port()
    with pytest.raises(KeyError):
        # WORLD_SIZE is required
        env.world_size()
    with pytest.raises(KeyError):
        # RANK is required
        env.global_rank()
    assert env.local_rank() == 0


@mock.patch.dict(
    os.environ, {
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    }
)
def test_attributes_from_environment_variables(caplog):
    """ Test that the torchelastic cluster environment takes the attributes from the environment variables. """
    env = KubeflowEnvironment()
    assert env.master_address() == "1.2.3.4"
    assert env.master_port() == 500
    assert env.world_size() == 20
    assert env.global_rank() == 1
    assert env.local_rank() == 0
    assert env.node_rank() == 1
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


@mock.patch.dict(
    os.environ, {
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    }
)
def test_is_using_kubeflow():
    assert KubeflowEnvironment.is_using_kubeflow()


@mock.patch.dict(
    os.environ, {
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
        "GROUP_RANK": "1",
    }
)
def test_is_using_kubeflow_torchelastic():
    assert not KubeflowEnvironment.is_using_kubeflow()
