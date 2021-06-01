import os
from unittest import mock

import pytest

from pytorch_lightning.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)


def _instantiate_cluster_environment(cluster_environment_class, environ_settings=None):
    cluster_environment = cluster_environment_class(environ_settings=environ_settings)
    return cluster_environment


@pytest.mark.parametrize(
    "cluster_environment_class",
    [
        KubeflowEnvironment,
        LightningEnvironment,
        SLURMEnvironment,
        TorchElasticEnvironment,
    ],
)
def test_set_default_environ_parameters(cluster_environment_class):
    """
    Test for setting default environ parameters.
    """
    _instantiate_cluster_environment(cluster_environment_class)
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") == "4"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"


@mock.patch.dict(
    os.environ,
    {
        "NCCL_NSOCKS_PERTHREAD": "3",
    },
)
@pytest.mark.parametrize(
    "cluster_environment_class",
    [
        KubeflowEnvironment,
        LightningEnvironment,
        SLURMEnvironment,
        TorchElasticEnvironment,
    ],
)
def test_not_overriden_environ_parameters(cluster_environment_class):
    """
    Test for not setting default environ parameters when parameter is already set in `os.environ`.
    """
    _instantiate_cluster_environment(cluster_environment_class)
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") == "3"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"


@pytest.mark.parametrize(
    "cluster_environment_class",
    [
        KubeflowEnvironment,
        LightningEnvironment,
        SLURMEnvironment,
        TorchElasticEnvironment,
    ],
)
def test_with_user_defined_environ_parameters(cluster_environment_class):
    """
    Test for overriding environ parameters when user provide `envrion_settings`.
    """
    _instantiate_cluster_environment(cluster_environment_class, environ_settings={"NCCL_NSOCKS_PERTHREAD": "1"})
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREAD") == "1"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"
