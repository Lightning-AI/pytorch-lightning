import os
from unittest import mock

import pytest
from pytorch_lightning.plugins.environments import (
    KubeflowEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
    LightningEnvironment,
)


@pytest.mark.parametrize(
    ["cluster_environment_type"],
    [("kubeflow",), ("lightning",), ("slurm",), ("torchelastic",)],
)
def test_default_environ_parameters(cluster_environment_type: str):
    """
    Test for setting default environ parameters.
    """
    if cluster_environment_type == "kubeflow":
        KubeflowEnvironment()
    elif cluster_environment_type == "lightning":
        LightningEnvironment()
    elif cluster_environment_type == "slurm":
        SLURMEnvironment()
    elif cluster_environment_type == "torchelastic":
        TorchElasticEnvironment()
    assert os.environ.get("NCCL_NSOCKS_PERTHREA") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREA") == "4"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"


@mock.patch.dict(
    os.environ,
    {
        "NCCL_NSOCKS_PERTHREA": "3",
    },
)
@pytest.mark.parametrize(
    ["cluster_environment_type"],
    [("kubeflow",), ("lightning",), ("slurm",), ("torchelastic",)],
)
def test_with_not_overriden_default_environ_parameters(cluster_environment_type: str):
    """
    Test for not setting default environ parameters when parameter is already set in `os.environ`.
    """
    if cluster_environment_type == "kubeflow":
        KubeflowEnvironment()
    elif cluster_environment_type == "lightning":
        LightningEnvironment()
    elif cluster_environment_type == "slurm":
        SLURMEnvironment()
    elif cluster_environment_type == "torchelastic":
        TorchElasticEnvironment()
    assert os.environ.get("NCCL_NSOCKS_PERTHREA") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREA") == "3"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"


@pytest.mark.parametrize(
    ["cluster_environment_type"],
    [("kubeflow",), ("lightning",), ("slurm",), ("torchelastic",)],
)
def test_with_user_defined_environ_parameters(cluster_environment_type: str):
    """
    Test for overriding environ parameters when user provide `envrion_settings`.
    """
    if cluster_environment_type == "kubeflow":
        KubeflowEnvironment(environ_settings={"NCCL_NSOCKS_PERTHREA": "1"})
    elif cluster_environment_type == "lightning":
        LightningEnvironment(environ_settings={"NCCL_NSOCKS_PERTHREA": "1"})
    elif cluster_environment_type == "slurm":
        SLURMEnvironment(environ_settings={"NCCL_NSOCKS_PERTHREA": "1"})
    elif cluster_environment_type == "torchelastic":
        TorchElasticEnvironment(environ_settings={"NCCL_NSOCKS_PERTHREA": "1"})
    assert os.environ.get("NCCL_NSOCKS_PERTHREA") is not None
    assert os.environ.get("NCCL_NSOCKS_PERTHREA") == "1"
    assert os.environ.get("NCCL_SOCKET_NTHREADS") is not None
    assert os.environ.get("NCCL_SOCKET_NTHREADS") == "2"
