import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin, DDPShardedPlugin, DDP2Plugin, DeepSpeedPlugin
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment, TorchElasticEnvironment
from pytorch_lightning.utilities import rank_zero_only


def environment_combinations():
    expected = dict(global_rank=3, local_rank=1, node_rank=1, world_size=4)
    # Lightning
    variables = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,4",
        "LOCAL_RANK": "1",
        "NODE_RANK": "1",
        "WORLD_SIZE": "8",
    }
    environment = LightningEnvironment()
    yield environment, variables, expected
    # SLURM
    variables = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,4",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_LOCALID": "1",
        "SLURM_NODEID": "1",
        "SLURM_PROCID": "3",
        "SLURM_NTASKS": "4",
    }
    environment = SLURMEnvironment()
    yield environment, variables, expected
    # TorchElastic
    variables = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,4",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "1",
        "RANK": "3",
        "WORLD_SIZE": "4",
    }
    environment = TorchElasticEnvironment()
    yield environment, variables, expected


@pytest.mark.parametrize("plugin_cls", [
    DDPPlugin,
    DDPShardedPlugin,
    DDP2Plugin,
    # DeepSpeedPlugin,
])
def test_ranks_avalable(plugin_cls):
    """ Test that the rank information is readily available after Trainer initialization. """
    num_nodes = 2
    for cluster, variables, expected in environment_combinations():

        if plugin_cls == DDP2Plugin:
            expected.update(global_rank=expected["node_rank"], world_size=num_nodes)

        with mock.patch.dict(os.environ, variables):
            plugin = plugin_cls(
                parallel_devices=[torch.device("cuda", 1), torch.device("cuda", 2)],
                num_nodes=num_nodes,
                cluster_environment=cluster,
            )
            trainer = Trainer(plugins=[plugin])
            assert rank_zero_only.rank == expected["global_rank"]
            assert trainer.global_rank == expected["global_rank"]
            assert trainer.local_rank == expected["local_rank"]
            assert trainer.node_rank == expected["node_rank"]
            assert trainer.world_size == expected["world_size"]
