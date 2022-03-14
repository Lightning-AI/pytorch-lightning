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
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment, TorchElasticEnvironment
from pytorch_lightning.strategies import DDP2Strategy, DDPShardedStrategy, DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tests.helpers.runif import RunIf


def environment_combinations():
    expected = dict(global_rank=3, local_rank=1, node_rank=1, world_size=4)
    # Lightning
    variables = {"CUDA_VISIBLE_DEVICES": "0,1,2,4", "LOCAL_RANK": "1", "NODE_RANK": "1", "WORLD_SIZE": "8"}
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
        "LOCAL_WORLD_SIZE": "2",
    }
    environment = TorchElasticEnvironment()
    yield environment, variables, expected


@pytest.mark.parametrize(
    "strategy_cls",
    [DDPStrategy, DDPShardedStrategy, DDP2Strategy, pytest.param(DeepSpeedStrategy, marks=RunIf(deepspeed=True))],
)
@mock.patch("pytorch_lightning.accelerators.gpu.GPUAccelerator.is_available", return_value=True)
def test_ranks_available_manual_strategy_selection(mock_gpu_acc_available, strategy_cls):
    """Test that the rank information is readily available after Trainer initialization."""
    num_nodes = 2
    for cluster, variables, expected in environment_combinations():

        if strategy_cls == DDP2Strategy:
            expected.update(global_rank=expected["node_rank"], world_size=num_nodes)

        with mock.patch.dict(os.environ, variables):
            strategy = strategy_cls(
                parallel_devices=[torch.device("cuda", 1), torch.device("cuda", 2)], cluster_environment=cluster
            )
            trainer = Trainer(strategy=strategy, num_nodes=num_nodes)
            assert rank_zero_only.rank == expected["global_rank"]
            assert trainer.global_rank == expected["global_rank"]
            assert trainer.local_rank == expected["local_rank"]
            assert trainer.node_rank == expected["node_rank"]
            assert trainer.world_size == expected["world_size"]


@pytest.mark.parametrize(
    "trainer_kwargs",
    [
        dict(strategy="ddp", accelerator="gpu", devices=[1, 2]),
        dict(strategy="ddp_sharded", accelerator="gpu", devices=[1, 2]),
        dict(strategy="ddp2", accelerator="gpu", devices=[1, 2]),
        dict(strategy="ddp_spawn", accelerator="cpu", devices=2),
        dict(strategy="ddp_spawn", accelerator="gpu", devices=[1, 2]),
    ],
)
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=4)
def test_ranks_available_automatic_strategy_selection(mock0, mock1, trainer_kwargs):
    """Test that the rank information is readily available after Trainer initialization."""
    num_nodes = 2
    trainer_kwargs.update(num_nodes=num_nodes)

    for cluster, variables, expected in environment_combinations():

        if trainer_kwargs["strategy"] == "ddp2":
            expected.update(global_rank=expected["node_rank"], world_size=num_nodes)
        if trainer_kwargs["strategy"] == "ddp_spawn":
            if isinstance(cluster, (SLURMEnvironment, TorchElasticEnvironment)):
                # slurm and torchelastic do not work with spawn strategies
                continue
            # when using spawn, we don't reach rank > 0 until we call Trainer.fit()
            expected.update(global_rank=(expected["node_rank"] * 2), local_rank=0)

        with mock.patch.dict(os.environ, variables):
            trainer = Trainer(**trainer_kwargs)
            assert type(trainer.strategy.cluster_environment) is type(cluster)
            assert rank_zero_only.rank == expected["global_rank"]
            assert trainer.global_rank == expected["global_rank"]
            assert trainer.local_rank == expected["local_rank"]
            assert trainer.node_rank == expected["node_rank"]
            assert trainer.world_size == expected["world_size"]
