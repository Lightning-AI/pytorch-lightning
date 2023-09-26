# Copyright The Lightning AI team.
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
from lightning.fabric.plugins.environments import LightningEnvironment, SLURMEnvironment, TorchElasticEnvironment
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from tests_pytorch.helpers.runif import RunIf


def environment_combinations():
    expected = {"global_rank": 3, "local_rank": 1, "node_rank": 1, "world_size": 4}
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
        "SLURM_NTASKS_PER_NODE": "2",
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
        "TORCHELASTIC_RUN_ID": "1",
    }
    environment = TorchElasticEnvironment()
    yield environment, variables, expected


@RunIf(mps=False)
@pytest.mark.parametrize(
    "strategy_cls",
    [DDPStrategy, pytest.param(DeepSpeedStrategy, marks=RunIf(deepspeed=True))],
)
@mock.patch("lightning.pytorch.accelerators.cuda.CUDAAccelerator.is_available", return_value=True)
def test_ranks_available_manual_strategy_selection(_, strategy_cls):
    """Test that the rank information is readily available after Trainer initialization."""
    num_nodes = 2
    for cluster, variables, expected in environment_combinations():
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
        {"strategy": "ddp", "accelerator": "cpu", "devices": 2},
        {"strategy": "ddp_spawn", "accelerator": "cpu", "devices": 2},
        pytest.param({"strategy": "ddp", "accelerator": "gpu", "devices": [1, 2]}, marks=RunIf(mps=False)),
        pytest.param({"strategy": "ddp_spawn", "accelerator": "gpu", "devices": [1, 2]}, marks=RunIf(mps=False)),
    ],
)
def test_ranks_available_automatic_strategy_selection(cuda_count_4, trainer_kwargs):
    """Test that the rank information is readily available after Trainer initialization."""
    num_nodes = 2
    trainer_kwargs.update(num_nodes=num_nodes)

    for cluster, variables, expected in environment_combinations():
        if trainer_kwargs["strategy"] == "ddp_spawn":
            if isinstance(cluster, (SLURMEnvironment, TorchElasticEnvironment)):
                # slurm and torchelastic do not work with spawn strategies
                continue
            # when using spawn, we don't reach rank > 0 until we call Trainer.fit()
            # LOCAL_RANK is only set after we spawned
            if "LOCAL_RANK" not in variables:
                expected.update(global_rank=(expected["node_rank"] * 2), local_rank=0)

        with mock.patch.dict(os.environ, variables):
            trainer = Trainer(**trainer_kwargs)
            assert type(trainer.strategy.cluster_environment) is type(cluster)
            assert rank_zero_only.rank == expected["global_rank"]
            assert trainer.global_rank == expected["global_rank"]
            assert trainer.local_rank == expected["local_rank"]
            assert trainer.node_rank == expected["node_rank"]
            assert trainer.world_size == expected["world_size"]
