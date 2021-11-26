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
from typing import Mapping
from unittest import mock

import pytest
import torch
import torch.multiprocessing as mp

import tests.helpers.utils as tutils
from pytorch_lightning.utilities.distributed import _collect_states_on_rank_zero
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("env_vars", [{"RANK": "0"}, {"SLURM_PROCID": "0"}])
def test_rank_zero_known_cluster_envs(env_vars: Mapping[str, str]):
    """Test that SLURM environment variables are properly checked for rank_zero_only."""
    from pytorch_lightning.utilities.distributed import _get_rank, rank_zero_only

    rank_zero_only.rank = _get_rank()

    with mock.patch.dict(os.environ, env_vars):
        from pytorch_lightning.utilities.distributed import _get_rank, rank_zero_only

        rank_zero_only.rank = _get_rank()

        @rank_zero_only
        def foo():  # The return type is optional because on non-zero ranks it will not be called
            return 1

        x = foo()
        assert x == 1


@pytest.mark.parametrize("rank_key,rank", [("RANK", "1"), ("SLURM_PROCID", "2"), ("LOCAL_RANK", "3")])
def test_rank_zero_none_set(rank_key, rank):
    """Test that function is not called when rank environment variables are not global zero."""

    with mock.patch.dict(os.environ, {rank_key: rank}):
        from pytorch_lightning.utilities.distributed import _get_rank, rank_zero_only

        rank_zero_only.rank = _get_rank()

        @rank_zero_only
        def foo():
            return 1

        x = foo()
        assert x is None


def _test_collect_states(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"

    torch.cuda.set_device(f"cuda:{rank}")

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    state = {"something": torch.tensor([rank])}
    collected_state = _collect_states_on_rank_zero(state)
    assert collected_state == {1: {"something": torch.tensor([1])}, 0: {"something": torch.tensor([0])}}


@RunIf(skip_windows=True, min_gpus=2, min_torch="1.10")
def test_collect_states():
    """This test ensures state are properly collected across processes.

    This would be used to collect dataloader states as an example.
    """
    tutils.set_random_main_port()
    mp.spawn(_test_collect_states, args=(2,), nprocs=2)
