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
