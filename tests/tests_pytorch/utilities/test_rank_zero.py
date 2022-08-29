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
import sys
from unittest import mock

import pytest

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import _get_rank, _rank_prefixed_message


@pytest.mark.parametrize(
    "env_vars, expected",
    [
        ({"RANK": "0"}, 1),
        ({"SLURM_PROCID": "0"}, 1),
        ({"LOCAL_RANK": "0"}, 1),
        ({"JSM_NAMESPACE_RANK": "0"}, 1),
        ({}, 1),
        ({"RANK": "1"}, None),
        ({"SLURM_PROCID": "2"}, None),
        ({"LOCAL_RANK": "3"}, None),
        ({"JSM_NAMESPACE_RANK": "4"}, None),
    ],
)
def test_rank_zero_known_environment_variables(env_vars, expected):
    """Test that rank environment variables are properly checked for rank_zero_only."""
    with mock.patch.dict(os.environ, env_vars):
        # force module reload to re-trigger the rank_zero_only.rank global computation
        sys.modules.pop("pytorch_lightning.utilities.rank_zero", None)
        from pytorch_lightning.utilities.rank_zero import rank_zero_only

        @rank_zero_only
        def foo():
            return 1

        assert foo() == expected


@pytest.mark.parametrize(
    "environ,expected_rank",
    [
        ({"JSM_NAMESPACE_RANK": "3"}, 3),
        ({"JSM_NAMESPACE_RANK": "3", "SLURM_PROCID": "2"}, 2),
        ({"JSM_NAMESPACE_RANK": "3", "SLURM_PROCID": "2", "LOCAL_RANK": "1"}, 1),
        ({"JSM_NAMESPACE_RANK": "3", "SLURM_PROCID": "2", "LOCAL_RANK": "1", "RANK": "0"}, 0),
    ],
)
def test_rank_zero_priority(environ, expected_rank):
    """Test the priority in which the rank gets determined when multiple environment variables are available."""
    with mock.patch.dict(os.environ, environ):
        assert _get_rank() == expected_rank


@pytest.mark.parametrize("trainer", [Trainer(), None])
@pytest.mark.parametrize(
    "rank_zero_only, world_size, global_rank, expected_log",
    [
        (False, 1, 0, "bar"),
        (False, 2, 0, "[rank: 0] bar"),
        (False, 2, 1, "[rank: 1] bar"),
        (True, 1, 0, "bar"),
        (True, 2, 0, "[rank: 0] bar"),
        (True, 2, 1, None),
    ],
)
def test_rank_prefixed_message_with_trainer(trainer, rank_zero_only, world_size, global_rank, expected_log):
    # set the global_rank and world_size if trainer is not None
    # or else always expect the simple logging message
    if trainer:
        trainer.strategy.global_rank = global_rank
        trainer.strategy.world_size = world_size
    else:
        expected_log = "bar"

    message = _rank_prefixed_message("bar", trainer=trainer, rank_zero_only=rank_zero_only)
    assert message == expected_log


@pytest.mark.parametrize("env_vars", [{"RANK": "0"}, {"RANK": "1"}, {"RANK": "4"}])
def test_rank_prefixed_message_with_env_vars(env_vars):
    with mock.patch.dict(os.environ, env_vars, clear=True):
        rank = _get_rank()
        message = _rank_prefixed_message("bar")

    assert message == f"[rank: {rank}] bar"
