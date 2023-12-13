import os
import sys
from unittest import mock

import pytest
from lightning.fabric.utilities.rank_zero import _get_rank


@pytest.mark.parametrize(
    ("env_vars", "expected"),
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
def test_rank_zero_known_environment_variables(env_vars, expected, monkeypatch):
    """Test that rank environment variables are properly checked for rank_zero_only."""
    with mock.patch.dict(os.environ, env_vars):
        # force module reload to re-trigger the rank_zero_only.rank global computation
        monkeypatch.delitem(sys.modules, "lightning_utilities.core.rank_zero", raising=False)
        monkeypatch.delitem(sys.modules, "lightning.fabric.utilities.rank_zero", raising=False)
        from lightning.fabric.utilities.rank_zero import rank_zero_only

        @rank_zero_only
        def foo():
            return 1

        assert foo() == expected


@pytest.mark.parametrize(
    ("environ", "expected_rank"),
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
