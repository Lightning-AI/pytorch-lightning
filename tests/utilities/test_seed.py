import os

from unittest import mock
import pytest

import pytorch_lightning.utilities.seed as seed_utils


@mock.patch.dict(os.environ, {}, clear=True)
def test_seed_stays_same_with_multiple_seed_everything_calls():
    """
    Ensure that after the initial seed everything,
    the seed stays the same for the same run.
    """
    with pytest.warns(UserWarning, match="No correct seed found"):
        seed_utils.seed_everything()
    initial_seed = os.environ.get("PL_GLOBAL_SEED")

    with pytest.warns(None) as record:
        seed_utils.seed_everything()
    assert not record  # does not warn
    seed = os.environ.get("PL_GLOBAL_SEED")

    assert initial_seed == seed


@mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "2020"}, clear=True)
def test_correct_seed_with_environment_variable():
    """
    Ensure that the PL_GLOBAL_SEED environment is read
    """
    assert seed_utils.seed_everything() == 2020


@mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "invalid"}, clear=True)
@mock.patch.object(seed_utils, attribute='_select_seed_randomly', new=lambda *_: 123)
def test_invalid_seed():
    """
    Ensure that we still fix the seed even if an invalid seed is given
    """
    with pytest.warns(UserWarning, match="No correct seed found"):
        seed = seed_utils.seed_everything()
    assert seed == 123


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(seed_utils, attribute='_select_seed_randomly', new=lambda *_: 123)
@pytest.mark.parametrize("seed", (10e9, -10e9))
def test_out_of_bounds_seed(seed):
    """
    Ensure that we still fix the seed even if an out-of-bounds seed is given
    """
    with pytest.warns(UserWarning, match="is not in bounds"):
        actual = seed_utils.seed_everything(seed)
    assert actual == 123
