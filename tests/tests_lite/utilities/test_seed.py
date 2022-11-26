import os
from unittest import mock

import pytest
import torch

import lightning_lite.utilities
from lightning_lite.utilities import seed as seed_utils
from lightning_lite.utilities.seed import _collect_rng_states, _set_rng_states


@mock.patch.dict(os.environ, {}, clear=True)
def test_seed_stays_same_with_multiple_seed_everything_calls():
    """Ensure that after the initial seed everything, the seed stays the same for the same run."""
    with pytest.warns(UserWarning, match="No seed found"):
        lightning_lite.utilities.seed.seed_everything()
    initial_seed = os.environ.get("PL_GLOBAL_SEED")

    with pytest.warns(None) as record:
        lightning_lite.utilities.seed.seed_everything()
    assert not record  # does not warn
    seed = os.environ.get("PL_GLOBAL_SEED")

    assert initial_seed == seed


@mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "2020"}, clear=True)
def test_correct_seed_with_environment_variable():
    """Ensure that the PL_GLOBAL_SEED environment is read."""
    assert lightning_lite.utilities.seed.seed_everything() == 2020


@mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "invalid"}, clear=True)
@mock.patch.object(seed_utils, attribute="_select_seed_randomly", new=lambda *_: 123)
def test_invalid_seed():
    """Ensure that we still fix the seed even if an invalid seed is given."""
    with pytest.warns(UserWarning, match="Invalid seed found"):
        seed = lightning_lite.utilities.seed.seed_everything()
    assert seed == 123


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch.object(seed_utils, attribute="_select_seed_randomly", new=lambda *_: 123)
@pytest.mark.parametrize("seed", (10e9, -10e9))
def test_out_of_bounds_seed(seed):
    """Ensure that we still fix the seed even if an out-of-bounds seed is given."""
    with pytest.warns(UserWarning, match="is not in bounds"):
        actual = lightning_lite.utilities.seed.seed_everything(seed)
    assert actual == 123


def test_reset_seed_no_op():
    """Test that the reset_seed function is a no-op when seed_everything() was not used."""
    assert "PL_GLOBAL_SEED" not in os.environ
    seed_before = torch.initial_seed()
    lightning_lite.utilities.seed.reset_seed()
    assert torch.initial_seed() == seed_before
    assert "PL_GLOBAL_SEED" not in os.environ


@pytest.mark.parametrize("workers", (True, False))
def test_reset_seed_everything(workers):
    """Test that we can reset the seed to the initial value set by seed_everything()"""
    assert "PL_GLOBAL_SEED" not in os.environ
    assert "PL_SEED_WORKERS" not in os.environ

    lightning_lite.utilities.seed.seed_everything(123, workers)
    before = torch.rand(1)
    assert os.environ["PL_GLOBAL_SEED"] == "123"
    assert os.environ["PL_SEED_WORKERS"] == str(int(workers))

    lightning_lite.utilities.seed.reset_seed()
    after = torch.rand(1)
    assert os.environ["PL_GLOBAL_SEED"] == "123"
    assert os.environ["PL_SEED_WORKERS"] == str(int(workers))
    assert torch.allclose(before, after)


def test_backward_compatibility_rng_states_dict():
    """Test that an older rng_states_dict without the "torch.cuda" key does not crash."""
    states = _collect_rng_states()
    assert "torch.cuda" in states
    states.pop("torch.cuda")
    _set_rng_states(states)
