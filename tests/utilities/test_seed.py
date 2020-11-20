import os

import pytest

import pytorch_lightning.utilities.seed as seed_utils


def test_seed_stays_same_with_multiple_seed_everything_calls():
    """
    Ensure that after the initial seed everything,
    the seed stays the same for the same run.
    """
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]

    with pytest.warns(UserWarning, match="No correct seed found"):
        seed_utils.seed_everything()
    initial_seed = os.environ.get("PL_GLOBAL_SEED")

    with pytest.warns(None) as record:
        seed_utils.seed_everything()
    assert not record  # does not warn
    seed = os.environ.get("PL_GLOBAL_SEED")

    assert initial_seed == seed
    del os.environ["PL_GLOBAL_SEED"]


def test_correct_seed_with_environment_variable(monkeypatch):
    """
    Ensure that the PL_GLOBAL_SEED environment is read
    """
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    expected = 2020
    monkeypatch.setenv("PL_GLOBAL_SEED", str(expected))
    assert seed_utils.seed_everything() == expected
    del os.environ["PL_GLOBAL_SEED"]


def test_invalid_seed(monkeypatch):
    """
    Ensure that we still fix the seed even if an invalid seed is given
    """
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    expected = 123
    monkeypatch.setenv("PL_GLOBAL_SEED", "invalid")
    monkeypatch.setattr(seed_utils, "_select_seed_randomly", lambda *_: expected)
    with pytest.warns(UserWarning, match="No correct seed found"):
        seed = seed_utils.seed_everything()
    assert seed == expected
    del os.environ["PL_GLOBAL_SEED"]


@pytest.mark.parametrize("seed", (10e9, -10e9))
def test_out_of_bounds_seed(monkeypatch, seed):
    """
    Ensure that we still fix the seed even if an out-of-bounds seed is given
    """
    if "PL_GLOBAL_SEED" in os.environ:
        del os.environ["PL_GLOBAL_SEED"]
    expected = 123
    monkeypatch.setattr(seed_utils, "_select_seed_randomly", lambda *_: expected)
    with pytest.warns(UserWarning, match="is not in bounds"):
        actual = seed_utils.seed_everything(seed)
    assert actual == expected
    del os.environ["PL_GLOBAL_SEED"]
