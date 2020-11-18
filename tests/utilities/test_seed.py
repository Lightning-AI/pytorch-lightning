import os

import pytorch_lightning as pl


def test_seed_stays_same_with_multiple_seed_everything_calls():
    """
    Test to ensure that after the initial seed everything, the seed stays the same for the same run.
    """

    pl.utilities.seed.seed_everything()
    initial_seed = os.environ.get('PL_GLOBAL_SEED')

    pl.utilities.seed.seed_everything()
    seed = os.environ.get('PL_GLOBAL_SEED')

    assert initial_seed == seed
