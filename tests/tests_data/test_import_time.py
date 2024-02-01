import os
from time import time


def test_lightning_data_import():
    t0 = time()
    code = "import lightning; assert 'LightningModule' in dir(lightning)"
    assert not os.system(f'python -c "{code}"')
    t1 = time()

    t2 = time()
    code = "import lightning; assert 'LightningModule' not in dir(lightning)"
    assert not os.system(f'LIGHTNING_LAZY_IMPORTS=1 python -c "{code}"')
    t3 = time()

    assert (t1 - t0) > (t3 - t2) + 1
