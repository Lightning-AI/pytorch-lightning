import pytest

import tests.base.utils as tutils
from pytorch_lightning import Trainer, Tuner
from tests.base import EvalModelTemplate

def test_call_order(tmpdir):
    """ Check that an warning occurs if the methods are called in a different
        order than expected """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1
    )
    tuner = Tuner(trainer)
    
    # Wrong call order, should give warning
    with pytest.warns(UserWarning):
        _ = tuner.lr_find(model)
        _ = tuner.scale_batch_size(model)