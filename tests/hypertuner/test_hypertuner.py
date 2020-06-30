import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer, HyperTuner
from tests.base import EvalModelTemplate


def test_call_order(tmpdir):
    """ Check that an warning occurs if the methods are called in a different
        order than expected """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    tuner = HyperTuner(trainer)

    # Wrong call order, should give warning
    with pytest.warns(UserWarning):
        _ = tuner.lr_find(model)
        _ = tuner.scale_batch_size(model)
