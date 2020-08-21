import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.hypertuner import HyperTuner
from tests.base import EvalModelTemplate


@pytest.mark.parametrize(['first_method', 'second_method'],[
    pytest.param('lr_find','scale_batch_size'),
    pytest.param('n_worker_search', 'scale_batch_size')
])
def test_call_order(tmpdir, first_method, second_method):
    """ Check that an warning occurs if the methods are called in a different
        order than expected """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )
    tuner = HyperTuner(trainer, model)

    # Wrong call order, should give warning
    with pytest.warns(UserWarning):
        _ = getattr(tuner, first_method)()
        _ = getattr(tuner, second_method)()
