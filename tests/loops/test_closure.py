import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


def test_optimizer_step_no_closure_raises(tmpdir):
    class TestModel(BoringModel):
        def optimizer_step(
            self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None, **_
        ):
            pass

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)
