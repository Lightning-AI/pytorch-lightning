import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


def test_optimizer_step_no_closure_raises(tmpdir):
    class TestModel(BoringModel):
        def optimizer_step(
            self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None, **_
        ):
            # does not call `optimizer_closure()`
            pass

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            class BrokenSGD(torch.optim.SGD):
                def step(self, closure=None):
                    # forgot to pass the closure
                    return super().step()

            return BrokenSGD(self.layer.parameters(), lr=0.1)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(MisconfigurationException, match="The closure hasn't been executed"):
        trainer.fit(model)
