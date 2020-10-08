from tests.base.boring_model import BoringModel
import os
from pytorch_lightning import Trainer
import warnings


def test_no_depre_without_epoch_end(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            return acc

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )

    with warnings.catch_warnings(record=True) as w:
        trainer.fit(model)

        for msg in w:
            assert 'should not return anything ' not in str(msg)
