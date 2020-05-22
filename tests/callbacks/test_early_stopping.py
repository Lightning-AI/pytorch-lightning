import pytest

import tests.base.utils as tutils
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import EvalModelTemplate
from pathlib import Path


# TODO remove this test
def test_early_stopping_no_val_step(tmpdir):
    """Test that early stopping callback falls back to training metrics when no validation defined."""

    class CurrentModel(EvalModelTemplate):
        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            output.update({'my_train_metric': output['loss']})  # could be anything else
            return output

    model = CurrentModel()
    model.validation_step = None
    model.val_dataloader = None

    stopping = EarlyStopping(monitor='my_train_metric', min_delta=0.1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=stopping,
        overfit_pct=0.20,
        max_epochs=5,
    )
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'
    assert trainer.current_epoch < trainer.max_epochs


def test_pickling(tmpdir):
    import pickle
    early_stopping = EarlyStopping()
    early_stopping_pickled = pickle.dumps(early_stopping)
    early_stopping_loaded = pickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)