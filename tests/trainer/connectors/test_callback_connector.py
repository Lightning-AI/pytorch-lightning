from unittest.mock import Mock

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)


def test_checkpoint_callbacks_are_last(tmpdir):
    """ Test that checkpoint callbacks always get moved to the end of the list, with preserved order. """
    checkpoint1 = ModelCheckpoint(tmpdir)
    checkpoint2 = ModelCheckpoint(tmpdir)
    lr_monitor = LearningRateMonitor()
    progress_bar = ProgressBar()

    model = Mock()
    model.configure_callbacks.return_value = []
    trainer = Trainer(callbacks=[checkpoint1, progress_bar, lr_monitor, checkpoint2])
    assert trainer.callbacks == [progress_bar, lr_monitor, checkpoint1, checkpoint2]
