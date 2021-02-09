from unittest.mock import Mock

import torch

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ProgressBar
from tests.helpers import BoringModel


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


class StatefulCallback0(Callback):

    def on_save_checkpoint(self, trainer, pl_module):
        return {"content0": 0}


class StatefulCallback1(Callback):

    def on_save_checkpoint(self, trainer, pl_module):
        return {"content1": 1}


def test_all_callback_states_saved_before_checkpoint_callback(tmpdir):
    """ Test that all callback states get saved even if the ModelCheckpoint is not given as last. """

    callback0 = StatefulCallback0()
    callback1 = StatefulCallback1()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename="all_states")
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        limit_val_batches=1,
        callbacks=[callback0, checkpoint_callback, callback1]
    )
    trainer.fit(model)

    ckpt = torch.load(str(tmpdir / "all_states.ckpt"))
    state0 = ckpt["callbacks"][type(callback0)]
    state1 = ckpt["callbacks"][type(callback1)]
    assert "content0" in state0 and state0["content0"] == 0
    assert "content1" in state1 and state1["content1"] == 1
    assert type(checkpoint_callback) in ckpt["callbacks"]
