import os.path
from unittest.mock import Mock

import pytest
from core.callbacks import PLAppArtifactsTracker, PLAppProgressTracker, PLAppSummary
from core.components.script_runner import ScriptRunner

from lightning.app.storage import Path
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger


@pytest.mark.parametrize("rank", (0, 1))
def test_progress_tracker_enabled(rank):
    trainer = Mock()
    trainer.global_rank = rank
    trainer.is_global_zero = rank == 0
    work = Mock()
    tracker = PLAppProgressTracker(work)
    assert not tracker.is_enabled
    tracker.setup(trainer, Mock(), Mock())
    assert tracker.is_enabled == trainer.is_global_zero


def test_summary_callback_tracks_hyperparameters():
    class ModelWithParameters(LightningModule):
        def __init__(self, float_arg=0.1, int_arg=5, bool_arg=True, string_arg="string"):
            super().__init__()
            self.save_hyperparameters()

    model = ModelWithParameters()
    work = Mock()
    summary = PLAppSummary(work)
    trainer = Trainer(max_epochs=22, callbacks=[summary])  # this triggers the `Callback.on_init_end` hook
    summary.setup(trainer, model)
    assert work.model_hparams == {
        "float_arg": "0.1",
        "int_arg": "5",
        "bool_arg": "True",
        "string_arg": "string",
    }

    assert work.trainer_hparams["max_epochs"] == "22"
    assert work.trainer_hparams["logger"] == "True"
    assert "ModelCheckpoint" in work.trainer_hparams["callbacks"]
    assert "PLAppSummary" in work.trainer_hparams["callbacks"]


def test_artifacts_tracker(tmpdir):
    work = ScriptRunner(root_path=os.path.dirname(__file__), script_path=__file__)
    tracker = PLAppArtifactsTracker(work=work)
    trainer = Mock()

    trainer.loggers = []
    trainer.default_root_dir = "default_root_dir"
    tracker.setup(trainer=trainer, pl_module=Mock())
    assert work.log_dir == Path("default_root_dir")
    assert not work.logger_metadatas

    trainer.loggers = [TensorBoardLogger(save_dir=tmpdir)]
    trainer.logger = trainer.loggers[0]
    tracker.setup(trainer=trainer, pl_module=Mock())
    assert work.log_dir == Path(tmpdir / "lightning_logs" / "version_0")
    assert len(work.logger_metadatas) == 1
    assert work.logger_metadatas[0] == {"class_name": "TensorBoardLogger"}

    # call setup a second time and the metadata length should not change
    tracker.setup(trainer=trainer, pl_module=Mock())
    assert len(work.logger_metadatas) == 1
