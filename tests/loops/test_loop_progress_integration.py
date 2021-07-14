from pytorch_lightning import Trainer
from tests.loops.test_loops import _collect_loop_progress


def test_loop_progress_integration():
    trainer = Trainer()
    # check no progresses are shared
    assert trainer.fit_loop.epoch_progress is not trainer.validate_loop.dataloader_progress
    assert trainer.validate_loop.dataloader_progress is not trainer.test_loop.dataloader_progress
    assert trainer.test_loop.dataloader_progress is not trainer.predict_loop.dataloader_progress
    # check the validation progresses are not shared
    assert trainer.fit_loop.epoch_loop.val_loop.dataloader_progress is not trainer.validate_loop.dataloader_progress
    # check recursive collection of progresses
    progresses = _collect_loop_progress(trainer.fit_loop)
    assert progresses["epoch_progress"] is trainer.fit_loop.epoch_progress
    assert progresses["epoch_loop"]["batch_progress"] is trainer.fit_loop.epoch_loop.batch_progress
    assert progresses["epoch_loop"]["val_loop"]["dataloader_progress"
                                                ] is trainer.fit_loop.epoch_loop.val_loop.dataloader_progress
    assert progresses["epoch_loop"]["val_loop"]["epoch_loop"][
        "batch_progress"] is trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress
